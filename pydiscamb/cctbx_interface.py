from cctbx.xray.structure_factors.gradients_base import gradients_base
from cctbx.xray.structure_factors.gradients_direct import gradients_direct
from cctbx.xray.structure_factors.manager import managed_calculation_base
from cctbx.xray.structure_factors.from_scatterers_direct import from_scatterers_direct
from cctbx.array_family import flex

from pydiscamb.discamb_wrapper import DiscambWrapper, FCalcMethod


class gradients_discamb(gradients_direct):
    # TODO consider moving this class to cctbx
    # TODO add timings
    def __init__(
        self,
        xray_structure,
        u_iso_refinable_params,
        miller_set,
        d_target_d_f_calc,
        n_parameters,
        manager=None,
        cos_sin_table=False,
    ):
        gradients_base.__init__(
            self,
            manager,
            xray_structure,
            miller_set,
            algorithm="discamb",
        )
        self._results = CctbxGradientsResult(
            self.xray_structure(), miller_set, d_target_d_f_calc
        )
        self.d_target_d_site_cart_was_used = False
        self.d_target_d_u_cart_was_used = False

class from_scatterers_discamb(from_scatterers_direct):

  def __init__(self, xray_structure,
                     miller_set,
                     manager=None,
                     cos_sin_table=False,
                     algorithm="discamb"):
    # TODO add timings
    managed_calculation_base.__init__(self,
      manager, xray_structure, miller_set, algorithm="discamb")
    self._results = CctbxStructureFactorsResult(xray_structure, miller_set)



class CctbxGradientsResult:
    def __init__(self, xrs, miller_set, d_target_d_f_calc):
        w = DiscambWrapper(xrs, FCalcMethod.TAAM)
        w.set_indices(miller_set.indices())
        self._grads = w.d_target_d_params(list(d_target_d_f_calc))


        self._d_target_d_site_frac = flex.vec3_double(d_target_d_f_calc.size(), (0, 0, 0))
        self._d_target_d_u_iso = flex.double(d_target_d_f_calc.size(), 0)
        self._d_target_d_u_star = flex.sym_mat3(d_target_d_f_calc.size(), (0, 0, 0, 0, 0, 0))
        self._d_target_d_occupancy = flex.double(d_target_d_f_calc.size(), 0)
        self._d_target_d_fp = flex.double(d_target_d_f_calc.size(), 0)
        self._d_target_d_fdp = flex.double(d_target_d_f_calc.size(), 0)

        size = 0
        for s in self.xray_structure().scatterer_flags():
            size += 3 * s.grad_site()
            size += 1 * s.grad_u_iso()
            size += 6 * s.grad_u_aniso()
            size += 1 * s.grad_occupancy()
        self._packed = flex.double(size, 0)

        packed_ind = 0
        for i, s in enumerate(self.xray_structure().scatterer_flags()):
            if s.grad_site():
                for j in range(3):
                    self._d_target_d_site_frac[i] = self._grads[i].site_derivatives[j]
                    self._packed[packed_ind] = self._grads[i].site_derivatives[j]
                    packed_ind += 1
            if s.grad_u_iso():
                self._d_target_d_u_iso[i] = self._grads[i].adp_derivatives[0]
                self._packed[packed_ind] = self._grads[i].adp_derivatives[0]
                packed_ind += 1
            if s.grad_u_aniso():
                for j in range(6):
                    self._d_target_d_u_star[i] = self._grads[i].adp_derivatives[j]
                    self._packed[packed_ind] = self._grads[i].adp_derivatives[j]
                    packed_ind += 1
            if s.grad_occupancy():
                self._d_target_d_occupancy[i] = self._grads[i].occupancy_derivatives
                self._packed[packed_ind] = self._grads[i].occupancy_derivatives
                packed_ind += 1
        assert packed_ind == size

    def d_target_d_site_frac(self):
        return self._d_target_d_site_frac
    def d_target_d_u_iso(self):
        return self._d_target_d_u_iso
    def d_target_d_u_star(self):
        return self._d_target_d_u_star
    def d_target_d_occupancy(self):
        return self._d_target_d_occupancy
    def d_target_d_fp(self):
        return self._d_target_d_fp
    def d_target_d_fdp(self):
        return self._d_target_d_fdp
    def packed(self):
        return self._packed


class CctbxStructureFactorsResult:
    def __init__(self, xrs, miller_set):
        w = DiscambWrapper(xrs, FCalcMethod.TAAM)
        w.set_indices(miller_set.indices())
        self._fcalc = w.f_calc()

    def f_calc(self):
        return self._fcalc