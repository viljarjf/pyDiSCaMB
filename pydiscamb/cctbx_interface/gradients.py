from cctbx.array_family import flex
from cctbx.xray.structure_factors.gradients_base import gradients_base
from cctbx.xray.structure_factors.gradients_direct import gradients_direct

from pydiscamb.cctbx_interface.phil_scope import scope_to_taam_dict
from pydiscamb.discamb_wrapper import DiscambWrapper, FCalcMethod


class gradients_taam(gradients_direct):
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
        extra_params=None,
    ):
        if extra_params is None:
            extra_params_dict = {}
        else:
            extra_params_dict = scope_to_taam_dict(extra_params)

        gradients_base.__init__(
            self,
            manager,
            xray_structure,
            miller_set,
            algorithm="taam",
        )
        self._results = CctbxGradientsResult(
            self.xray_structure(),
            miller_set,
            d_target_d_f_calc,
            n_parameters,
            FCalcMethod.TAAM,
            **extra_params_dict,
        )
        self.d_target_d_site_cart_was_used = False
        self.d_target_d_u_cart_was_used = False


class CctbxGradientsResult:
    def __init__(
        self, xrs, miller_set, d_target_d_f_calc, n_parameters, method, **kwargs
    ):
        w = DiscambWrapper(xrs, method, **kwargs)
        w.set_indices(miller_set.indices())

        self._d_target_d_site_frac = flex.vec3_double(
            xrs.scatterers().size(), (0, 0, 0)
        )
        self._d_target_d_u_iso = flex.double(xrs.scatterers().size(), 0)
        self._d_target_d_u_star = flex.sym_mat3_double(
            xrs.scatterers().size(), (0, 0, 0, 0, 0, 0)
        )
        self._d_target_d_occupancy = flex.double(xrs.scatterers().size(), 0)
        self._d_target_d_fp = flex.double(xrs.scatterers().size(), 0)
        self._d_target_d_fdp = flex.double(xrs.scatterers().size(), 0)

        grads = w.selected_d_target_d_params(
            list(d_target_d_f_calc),
            any(s.grad_site() for s in xrs.scatterer_flags()),
            any(s.grad_u_iso() or s.grad_u_aniso() for s in xrs.scatterer_flags()),
            any(s.grad_occupancy() for s in xrs.scatterer_flags()),
            any(s.grad_fp() or s.grad_fdp() for s in xrs.scatterer_flags()),
        )
        for i, s in enumerate(xrs.scatterer_flags()):
            if s.grad_site():
                self._d_target_d_site_frac[i] = grads[i].site_derivatives
            if s.grad_u_iso() and s.use_u_iso():
                self._d_target_d_u_iso[i] = grads[i].adp_derivatives[0]
            if s.grad_u_aniso() and s.use_u_aniso():
                self._d_target_d_u_star[i] = grads[i].adp_derivatives
            if s.grad_occupancy():
                self._d_target_d_occupancy[i] = grads[i].occupancy_derivatives

        self._packed = flex.double(n_parameters, 0)
        if n_parameters <= 0:
            return

        packed_ind = 0
        for i, s in enumerate(xrs.scatterer_flags()):
            if s.grad_site():
                for j in range(3):
                    self._packed[packed_ind] = grads[i].site_derivatives[j]
                    packed_ind += 1
            if s.grad_u_iso():
                self._packed[packed_ind] = grads[i].adp_derivatives[0]
                packed_ind += 1
            if s.grad_u_aniso():
                for j in range(6):
                    self._packed[packed_ind] = grads[i].adp_derivatives[j]
                    packed_ind += 1
            if s.grad_occupancy():
                self._packed[packed_ind] = grads[i].occupancy_derivatives
                packed_ind += 1
        assert packed_ind == n_parameters

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
