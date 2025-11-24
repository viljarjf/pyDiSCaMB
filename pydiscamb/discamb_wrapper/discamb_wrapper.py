import csv
import os, sys
from pathlib import Path
from typing import TYPE_CHECKING, List, Tuple, overload

from cctbx.array_family import flex

from pydiscamb._cpp_module import FCalcDerivatives, PythonInterface, TargetDerivatives
from pydiscamb.discamb_wrapper.factory_methods import FactoryMethodsMixin
from pydiscamb.discamb_wrapper.fcalc_method import FCalcMethod

if TYPE_CHECKING:
    from cctbx.xray.structure import structure
    from cctbx import miller


def _concat_scatterer_labels(xrs: "structure") -> str:
    return "".join(s.label for s in xrs.scatterers())

class DiscambWrapper(PythonInterface, FactoryMethodsMixin):

    atom_type_assignment: "dict[str, tuple[str, str]]"

    def __init__(
        self, xrs: "structure", method: FCalcMethod = FCalcMethod.IAM, **kwargs
    ):
        """
        Initialize a wrapper object for structure factor calculations using DiSCaMB.
        Pass an xray structure and either a FCalcMethod,
        or kwargs which are passed to  C++: :code:`discamb::SfCalculator::create`,
        found in discamb/Scattering/SfCalculator.h

        Notes
        -----
        The :code:`method` parameter is used for lookup of sensible default
        values to be sent to :code:`discamb::SfCalculator::create`.
        Any kwargs present will override the defaults.
        The defaults are e.g. inferring the scattering table from `xrs`,
        and using the default TAAM databank.
        See :code:`FCalcMethod.to_dict` for details.

        Parameters
        ----------
        xrs : structure
            Object defining the symmetry, atom positions, ADPs, occupancy ect.
        method : FCalcMethod, optional
            Which structure factor model to use, by default FCalcMethod.IAM
        """
        calculator_params = method.to_dict(xrs, kwargs)
        super().__init__(xrs, calculator_params)

        # Store some stuff from xrs
        self._atomstr = _concat_scatterer_labels(xrs)
        self._crystal_symmetry = xrs.crystal_symmetry()
        self._scatterer_flags = xrs.scatterer_flags()
        self._anomalous_flag = xrs.scatterers().count_anomalous() > 0

        # Get assignment info
        if calculator_params["model"] != "taam":
            # TODO accept aliases? like "matts" or "ubdb"
            self.atom_type_assignment = {
                atom.label: ("", "") for atom in xrs.scatterers()
            }
            return
        assignment_csv = Path(calculator_params["assignment csv"])
        assert assignment_csv.exists(), str(assignment_csv)
        with assignment_csv.open("r") as f:
            self.atom_type_assignment = {
                label: (atomtype, lcs)
                for label, atomtype, lcs in csv.reader(f, delimiter=";", strict=True)
            }
        if kwargs.get("assignment_csv") is None:
            # If we made a tempfile, we are responsible for cleaning it up,
            # as per the docs for tempfile.mkstemp
            try:
                os.remove(calculator_params["assignment csv"])
            except PermissionError:
                # Could not delete the file. Should be fine, it's in the tmp dir.
                # As of May 2025, this error is consistently thrown in the windows testrunners of cctbx
                pass

    def show_atom_type_assignment(self, log=sys.stdout):
        ata = self.atom_type_assignment

        multipolar = {
            k: v for k, v in ata.items() if (v[0] != "" and not v[0].endswith("000"))
        }
        spherical_multipolar = {k: v for k, v in ata.items() if v[0].endswith("000")}
        spherical_iam = {k: v for k, v in ata.items() if (v[0] == "" and v[1] == "")}

        n_atoms = len(ata)
        n_multipolar = len(multipolar.keys())
        n_spherical_multipolar = len(spherical_multipolar.keys())
        n_spherical_iam = len(spherical_iam.keys())

        assert n_multipolar + n_spherical_multipolar + n_spherical_iam == n_atoms

        n_spherical_multipolar_water = sum(
            1 for k in spherical_multipolar if "HOH" in k
        )
        n_spherical_iam_water = sum(1 for k in spherical_iam if "HOH" in k)

        print("Total number of atoms:", n_atoms, file=log)
        print("Number of atoms by assignment:\nTAAM atom type:", n_multipolar, file=log)
        print(
            "IAM (Slater):",
            n_spherical_multipolar,
            f"(in HOH: {n_spherical_multipolar_water})",
            file=log,
        )
        print(
            "IAM (Gaussian):",
            n_spherical_iam,
            file=log,
        )

        if n_spherical_multipolar > 0:
            print(
                "\nThe following atoms have spherical multipolar parameters:", file=log
            )
            for k in spherical_multipolar.keys():
                print(k.split('"')[1], file=log)

        if n_spherical_iam > 0:
            print("\nThe following atoms use regulard IAM:", file=log)
            for k in spherical_iam.keys():
                print(k.split('"')[1], file=log)

    def update_structure(self, xrs: "structure"):
        if self._atomstr != _concat_scatterer_labels(
            xrs
        ) or not self._crystal_symmetry.is_identical_symmetry(xrs.crystal_symmetry()):
            raise ValueError(
                "Incompatible structures. "
                "Must have same scatterers in the same order, and same unit cell"
            )
        self._scatterer_flags = xrs.scatterer_flags()
        self._anomalous_flag = xrs.scatterers().count_anomalous() > 0
        super().update_structure(xrs)

    def set_d_min(self, d_min: float):
        from cctbx import miller

        miller_set = miller.build_set(
            self._crystal_symmetry, self._anomalous_flag, d_min=d_min
        )
        self.set_indices(miller_set.indices())

    ## Annotations
    @overload
    def f_calc(self, miller_set: None) -> flex.complex_double: ...

    @overload
    def f_calc(self, miller_set: float) -> flex.complex_double: ...

    @overload
    def f_calc(self, miller_set: "miller.set") -> "miller.array": ...

    @overload
    def d_f_calc_hkl_d_params(self, h: Tuple[int, int, int]) -> FCalcDerivatives: ...

    @overload
    def d_f_calc_hkl_d_params(self, h: int, k: int, l: int) -> FCalcDerivatives: ...

    @overload
    def d_target_d_params(self, d_target_d_f_calc: "miller.array") -> flex.double: ...

    @overload
    def d_target_d_params(self, d_target_d_f_calc: list) -> List[TargetDerivatives]: ...

    ## Implementations
    def f_calc(self, miller_set=None):
        if miller_set is None:
            return flex.complex_double(super().f_calc())
        if isinstance(miller_set, (int, float)):  # d_min
            self.set_d_min(miller_set)
            return self.f_calc()
        from cctbx import miller

        if not isinstance(miller_set, miller.set):
            raise ValueError("`miller_set` must be of type `cctbx.miller.set")

        self.set_indices(miller_set.indices())
        res = self.f_calc()
        return miller_set.array(data=res)

    def d_f_calc_hkl_d_params(self, h, k=None, l=None):
        if isinstance(h, tuple):
            assert len(h) == 3
            h, k, l = h
            return self.d_f_calc_hkl_d_params(h, k, l)
        return super().d_f_calc_hkl_d_params(h, k, l)

    def d_target_d_params(self, d_target_d_f_calc):
        if isinstance(d_target_d_f_calc, list):
            return super().d_target_d_params(d_target_d_f_calc)

        self.set_indices(d_target_d_f_calc.indices())
        res = self.d_target_d_params(list(d_target_d_f_calc.data()))

        # Pack selected gradients
        size = 0
        for s in self._scatterer_flags:
            size += 3 * s.grad_site()
            size += 1 * s.grad_u_iso()
            size += 6 * s.grad_u_aniso()
            size += 1 * s.grad_occupancy()
        out = flex.double(size, 0)
        o_ind = 0
        for s_ind, s in enumerate(self._scatterer_flags):
            if s.grad_site():
                out[o_ind] = res[s_ind].site_derivatives[0]
                o_ind += 1
                out[o_ind] = res[s_ind].site_derivatives[1]
                o_ind += 1
                out[o_ind] = res[s_ind].site_derivatives[2]
                o_ind += 1
            if s.grad_u_iso():
                out[o_ind] = res[s_ind].adp_derivatives[0]
                o_ind += 1
            if s.grad_u_aniso():
                out[o_ind] = res[s_ind].adp_derivatives[0]
                o_ind += 1
                out[o_ind] = res[s_ind].adp_derivatives[1]
                o_ind += 1
                out[o_ind] = res[s_ind].adp_derivatives[2]
                o_ind += 1
                out[o_ind] = res[s_ind].adp_derivatives[3]
                o_ind += 1
                out[o_ind] = res[s_ind].adp_derivatives[4]
                o_ind += 1
                out[o_ind] = res[s_ind].adp_derivatives[5]
                o_ind += 1
            if s.grad_occupancy():
                out[o_ind] = res[s_ind].occupancy_derivatives
                o_ind += 1
        assert o_ind == size
        return out
