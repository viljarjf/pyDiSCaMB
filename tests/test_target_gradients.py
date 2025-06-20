import mmtbx.f_model
import numpy as np
import pytest
from cctbx.array_family import flex
from pydiscamb import DiscambWrapper, FCalcMethod
from pydiscamb._cpp_module import TargetDerivatives


class TestTargetGradients:
    def test_simple(self, random_structure):

        # Use f_calc as f_obs
        d_min = 2
        f_obs = abs(random_structure.structure_factors(d_min=d_min).f_calc())

        # Set up model
        random_structure.shake_sites_in_place(rms_difference=0.1)
        scatterers = random_structure.scatterers()
        model = mmtbx.f_model.manager(f_obs=f_obs, xray_structure=random_structure)
        model.sfg_params.algorithm = "direct"
        target = model.target_functor()(compute_gradients=True)

        # Get target derivatives
        d_target_d_fcalc = target.d_target_d_f_calc_work()

        # Calculate derivatives with discamb
        w = DiscambWrapper(random_structure)
        w.set_indices(d_target_d_fcalc.indices())
        discamb_result = w.d_target_d_params(list(d_target_d_fcalc.data()))

        ### Check equality
        iselection = flex.bool(scatterers.size(), True).iselection()

        # Site
        scatterers.flags_set_grads(state=False)
        scatterers.flags_set_grad_site(iselection=iselection)
        g = flex.vec3_double(target.gradients_wrt_atomic_parameters().packed())
        expected = np.array(g)
        actual = np.array([res.site_derivatives for res in discamb_result])
        assert pytest.approx(expected, abs=1e-5, rel=1e-5) == actual

        # ADP
        scatterers.flags_set_grads(state=False)
        scatterers.flags_set_grad_u_iso(iselection=iselection)
        g = target.gradients_wrt_atomic_parameters().packed()
        assert (
            pytest.approx(np.array(g), abs=1e-5, rel=1e-5)
            == np.array([res.adp_derivatives for res in discamb_result]).flatten()
        )

        # Occupancy
        scatterers.flags_set_grads(state=False)
        scatterers.flags_set_grad_occupancy(iselection=iselection)
        g = target.gradients_wrt_atomic_parameters().packed()
        assert pytest.approx(list(g), abs=1e-5, rel=1e-5) == [
            res.occupancy_derivatives for res in discamb_result
        ]

    def test_u_aniso(self, random_structure_u_aniso):

        # Use f_calc as f_obs
        d_min = 2
        f_obs = abs(random_structure_u_aniso.structure_factors(d_min=d_min).f_calc())

        # Set up model
        random_structure_u_aniso.shake_sites_in_place(rms_difference=0.1)
        scatterers = random_structure_u_aniso.scatterers()
        model = mmtbx.f_model.manager(
            f_obs=f_obs, xray_structure=random_structure_u_aniso
        )
        model.sfg_params.algorithm = "direct"
        target = model.target_functor()(compute_gradients=True)

        # Get target derivatives
        d_target_d_fcalc = target.d_target_d_f_calc_work()

        # Calculate derivatives with discamb
        w = DiscambWrapper(random_structure_u_aniso)
        w.set_indices(d_target_d_fcalc.indices())
        discamb_result = w.d_target_d_params(list(d_target_d_fcalc.data()))

        ### Check equality
        iselection = flex.bool(scatterers.size(), True).iselection()
        scatterers.flags_set_grads(state=False)
        scatterers.flags_set_grad_u_aniso(iselection=iselection)
        g = target.gradients_wrt_atomic_parameters().packed()
        assert (
            pytest.approx(np.array(g), abs=1e-5, rel=1e-5)
            == np.array([res.adp_derivatives for res in discamb_result]).flatten()
        )

    def test_incorrect_size(self, random_structure):
        d_min = 2
        f_obs = abs(random_structure.structure_factors(d_min=d_min).f_calc())

        # Set up model
        random_structure.shake_sites_in_place(rms_difference=0.1)
        model = mmtbx.f_model.manager(f_obs=f_obs, xray_structure=random_structure)
        model.sfg_params.algorithm = "direct"
        target = model.target_functor()(compute_gradients=True)

        # Get target derivatives
        d_target_d_fcalc = target.d_target_d_f_calc_work()

        # Calculate derivatives with discamb
        w = DiscambWrapper(random_structure)
        w.set_indices(d_target_d_fcalc.indices())

        with pytest.raises(AssertionError):
            # Supply incorrect length list
            w.d_target_d_params(list(d_target_d_fcalc.data())[1:])

    def test_result_type(self, tyrosine):

        f_obs = abs(tyrosine.structure_factors(d_min=2).f_calc())
        model = mmtbx.f_model.manager(f_obs=f_obs, xray_structure=tyrosine)
        target = model.target_functor()(compute_gradients=True)
        d_target_d_f_calc = target.d_target_d_f_calc_work()

        iselection = flex.bool(tyrosine.scatterers().size(), True).iselection()
        tyrosine.scatterers().flags_set_grads(state=False)
        tyrosine.scatterers().flags_set_grad_site(iselection)

        w = DiscambWrapper(tyrosine)
        assert isinstance(w.d_target_d_params(d_target_d_f_calc), flex.double)
        assert isinstance(w.d_target_d_params(list(d_target_d_f_calc.data())), list)
        assert isinstance(
            w.d_target_d_params(list(d_target_d_f_calc.data()))[0], TargetDerivatives
        )

    def test_no_flags(self, tyrosine):

        f_obs = abs(tyrosine.structure_factors(d_min=2).f_calc())
        model = mmtbx.f_model.manager(f_obs=f_obs, xray_structure=tyrosine)
        target = model.target_functor()(compute_gradients=True)
        d_target_d_f_calc = target.d_target_d_f_calc_work()

        w = DiscambWrapper(tyrosine)
        res = w.d_target_d_params(d_target_d_f_calc)
        assert res.size() == 0

    @pytest.mark.slow
    def test_correctness_all_flags(self, lysozyme):
        lysozyme.scatterers().convert_to_anisotropic(lysozyme.unit_cell())
        f_obs = abs(lysozyme.structure_factors(d_min=1).f_calc())

        lysozyme.shake_sites_in_place(rms_difference=0.1)
        lysozyme.shake_adp()
        lysozyme.shake_occupancies()

        iselection = flex.bool(lysozyme.scatterers().size(), True).iselection()
        lysozyme.scatterers().flags_set_grads(state=False)
        lysozyme.scatterers().flags_set_grad_site(iselection)
        lysozyme.scatterers().flags_set_grad_occupancy(iselection)
        lysozyme.scatterers().flags_set_grad_u_aniso(iselection)

        model = mmtbx.f_model.manager(f_obs=f_obs, xray_structure=lysozyme)
        model.sfg_params.algorithm = "direct"
        target = model.target_functor()(compute_gradients=True)
        d_target_d_f_calc = target.d_target_d_f_calc_work()

        w = DiscambWrapper(lysozyme)
        res = w.d_target_d_params(d_target_d_f_calc)

        # x, y, z, U11, U22, U33, U12, U13, U23, occupancy = 10
        assert res.size() == lysozyme.scatterers().size() * 10

        # Compare with cctbx
        exp = target.gradients_wrt_atomic_parameters().packed()
        assert pytest.approx(list(res), abs=1e-6, rel=1e-5) == list(exp)

    @pytest.mark.slow
    def test_multiple_assorted_flags(self, lysozyme):
        xrs = lysozyme

        import random

        random.seed(0)
        flex.set_random_seed(0)

        f_obs = abs(xrs.structure_factors(d_min=2).f_calc())

        xrs.shake_sites_in_place(rms_difference=0.1)
        xrs.shake_adp()
        xrs.shake_occupancies()

        xrs.scatterers().flags_set_grads(state=False)
        iselection = flex.bool(
            [
                random.randint(0, 1) if s.flags.use_u_iso() else 0
                for s in xrs.scatterers()
            ]
        ).iselection()
        xrs.scatterers().flags_set_grad_u_iso(iselection)
        iselection = flex.bool(
            [
                random.randint(0, 1) if s.flags.use_u_aniso() else 0
                for s in xrs.scatterers()
            ]
        ).iselection()
        xrs.scatterers().flags_set_grad_u_aniso(iselection)
        iselection = flex.bool(
            [random.randint(0, 1) for _ in range(xrs.scatterers().size())]
        ).iselection()
        xrs.scatterers().flags_set_grad_site(iselection)
        iselection = flex.bool(
            [random.randint(0, 1) for _ in range(xrs.scatterers().size())]
        ).iselection()
        xrs.scatterers().flags_set_grad_occupancy(iselection)

        model = mmtbx.f_model.manager(f_obs=f_obs, xray_structure=xrs)
        model.sfg_params.algorithm = "direct"
        target = model.target_functor()(compute_gradients=True)
        d_target_d_f_calc = target.d_target_d_f_calc_work()

        w = DiscambWrapper(xrs)
        res = w.d_target_d_params(d_target_d_f_calc)

        exp = target.gradients_wrt_atomic_parameters().packed()

        assert exp.size() == res.size()
        assert pytest.approx(list(res), abs=1e-6, rel=1e-5) == list(exp)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "algorithm",
        [
            pytest.param(
                "standard",
                marks=pytest.mark.xfail(
                    reason="Only works on macromolecule implementation for now"
                ),
            ),
            "non-standard or whatever. Should be macromolecule",
        ],
    )
    @pytest.mark.parametrize("site", [True, False])
    @pytest.mark.parametrize("adp", [True, False])
    @pytest.mark.parametrize("oc", [True, False])
    @pytest.mark.parametrize("fp", [True, False])
    def test_discamb_flags_taam(self, tyrosine, algorithm, site, adp, oc, fp):
        xrs = tyrosine

        import random

        random.seed(0)
        flex.set_random_seed(0)

        f_obs = abs(xrs.structure_factors(d_min=2).f_calc())
        xrs.shake_sites_in_place(rms_difference=0.1)
        model = mmtbx.f_model.manager(f_obs=f_obs, xray_structure=xrs)
        target = model.target_functor()(compute_gradients=True)
        d_target_d_f_calc = target.d_target_d_f_calc_work()

        w = DiscambWrapper(xrs, method=FCalcMethod.TAAM, algorithm=algorithm)
        w.set_indices(d_target_d_f_calc.indices())
        grads = w.selected_d_target_d_params(
            list(d_target_d_f_calc.data()),
            site,
            adp,
            oc,
            fp,
        )

        if site:
            assert any(any(i != 0 for i in g.site_derivatives) for g in grads)
        else:
            assert all(all(i == 0 for i in g.site_derivatives) for g in grads)

        if adp:
            assert any(any(i != 0 for i in g.adp_derivatives) for g in grads)
        else:
            assert all(all(i == 0 for i in g.adp_derivatives) for g in grads)

        if oc:
            assert any(g.occupancy_derivatives != 0 for g in grads)
        else:
            assert all(g.occupancy_derivatives == 0 for g in grads)

        # if fp:
        #     assert all(g.anomalous for g in grads)

        if site and adp and oc and algorithm == "standard":
            assert False, "All flags are set when algorithm is standard"

    @pytest.mark.xfail(reason="IAM does not support gradient flags yet")
    @pytest.mark.slow
    @pytest.mark.parametrize("site", [True, False])
    @pytest.mark.parametrize("adp", [True, False])
    @pytest.mark.parametrize("oc", [True, False])
    @pytest.mark.parametrize("fp", [True, False])
    def test_discamb_flags_iam(self, tyrosine, site, adp, oc, fp):
        xrs = tyrosine

        import random

        random.seed(0)
        flex.set_random_seed(0)

        f_obs = abs(xrs.structure_factors(d_min=2).f_calc())
        xrs.shake_sites_in_place(rms_difference=0.1)
        model = mmtbx.f_model.manager(f_obs=f_obs, xray_structure=xrs)
        target = model.target_functor()(compute_gradients=True)
        d_target_d_f_calc = target.d_target_d_f_calc_work()

        w = DiscambWrapper(xrs)
        w.set_indices(d_target_d_f_calc.indices())
        grads = w.selected_d_target_d_params(
            list(d_target_d_f_calc.data()),
            site,
            adp,
            oc,
            fp,
        )

        if site:
            assert any(any(i != 0 for i in g.site_derivatives) for g in grads)
        else:
            assert all(all(i == 0 for i in g.site_derivatives) for g in grads)

        if adp:
            assert any(any(i != 0 for i in g.adp_derivatives) for g in grads)
        else:
            assert all(all(i == 0 for i in g.adp_derivatives) for g in grads)

        if oc:
            assert any(g.occupancy_derivatives != 0 for g in grads)
        else:
            assert all(g.occupancy_derivatives == 0 for g in grads)

        # if fp:
        #     assert all(g.anomalous for g in grads)

        if site and adp and oc:
            assert False, "Default behaviour is all grads, we xfail it"

    @pytest.mark.slow
    def test_stale(self, lysozyme):
        """Fcalc is cached when computing gradients"""
        w = DiscambWrapper(lysozyme)

        f_obs = abs(lysozyme.structure_factors(d_min=1).f_calc())

        lysozyme.shake_sites_in_place(rms_difference=0.1)
        lysozyme.shake_adp()
        lysozyme.shake_occupancies()

        model = mmtbx.f_model.manager(f_obs=f_obs, xray_structure=lysozyme)
        target = model.target_functor()(compute_gradients=True)
        d_target_d_f_calc = target.d_target_d_f_calc_work()

        from time import perf_counter

        w.set_indices(d_target_d_f_calc.indices())
        start_slow = perf_counter()
        w.f_calc()
        end_slow = perf_counter()

        w.update_structure(lysozyme)  # make it stale
        w.d_target_d_params(d_target_d_f_calc)

        start_fast = perf_counter()
        w.f_calc()
        end_fast = perf_counter()

        time_slow = end_slow - start_slow
        time_fast = end_fast - start_fast
        assert time_fast < time_slow / 10
