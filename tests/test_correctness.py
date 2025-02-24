import pydiscamb

from cctbx.development import random_structure
from cctbx.sgtbx import space_group_info
from cctbx.array_family import flex
from cctbx.eltbx import xray_scattering

import pytest


def compare_structure_factors(f_calc_a, f_calc_b):
    f_calc_a = flex.abs(f_calc_a)
    f_calc_b = flex.abs(f_calc_b)
    scale = flex.sum(f_calc_a * f_calc_b) / flex.sum(f_calc_b * f_calc_b)
    num = flex.sum(flex.abs(f_calc_a - scale * f_calc_b))
    den = flex.sum(flex.abs(f_calc_a + scale * f_calc_b))
    diff = flex.abs(f_calc_a - f_calc_b)
    return num / den * 2 * 100.0  # , flex.mean(diff), flex.max(diff)


def get_random_crystal(
    space_group: int,
    with_adps: bool,
    with_occupancy: bool,
    with_anomalous: str,
    atoms: str,
    scattering_table: str,
) -> random_structure.xray_structure:
    # We use bool() on some of the params instead of true/false to make pytest output more readable
    if atoms == "single weak":
        elements = ["C"]
    if atoms == "single strong":
        elements = ["Au"]
    elif atoms == "many weak":
        elements = ["C", "O", "N", "H"] * 10
    elif atoms == "many strong":
        elements = ["Au", "Ag", "Cd", "Fe"] * 10
    elif atoms == "mixed strength":
        elements = ["Au", "Cd", "C", "O", "H"] * 10
    group = space_group_info(space_group)
    xrs = random_structure.xray_structure(
        space_group_info=group,
        elements=elements,
        general_positions_only=False,
        use_u_iso=with_adps == "random u_iso",
        use_u_aniso=with_adps == "random u_aniso",
        random_u_iso=bool(with_adps),
        random_occupancy=bool(with_occupancy),
    )
    if "fprime" in with_anomalous:
        xrs.shake_fps()
    if "fdoubleprime" in with_anomalous:
        xrs.shake_fdps()
    xrs.scattering_type_registry(table=scattering_table)
    return xrs


def get_IAM_correctness_score(
    xrs: random_structure.xray_structure, d_min: float = 2
) -> float:
    fcalc_cctbx = xrs.structure_factors(algorithm="direct", d_min=d_min).f_calc().data()
    fcalc_discamb = pydiscamb.calculate_structure_factors_IAM(xrs, d_min)

    fcalc_discamb = flex.complex_double(fcalc_discamb)
    score = compare_structure_factors(fcalc_cctbx, fcalc_discamb)
    return score


@pytest.mark.veryslow
@pytest.mark.slow
@pytest.mark.parametrize("space_group", range(1, 231))
def test_IAM_correctness_random_crystal(
    space_group: int,
):
    from itertools import product

    for args in product(
        ["random u_iso", "random u_aniso", None],
        ["random occupancy", None],
        ["no anomalous", "fprime", "fdoubleprime", "fprime + fdoubleprime"],
        ["single weak", "single strong", "many weak", "many strong", "mixed strength"],
        ["it1992", "wk1995", "electron"],
    ):
        xrs = get_random_crystal(space_group, *args)
        score = get_IAM_correctness_score(xrs)
        # Use 0.05% as threshold
        assert score < 0.0005


@pytest.mark.slow
def test_lysozyme_high_res(lysozyme):
    score = get_IAM_correctness_score(lysozyme, d_min=0.5)
    assert score < 0.0002


@pytest.mark.parametrize(
    "method",
    [
        pydiscamb.FCalcMethod.IAM,
        pydiscamb.FCalcMethod.TAAM,
    ],
)
def test_indices_order(method, tyrosine):
    from random import randint, shuffle

    # 1000 random indices
    inds_1 = [
        (
            randint(-20, 20),
            randint(-20, 20),
            randint(-20, 20),
        )
        for _ in range(1000)
    ]
    # Shuffle these indices, and keep the way it was shuffled
    shuffle_inds = list(range(len(inds_1)))
    shuffle(shuffle_inds)
    inds_2 = [inds_1[i] for i in shuffle_inds]

    # Calculate fcalc with both sets of indices
    w1 = pydiscamb.DiscambWrapper(tyrosine, method)
    w1.set_indices(inds_1)
    fc1 = w1.f_calc()

    w2 = pydiscamb.DiscambWrapper(tyrosine, method)
    w2.set_indices(inds_2)
    fc2 = w2.f_calc()

    # Compare
    for i, idx in enumerate(shuffle_inds):
        assert inds_1[idx] == inds_2[i]
        assert fc1[idx] == fc2[i]


class TestCustomTable:
    @pytest.fixture
    def elements(self):
        return ["C", "Fe"]

    @pytest.fixture
    def a(self):
        return [
            [0.0893, 0.2563, 0.7570, 1.0487, 0.3575],
            [0.1893, 0.2563, 0.7570, 1.0487, 0.3575],
        ]

    @pytest.fixture
    def b(self):
        return [
            [0.2465, 1.7100, 6.4094, 18.6113, 50.2523],
            [0.1465, 1.7100, 6.4094, 18.6113, 50.2523],
        ]

    @pytest.fixture
    def c(self):
        return [
            0.0,
            0.0,
        ]

    @pytest.fixture
    def xrs(self, elements, a, b, c):
        _xrs = random_structure.xray_structure(
            space_group_info=space_group_info(19),
            elements=elements * 3,
        )
        scattering_dictionary = {
            el: xray_scattering.gaussian(a, b, c)
            for el, a, b, c in zip(elements, a, b, c)
        }

        _xrs.scattering_type_registry(custom_dict=scattering_dictionary)
        return _xrs

    def test_custom_table(self, xrs, elements, a, b, c):
        sf_1 = xrs.structure_factors(d_min=2, algorithm="direct").f_calc()

        indices = [[int(h), int(k), int(l)] for h, k, l in sf_1.indices()]

        sf_2 = pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
            xrs, indices, elements, a, b, c
        )
        score = compare_structure_factors(sf_1.data(), flex.complex_double(sf_2))
        assert score < 1e-4

    def test_missing_element(self, xrs, a, b, c):
        elements = ["does not exist"] * len(a)
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, [[0, 0, 0]], elements, a, b, c
            )

    def test_wrong_length(self, xrs, elements, a, b, c):
        i = [[0, 0, 0]]
        # Too few
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, i, elements[:-1], a, b, c
            )
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, i, elements, a[:-1], b, c
            )
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, i, elements, a, b[:-1], c
            )
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, i, elements, a, b, c[:-1]
            )

        # Too many
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, i, elements + [elements[-1]], a, b, c
            )
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, i, elements, a + [a[-1]], b, c
            )
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, i, elements, a, b + [b[-1]], c
            )
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, i, elements, a, b, c + [c[-1]]
            )

    def test_wrong_length_gaussian_parameters(self, xrs, elements, a, b, c):
        i = [[0, 0, 0]]
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, i, elements, [ai[:-1] for ai in a], b, c
            )
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, i, elements, a, [bi[:-1] for bi in b], c
            )

    def test_hkl(self, xrs, elements, a, b, c):
        # Wrong number of elements
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, [[0, 0, 0], [0, 0]], elements, a, b, c
            )
        with pytest.raises(AssertionError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, [[0, 0, 0], [0, 0, 0, 0]], elements, a, b, c
            )
        # Wrong type
        with pytest.raises(TypeError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, [0, 0, 0], elements, a, b, c
            )
        with pytest.raises(TypeError):
            pydiscamb.wrapper_tests.f_calc_custom_gaussian_parameters(
                xrs, [(0.2, 0.3, 0.0)], elements, a, b, c
            )


if __name__ == "__main__":
    space_group: int = 19
    with_adps: bool = False
    with_occupancy: bool = False
    with_anomalous: str = ""
    atoms: str = "single weak"
    scattering_table: str = "electron"
    xrs = get_random_crystal(
        space_group, with_adps, with_occupancy, with_anomalous, atoms, scattering_table
    )
    score = get_IAM_correctness_score(xrs)
