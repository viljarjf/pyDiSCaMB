import pytest

from pydiscamb import DiscambWrapper, FCalcMethod
from pydiscamb.taam_parameters import is_MATTS_installed, get_TAAM_databanks

def test_init(tyrosine):
    w = DiscambWrapper(tyrosine, FCalcMethod.TAAM)


def test_f_calc(tyrosine):
    w = DiscambWrapper(tyrosine, FCalcMethod.TAAM)
    fc = w.f_calc(5.0)
    assert isinstance(fc[0], complex)


@pytest.mark.parametrize(
    ["table", "expected_R"], [("electron", 0.15), ("wk1995", 0.04)]
)
def test_f_calc_approx_IAM(tyrosine, table, expected_R):
    tyrosine.scattering_type_registry(table=table)
    iam = DiscambWrapper(tyrosine, FCalcMethod.IAM)
    taam = DiscambWrapper(tyrosine, FCalcMethod.TAAM)

    d_min = 2

    fc_iam = iam.f_calc(d_min)
    fc_taam = taam.f_calc(d_min)

    R = sum(abs(abs(a) - abs(b)) for a, b in zip(fc_iam, fc_taam)) / sum(
        abs(a) for a in fc_taam
    )
    assert R < expected_R


def test_from_parameters(tyrosine):
    wrapper = DiscambWrapper(
        tyrosine,
        model="taam",
        unit_cell_charge=1000,
        scale=True,
    )
    wrapper.set_d_min(2)
    fc = wrapper.f_calc()


def test_logging(tyrosine, tmp_path):
    wrapper = DiscambWrapper(
        tyrosine,
        model="matts",
        assignment_info=str(tmp_path / "assignment.txt"),
        parameters_info=str(tmp_path / "parameters.log"),
        multipole_cif=str(tmp_path / "structure.cif"),
        unit_cell_charge=0,
        scale=False,
    )
    assert (tmp_path / "assignment.txt").exists()
    assert (tmp_path / "parameters.log").exists()
    assert (tmp_path / "structure.cif").exists()


def test_unit_cell_charge_scaling(tyrosine):
    w1 = DiscambWrapper(
        tyrosine,
        model="matts",
        unit_cell_charge=-1000,
        scale=True,
    )
    w2 = DiscambWrapper(
        tyrosine,
        model="matts",
        unit_cell_charge=1000,
        scale=True,
    )
    assert not pytest.approx(w1.f_calc(2)) == w2.f_calc(2)


def test_unit_cell_charge_scaling_off(tyrosine):
    w1 = DiscambWrapper(
        tyrosine,
        model="matts",
        unit_cell_charge=1000,
        scale=False,
    )
    w2 = DiscambWrapper(
        tyrosine,
        model="matts",
        unit_cell_charge=0,
        scale=False,
    )
    assert pytest.approx(w1.f_calc(2)) == w2.f_calc(2)


def test_invalid_bank(tyrosine):
    with pytest.raises(
        RuntimeError,
        match="Problem with accessing/openning file 'non-existent bank file' for reading UBDB type bank",
    ):
        w = DiscambWrapper(
            tyrosine,
            model="matts",
            bank_path="non-existent bank file",
        )

@pytest.mark.skipif(not is_MATTS_installed(), reason="Must have MATTS installed")
def test_switching_banks(tyrosine):
    banks = get_TAAM_databanks()
    assert len(banks) > 1
    for bank in banks:
        w = DiscambWrapper(
            tyrosine,
            model="matts",
            bank_path=bank,
        )
