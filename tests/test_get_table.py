import pytest
from pydiscamb import get_table


@pytest.mark.parametrize(
    ["table_str", "expected_entries", "expected_number_of_gaussian_parameters"],
    [
        (
            "Waasmeier-Kirfel",
            211,
            5,
        ),
        (
            "electron-cctbx",
            98,
            5,
        ),
        (
            "electron-IT",
            98,
            5,
        ),
        (
            "IT92",
            213,
            4,
        ),
        (
            "empty, non-existing table",
            0,
            0,
        ),
    ],
)
def test_get_table(
    table_str: str, expected_entries: int, expected_number_of_gaussian_parameters: int
):
    table = get_table(table_str)
    assert len(table) == expected_entries
    for key, val in table.items():
        assert len(val.a) == expected_number_of_gaussian_parameters
        assert len(val.b) == expected_number_of_gaussian_parameters


@pytest.mark.parametrize(
    "table_names",
    [
        ("Waasmeier-Kirfel", "wk", "WK-95"),
        ("IT92", "it1992", "IT-1992"),
        ("electron-cctbx", "electron"),
    ],
)
def test_get_table_alias(table_names):
    tables = [get_table(name) for name in table_names]
    for table in tables:
        for key, val in table.items():
            for other in tables:
                assert other.get(key).a == val.a
                assert other.get(key).b == val.b
                assert other.get(key).c == val.c


@pytest.mark.parametrize(
    "table",
    [
        None,
        "electron",
        "wk1995",
    ],
)
def test_read_table_from_structure(table):
    from cctbx.development import random_structure as cctbx_random_structure
    from cctbx.sgtbx import space_group_info

    group = space_group_info(19)
    xrs = cctbx_random_structure.xray_structure(
        space_group_info=group,
        elements=["Au", "C"],
        general_positions_only=False,
        use_u_iso=True,
        random_u_iso=False,
        random_occupancy=False,
    )
    if table is not None:
        xrs.scattering_type_registry(table=table)

    import pydiscamb

    w = pydiscamb.DiscambWrapper(xrs)
    fc = w.f_calc(2)
