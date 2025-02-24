import pytest

from cctbx.xray import structure


@pytest.fixture
def random_structure() -> structure:
    from cctbx.development import random_structure as cctbx_random_structure
    from cctbx.sgtbx import space_group_info

    group = space_group_info(19)
    xrs = cctbx_random_structure.xray_structure(
        space_group_info=group,
        elements=["Au", "C"] * 3,
        general_positions_only=False,
        use_u_iso=True,
        random_u_iso=False,
        random_occupancy=False,
    )
    xrs.scattering_type_registry(table="electron")
    return xrs


@pytest.fixture
def random_structure_u_iso() -> structure:
    from cctbx.development import random_structure as cctbx_random_structure
    from cctbx.sgtbx import space_group_info

    group = space_group_info(19)
    xrs = cctbx_random_structure.xray_structure(
        space_group_info=group,
        elements=["Au", "C"] * 3,
        general_positions_only=False,
        use_u_iso=True,
        random_u_iso=True,
        random_occupancy=False,
    )
    xrs.scattering_type_registry(table="electron")
    return xrs


@pytest.fixture
def random_structure_u_aniso() -> structure:
    from cctbx.development import random_structure as cctbx_random_structure
    from cctbx.sgtbx import space_group_info

    group = space_group_info(19)
    xrs = cctbx_random_structure.xray_structure(
        space_group_info=group,
        elements=["Au", "C"] * 3,
        general_positions_only=False,
        use_u_aniso=True,
        random_u_iso=True,
        random_occupancy=False,
    )
    xrs.scattering_type_registry(table="electron")
    return xrs


@pytest.fixture
def large_random_structure() -> structure:
    from cctbx.development import random_structure as cctbx_random_structure
    from cctbx.sgtbx import space_group_info

    group = space_group_info(19)
    xrs = cctbx_random_structure.xray_structure(
        space_group_info=group,
        elements=["Au", "C", "H", "N", "O", "Ni", "Fe", "C", "O"] * 100,
        general_positions_only=False,
        use_u_iso=True,
        random_u_iso=False,
        random_occupancy=False,
    )
    xrs.scattering_type_registry(table="electron")
    return xrs


@pytest.fixture
def tyrosine() -> structure:
    import mmtbx.model
    import iotbx.pdb
    from libtbx.utils import null_out

    pdb_str = """
CRYST1   16.170   14.591   15.187  90.00  90.00  90.00 P 1
SCALE1      0.061843  0.000000  0.000000        0.00000
SCALE2      0.000000  0.068535  0.000000        0.00000
SCALE3      0.000000  0.000000  0.065846        0.00000
ATOM      1  N   TYR A   4       8.357   9.217   8.801  1.00 10.55           N
ATOM      2  CA  TYR A   4       9.150   8.055   9.050  1.00 10.24           C
ATOM      3  C   TYR A   4      10.419   8.399   9.804  1.00  9.86           C
ATOM      4  O   TYR A   4      10.726   9.591  10.050  1.00 11.39           O
ATOM      5  CB  TYR A   4       9.496   7.352   7.737  1.00 30.00           C
ATOM      6  CG  TYR A   4       8.296   6.791   7.006  1.00 30.00           C
ATOM      7  CD1 TYR A   4       7.820   5.517   7.293  1.00 30.00           C
ATOM      8  CD2 TYR A   4       7.642   7.534   6.034  1.00 30.00           C
ATOM      9  CE1 TYR A   4       6.724   5.000   6.628  1.00 30.00           C
ATOM     10  CE2 TYR A   4       6.545   7.024   5.364  1.00 30.00           C
ATOM     11  CZ  TYR A   4       6.091   5.758   5.665  1.00 30.00           C
ATOM     12  OH  TYR A   4       5.000   5.244   5.000  1.00 30.00           O
ATOM     13  OXT TYR A   4      11.170   7.502  10.187  1.00  9.86           O
ATOM     14  H1  TYR A   4       8.254   9.323   7.923  1.00 10.55           H
ATOM     15  H2  TYR A   4       7.560   9.120   9.184  1.00 10.55           H
ATOM     16  H3  TYR A   4       8.764   9.932   9.140  1.00 10.55           H
ATOM     17  HA  TYR A   4       8.637   7.441   9.599  1.00 10.24           H
ATOM     18  HB2 TYR A   4       9.929   7.989   7.148  1.00 30.00           H
ATOM     19  HB3 TYR A   4      10.097   6.615   7.928  1.00 30.00           H
ATOM     20  HD1 TYR A   4       8.245   5.005   7.942  1.00 30.00           H
ATOM     21  HD2 TYR A   4       7.946   8.389   5.830  1.00 30.00           H
ATOM     22  HE1 TYR A   4       6.415   4.146   6.829  1.00 30.00           H
ATOM     23  HE2 TYR A   4       6.116   7.532   4.714  1.00 30.00           H
ATOM     24  HH  TYR A   4       4.712   5.804   4.444  1.00 30.00           H
"""
    pdb_inp = iotbx.pdb.input(lines=pdb_str.split("\n"), source_info=None)
    model = mmtbx.model.manager(model_input=pdb_inp, log=null_out())
    xrs = model.get_xray_structure()
    xrs.scattering_type_registry(table="electron")
    return xrs


@pytest.fixture
def lysozyme() -> structure:
    import mmtbx.model
    import iotbx.pdb
    from libtbx.utils import null_out
    import requests

    data = requests.get("https://files.rcsb.org/view/7ULY.pdb")
    pdb_str = data.content.decode("utf-8")

    pdb_inp = iotbx.pdb.input(lines=pdb_str.split("\n"), source_info=None)
    model = mmtbx.model.manager(model_input=pdb_inp, log=null_out())
    xrs = model.get_xray_structure()
    xrs.scattering_type_registry(table="electron")
    return xrs
