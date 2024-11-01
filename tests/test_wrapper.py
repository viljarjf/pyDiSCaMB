import pytest


def test_import():
    import pydiscamb

    assert isinstance(pydiscamb.get_discamb_version(), str)


def test_init(random_structure):
    from pydiscamb import DiscambWrapper

    w = DiscambWrapper(random_structure)


def test_fcalc(random_structure):
    from pydiscamb import DiscambWrapper

    w = DiscambWrapper(random_structure)
    sf = w.f_calc(4.0)
    assert isinstance(sf[0], complex)


def test_update_structure(random_structure):
    from pydiscamb import DiscambWrapper

    wrapper = DiscambWrapper(random_structure)
    site = random_structure.scatterers()[0].site
    random_structure.scatterers()[0].site = (site[2], site[1], site[0])


def test_update_structure_recalculate_fcalc(random_structure):
    from pydiscamb import DiscambWrapper

    wrapper = DiscambWrapper(random_structure)
    sf_before = wrapper.f_calc(5)
    site = random_structure.scatterers()[0].site
    random_structure.scatterers()[0].site = (site[2], site[1], site[0])
    sf_after = wrapper.f_calc(5)
    assert not pytest.approx(sf_before) == sf_after
