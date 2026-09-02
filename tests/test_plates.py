from app.services.plates import layout_hint, normalize, syntax_ok, vote


def test_normalize_strips_space():
    assert normalize("gj 01 ab 1234") == "GJ01AB1234"


def test_standard_and_bh_syntax():
    assert syntax_ok("GJ01AB1234")
    assert syntax_ok("GJ01A1234")
    assert syntax_ok("26BH4567AB")
    assert syntax_ok("GJ18G1234")
    assert not syntax_ok("HELLO")
    assert not syntax_ok("GJ01AB12")


def test_independent_vote():
    assert vote(["GJ01AB1234", "GJ01AB1234", "GJ01AB12B4"]) == "GJ01AB1234"


def test_layout_hint_zero_as_g():
    assert layout_hint("GJG1AB1234") == "GJ01AB1234"
