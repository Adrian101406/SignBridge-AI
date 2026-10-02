import pytest

from signbridge_app.medical_guard import compare_critical_terms


@pytest.mark.parametrize(
    ("source", "proposal", "expected"),
    [
        ("Do not take 2 tablets", "Take two tablets", "negation"),
        ("Take 5 mg", "Take 10 mg", "number"),
        ("Take aspirin after meals", "Take ibuprofen after meals", "medication"),
        ("Use 2 mL", "Use 2 mg", "unit"),
    ],
)
def test_critical_change_is_flagged(source, proposal, expected):
    assert expected in compare_critical_terms(source, proposal)


def test_punctuation_and_capitalization_are_not_critical():
    assert compare_critical_terms("Take 2 mg Aspirin.", "take 2 mg aspirin") == ()
