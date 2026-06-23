from openlift.core.evidence import calculate_evidence_strength


def test_weak_below_70():
    assert calculate_evidence_strength(0.65) == "Weak"


def test_directional_70_to_80():
    assert calculate_evidence_strength(0.75) == "Directional"


def test_moderate_80_to_90():
    assert calculate_evidence_strength(0.85) == "Moderate"


def test_strong_90_to_95():
    assert calculate_evidence_strength(0.92) == "Strong"


def test_very_strong_above_95():
    assert calculate_evidence_strength(0.97) == "Very Strong"


def test_bad_match_score_downgrades_strong():
    assert calculate_evidence_strength(0.92, match_score=2.0) == "Moderate"


def test_bad_match_score_downgrades_very_strong():
    assert calculate_evidence_strength(0.97, match_score=2.0) == "Moderate"


def test_bad_match_score_downgrades_moderate():
    assert calculate_evidence_strength(0.85, match_score=2.0) == "Directional"


def test_wide_hdi_downgrades_strong():
    assert calculate_evidence_strength(0.92, relative_hdi_width=3.0) == "Moderate"


def test_good_match_does_not_downgrade():
    assert calculate_evidence_strength(0.97, match_score=0.5) == "Very Strong"
