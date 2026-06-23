def calculate_evidence_strength(
    p_positive: float,
    match_score: float = None,
    relative_hdi_width: float = None
) -> str:
    """
    Combines posterior probability, HDI width, and Matcher Distance Score 
    into a unified Evidence Strength Score.
    
    Levels: Weak, Directional, Moderate, Strong, Very Strong
    """
    if p_positive < 0.7:
        return "Weak"
        
    strength = "Directional"
    
    if p_positive >= 0.95:
        strength = "Very Strong"
    elif p_positive >= 0.90:
        strength = "Strong"
    elif p_positive >= 0.80:
        strength = "Moderate"

    # Downgrade if the match score is bad (assuming lower is better)
    if match_score is not None and match_score > 1.5:
        if strength in ["Very Strong", "Strong"]:
            return "Moderate"
        elif strength == "Moderate":
            return "Directional"

    # Downgrade if the HDI is extremely wide (noisy)
    # relative_hdi_width = (hdi_upper - hdi_lower) / mean_lift
    if relative_hdi_width is not None and relative_hdi_width > 2.0:
        if strength in ["Very Strong", "Strong"]:
            return "Moderate"
            
    return strength
