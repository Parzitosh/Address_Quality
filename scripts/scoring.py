WEIGHTS = {'pin': 15, 'geo': 25, 'premise': 20, 'street_locality': 20, 'standardization': 10, 'geo_confidence': 10}


def score_from_fraction(fraction, weight):
    return round(max(0, min(1, fraction)) * weight, 2)


def calculate_score(checks):
    scores = {f'{k}_score': score_from_fraction(checks[k], w) for k, w in WEIGHTS.items()}
    scores['quality_score'] = round(sum(scores.values()), 2)
    return scores



def classify_score(score):
    """
    Quality class is determined ONLY by the continuous score.

    Geographic/manual-review concerns are carried separately in
    qc_review_flag / qc_review_reason. This prevents a categorical
    override from collapsing materially different scores.
    """
    if score >= 90:
        return "EXCELLENT"
    if score >= 75:
        return "GOOD"
    if score >= 60:
        return "USABLE"
    if score >= 40:
        return "REQUIRES_CALL"
    return "INVALID"

