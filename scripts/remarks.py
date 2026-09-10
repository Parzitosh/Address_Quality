"""P1 QC remarks engine.

Remarks are evidence-based and separate from score classification.
Optional completeness gaps are not presented as defects for ACCEPT outcomes.
"""


def _reason_codes(review_reason: str) -> list[str]:
    return [x.strip() for x in (review_reason or "").split(";") if x.strip()]


def build_remarks(
    quality_class,
    *,
    pin_exists,
    pin_state_match,
    premise_present,
    building_present,
    street_present,
    locality_present,
    anchor_present,
    garbage_detected,
    critical_issue=False,
    review_flag=False,
    review_reason="",
    correction_suggestion="",
):
    codes = _reason_codes(review_reason)
    garbage = bool(garbage_detected)

    if critical_issue:
        if garbage:
            return (
                "Address is not usable because the free-text address lacks sufficient address information.",
                "The address appears to contain placeholder, metadata-only or otherwise insufficient text. A complete premise and locality/road address should be captured.",
                "ADDRESS RECAPTURE REQUIRED",
            )
        if pin_exists == "FAIL":
            return (
                "Address requires PIN correction before field use.",
                "The supplied PIN is invalid or is not present in the India Post reference master, so postal geography cannot be validated reliably.",
                "ADDRESS CORRECTION REQUIRED",
            )
        return (
            "Address requires correction before field use.",
            "A critical input prevents reliable downstream validation or field use.",
            "ADDRESS CORRECTION REQUIRED",
        )

    # Geographic review has priority because it is a verification issue,
    # not merely a completeness issue.
    if review_flag and codes:
        labels = {
            "PIN_STATE_MISMATCH": "PIN-State mismatch",
            "PIN_STATE_AMBIGUOUS": "PIN-State mapping is ambiguous",
            "VILLAGE_UNVALIDATED": "Village could not be validated in LGD",
            "VILLAGE_AMBIGUOUS": "Village resolution is ambiguous",
            "SUBDISTRICT_UNVALIDATED": "Subdistrict could not be validated in LGD",
            "SUBDISTRICT_AMBIGUOUS": "Subdistrict resolution is ambiguous",
            "POST_OFFICE_UNVALIDATED": "Post Office could not be validated against the PIN",
            "POST_OFFICE_AMBIGUOUS": "Post Office resolution is ambiguous",
            "LOCALITY_UNVALIDATED": "Explicit locality could not be validated in LGD",
            "LOCALITY_AMBIGUOUS": "Explicit locality resolution is ambiguous",
        }
        text = "; ".join(labels.get(c, c) for c in codes)
        geo_codes = {
            "PIN_STATE_MISMATCH", "PIN_STATE_AMBIGUOUS",
            "VILLAGE_UNVALIDATED", "VILLAGE_AMBIGUOUS",
            "SUBDISTRICT_UNVALIDATED", "SUBDISTRICT_AMBIGUOUS",
            "POST_OFFICE_UNVALIDATED", "POST_OFFICE_AMBIGUOUS",
            "LOCALITY_UNVALIDATED", "LOCALITY_AMBIGUOUS",
        }
        if any(c in geo_codes for c in codes):
            action = "VERIFY GEOGRAPHY" if any(c.startswith("PIN_STATE") for c in codes) else "VERIFY LOCALITY"
            return (
                "Address requires geographic verification before field use.",
                f"Geographic validation identified: {text}. The address may contain usable premise/locality information, but the flagged administrative entity should be verified.",
                action,
            )

    if quality_class == "EXCELLENT":
        if premise_present and street_present and locality_present:
            detail = "The address contains premise, route and locality detail with no material QC issue identified."
        elif premise_present and locality_present:
            detail = "The address contains a clear premise and locality. Additional route/building detail may improve precision but is not a material defect."
        else:
            detail = "No material address-quality issue was identified from the available checks."
        return "Address is high quality and suitable for field use.", detail, "ACCEPT"

    if quality_class == "GOOD":
        if premise_present and locality_present:
            return (
                "Address is usable for field operations with minor optional enrichment.",
                "The address contains meaningful premise and locality information. Missing optional detail is not treated as a correction requirement.",
                "ACCEPT – OPTIONAL ENRICHMENT",
            )
        return (
            "Address is usable, with some optional detail that could improve field precision.",
            "The address has sufficient geographic information for use; additional premise, route or locality detail may improve navigation.",
            "ACCEPT – OPTIONAL ENRICHMENT",
        )

    if quality_class == "USABLE":
        if not premise_present and not locality_present:
            return (
                "Address has basic geographic context but is not sufficiently specific for reliable field navigation.",
                "A more specific premise and locality/area reference is needed for dependable field navigation.",
                "ADDRESS ENRICHMENT",
            )
        if not premise_present:
            return (
                "Address is geographically usable but lacks a clear premise identifier.",
                "The location context is present, but a house, flat, plot, shop or other premise identifier was not detected.",
                "ADDRESS ENRICHMENT",
            )
        if not locality_present:
            return (
                "Address has premise information but limited locality detail.",
                "The premise is identifiable, but locality/area context is limited and may reduce field-navigation precision.",
                "ADDRESS ENRICHMENT",
            )
        if not street_present:
            return (
                "Address is usable and would benefit from more precise route detail.",
                "Premise and locality information are present. A road, street, sector, lane or equivalent route reference would improve navigation.",
                "ACCEPT – OPTIONAL ENRICHMENT",
            )
        return "Address is usable with minor detail limitations.", "The available address information is sufficient for use; additional detail may improve precision.", "ACCEPT – OPTIONAL ENRICHMENT"

    if quality_class == "REQUIRES_CALL":
        if not premise_present and not locality_present:
            detail = "The available address does not provide a reliable premise and locality combination for field navigation."
        elif not premise_present:
            detail = "Geographic context is available, but the specific property or premise cannot be identified confidently."
        else:
            detail = "The address contains some usable information, but the remaining gaps are significant enough to warrant clarification."
        return "Address needs customer clarification before reliable field use.", detail, "CUSTOMER CONTACT REQUIRED"

    return "Address requires correction before field use.", "The address quality is below the usable threshold and should be corrected or recaptured.", "ADDRESS CORRECTION REQUIRED"
