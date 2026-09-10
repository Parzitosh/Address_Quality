"""
QC Remarks Engine
-----------------
Human-readable interpretation layer for address-quality results.

Design principles:
- quality_class remains score-driven; this module does not change it.
- Geographic review is separated from general address completeness.
- Remarks describe the most material issue instead of dumping failed checks.
- Missing building/road/house details are treated as enrichment opportunities
  unless they materially affect usability.
- correction_suggestion is populated only when a concrete action is useful.
"""

from __future__ import annotations


# ---------------------------------------------------------------------
# Controlled QC reason codes
# ---------------------------------------------------------------------

def _reason_codes(review_reason: str) -> list[str]:
    text = (review_reason or "").upper()

    codes = []

    if "PIN-STATE" in text:
        if "AMBIGUOUS" in text:
            codes.append("PIN_STATE_AMBIGUOUS")
        else:
            codes.append("PIN_STATE_MISMATCH")
    if "VILLAGE RESOLUTION IS AMBIGUOUS" in text:
        codes.append("VILLAGE_AMBIGUOUS")
    elif "VILLAGE COULD NOT BE VALIDATED" in text:
        codes.append("VILLAGE_UNVALIDATED")
    if "SUBDISTRICT COULD NOT BE VALIDATED" in text:
        codes.append("SUBDISTRICT_UNVALIDATED")

    return codes


def _geo_reason_text(codes: list[str]) -> str:
    labels = {
        "PIN_STATE_MISMATCH": "PIN-State mismatch",
        "PIN_STATE_AMBIGUOUS": "PIN-State mapping is ambiguous",
        "VILLAGE_AMBIGUOUS": "village could not be uniquely resolved",
        "VILLAGE_UNVALIDATED": "village could not be validated",
        "SUBDISTRICT_UNVALIDATED": "subdistrict could not be validated",
    }
    return "; ".join(labels[c] for c in codes if c in labels)


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
    """
    Return:
        primary_qc_remark,
        detailed_qc_remark,
        qc_action

    correction_suggestion is deliberately NOT repeated inside the detailed
    remark. It remains its own output field in address_quality.py.
    """

    # ================================================================
    # 1. Normalize statuses
    # ================================================================

    pin_bad = pin_exists == "FAIL"
    pin_state_bad = pin_state_match == "FAIL"

    # garbage_detected is passed by address_quality.py as a boolean.
    garbage = bool(garbage_detected)

    geo_codes = _reason_codes(review_reason)

    # Add direct status evidence if the review reason was not populated.
    if pin_state_bad and "PIN_STATE_MISMATCH" not in geo_codes:
        geo_codes.append("PIN_STATE_MISMATCH")

    # ================================================================
    # 2. Critical / unusable addresses
    # ================================================================

    if critical_issue:
        if garbage:
            primary = (
                "Address is not usable because the supplied text does not "
                "contain sufficient address information."
            )
            detail = (
                "The address appears to contain insufficient or non-address "
                "text. A complete customer address should be captured before "
                "field use."
            )
            action = "ADDRESS RECAPTURE REQUIRED"

        elif pin_bad:
            primary = (
                "Address requires correction because the supplied PIN "
                "could not be validated."
            )
            detail = (
                "The PIN is invalid or was not found in the India Post "
                "reference master, so the postal location cannot be reliably "
                "validated."
            )
            action = "ADDRESS CORRECTION REQUIRED"

        else:
            primary = (
                "Address requires correction because a critical address "
                "input is unusable."
            )
            detail = (
                "One or more critical inputs prevent the address from being "
                "reliably used for downstream validation or field operations."
            )
            action = "ADDRESS CORRECTION REQUIRED"

        return primary, detail, action

    # ================================================================
    # 3. Geographic consistency issues take priority
    # ================================================================

    if review_flag and geo_codes:
        geo_text = _geo_reason_text(geo_codes)

        if any(
            c in geo_codes
            for c in (
                "PIN_STATE_MISMATCH",
                "PIN_STATE_AMBIGUOUS",
                "DISTRICT_STATE_MISMATCH",
            )
        ):
            primary = (
                "Address requires geographic verification because the "
                "administrative details are inconsistent with the supplied PIN."
            )
            detail = (
                f"Geographic validation identified: {geo_text}. "
                "The address may still contain usable premise/locality "
                "information, but the conflicting geographic field should be "
                "verified before relying on the location."
            )
            action = "VERIFY GEOGRAPHY"

        else:
            primary = (
                "Address requires locality verification because part of the "
                "administrative hierarchy could not be resolved confidently."
            )
            detail = (
                f"Geographic validation identified: {geo_text}. "
                "The supplied address is not necessarily invalid, but the "
                "local administrative entity needs verification."
            )
            action = "VERIFY LOCALITY"

        return primary, detail, action

    # ================================================================
    # 4. Good / excellent addresses
    # ================================================================

    if quality_class == "EXCELLENT":
        if premise_present and street_present and locality_present:
            primary = (
                "Address is complete, specific and suitable for field use."
            )
            detail = (
                "The address contains usable premise information together "
                "with road/street and locality context, with no material "
                "quality issue requiring correction."
            )
        elif premise_present and locality_present:
            primary = (
                "Address is sufficiently specific and suitable for field use."
            )
            detail = (
                "The address has a clear premise and locality context. "
                "Additional building or street detail may improve precision "
                "but is not a material quality issue."
            )
        else:
            primary = (
                "Address is high quality and suitable for field use."
            )
            detail = (
                "No material address-quality issue was identified. "
                "Any missing optional detail is treated as enrichment rather "
                "than a correction requirement."
            )

        return primary, detail, "ACCEPT"

    if quality_class == "GOOD":
        if premise_present and locality_present:
            primary = (
                "Address is usable for field operations with only minor "
                "enrichment possible."
            )
            detail = (
                "The address contains meaningful location and premise "
                "information. Any missing building or road detail is an "
                "optional precision improvement rather than a blocking issue."
            )
            return primary, detail, "ACCEPT – OPTIONAL ENRICHMENT"

        if locality_present or anchor_present:
            primary = (
                "Address is usable, but additional location detail would "
                "improve field precision."
            )
            detail = (
                "The address has a usable geographic anchor, but some "
                "premise/street/locality detail is limited."
            )
            return primary, detail, "ACCEPT – OPTIONAL ENRICHMENT"

    # ================================================================
    # 5. USABLE addresses
    # ================================================================

    if quality_class == "USABLE":
        if not premise_present and not street_present and not locality_present:
            primary = (
                "Address has basic geographic context but is not sufficiently "
                "specific for reliable field navigation."
            )
            detail = (
                "The address lacks clear premise, street and locality detail. "
                "More specific location information is required to improve "
                "field usability."
            )
            return primary, detail, "ADDRESS ENRICHMENT"

        if not premise_present:
            primary = (
                "Address is usable at a geographic level but lacks a clear "
                "premise identifier."
            )
            detail = (
                "The location context is present, but a house, flat, plot, "
                "shop or other premise identifier was not detected."
            )
            return primary, detail, "ADDRESS ENRICHMENT"

        if not locality_present:
            primary = (
                "Address has premise information but limited locality detail."
            )
            detail = (
                "The premise is identifiable, but locality/area context is "
                "limited and may reduce field navigation precision."
            )
            return primary, detail, "ADDRESS ENRICHMENT"

        if not street_present:
            primary = (
                "Address is usable but would benefit from more precise "
                "road/street context."
            )
            detail = (
                "Premise and locality information are present. A road, street, "
                "sector or equivalent route reference would improve navigation."
            )
            return primary, detail, "ACCEPT – OPTIONAL ENRICHMENT"

        primary = "Address is usable with minor detail limitations."
        detail = (
            "The address contains meaningful location information, but "
            "additional detail may improve field-level precision."
        )
        return primary, detail, "ACCEPT – OPTIONAL ENRICHMENT"

    # ================================================================
    # 6. REQUIRES_CALL
    # ================================================================

    if quality_class == "REQUIRES_CALL":
        if not premise_present and not locality_present:
            primary = (
                "Address needs customer clarification because it lacks "
                "sufficient specific location detail."
            )
            detail = (
                "The available address does not provide a reliable premise "
                "and locality combination for field navigation."
            )
        elif not premise_present:
            primary = (
                "Address needs customer clarification because the premise "
                "cannot be identified confidently."
            )
            detail = (
                "Geographic context is available, but the specific property "
                "or premise cannot be identified from the supplied address."
            )
        else:
            primary = (
                "Address needs customer clarification before reliable field use."
            )
            detail = (
                "The address contains some usable information, but the "
                "remaining gaps are significant enough to warrant clarification."
            )

        return primary, detail, "CUSTOMER CONTACT REQUIRED"

    # ================================================================
    # 7. INVALID / fallback
    # ================================================================

    if quality_class == "INVALID":
        primary = "Address requires correction before it can be used."
        detail = (
            "The address quality is below the usable threshold and should "
            "be corrected or recaptured."
        )
        return primary, detail, "ADDRESS CORRECTION REQUIRED"

    # Defensive fallback.
    primary = "Address requires review before field use."
    detail = (
        "The address could not be assigned a standard QC interpretation "
        "from the available validation results."
    )
    return primary, detail, "VERIFY GEOGRAPHY"
