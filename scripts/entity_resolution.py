from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

try:
    from rapidfuzz import fuzz, process

    RAPIDFUZZ_AVAILABLE = True

except ImportError:

    RAPIDFUZZ_AVAILABLE = False


# ============================================================
# ALIASES
# ============================================================

STATE_ALIASES = {
    "HR": "HARYANA",
    "UP": "UTTAR PRADESH",
    "MP": "MADHYA PRADESH",
    "HP": "HIMACHAL PRADESH",
    "RJ": "RAJASTHAN",
    "PB": "PUNJAB",
    "WB": "WEST BENGAL",
    "UK": "UTTARAKHAND",
    "CG": "CHHATTISGARH",
    "CHHATISGARH": "CHHATTISGARH",
    "J AND K": "JAMMU AND KASHMIR",
    "J K": "JAMMU AND KASHMIR",
    "J&K": "JAMMU AND KASHMIR",
    "AP": "ANDHRA PRADESH",
    "TS": "TELANGANA",
    "ORISSA": "ODISHA",
    "PONDICHERRY": "PUDUCHERRY",
}


COMMON_ALIASES = {
    "GURGAON": "GURUGRAM",
    "BANGALORE": "BENGALURU",
    "BOMBAY": "MUMBAI",
    "CALCUTTA": "KOLKATA",
    "MADRAS": "CHENNAI",
    "POONA": "PUNE",
    "TRIVANDRUM": "THIRUVANANTHAPURAM",
    "COCHIN": "KOCHI",
}


LABELS = {
    "VILL",
    "VILLAGE",
    "GRAM",
    "GAON",
    "PO",
    "P O",
    "POST",
    "POST OFFICE",
    "DIST",
    "DISTRICT",
    "DT",
    "TEHSIL",
    "TAHSIL",
    "TALUK",
    "TALUKA",
    "SUBDISTRICT",
    "SUB DISTRICT",
}


# ============================================================
# THRESHOLDS
# ============================================================

FUZZY_REVIEW = 0.85
FUZZY_PASS = 0.90
FUZZY_VERY_STRONG = 0.95


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(value: object) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).upper().strip()

    text = text.replace("&", " AND ")

    text = re.sub(
        r"[^A-Z0-9\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_entity(value: object) -> str:

    text = normalize_text(value)

    if not text:
        return ""

    parts = text.split()

    # Remove administrative labels from beginning.
    while parts and parts[0] in LABELS:
        parts.pop(0)

    text = " ".join(parts)

    # Common geographic aliases.
    text = COMMON_ALIASES.get(
        text,
        text,
    )

    return text


def canonical_state(value: object) -> str:

    text = normalize_text(value)

    if not text:
        return ""

    return STATE_ALIASES.get(
        text,
        COMMON_ALIASES.get(
            text,
            text,
        ),
    )


def normalize_pin(value: object) -> str:

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    text = str(value).strip()

    # Excel numeric representation:
    # 122001.0 -> 122001
    if re.fullmatch(
        r"\d+\.0+",
        text,
    ):
        text = text.split(".")[0]

    digits = re.sub(
        r"\D",
        "",
        text,
    )

    if len(digits) == 6:
        return digits

    return ""


def tokenize(value: object) -> set:

    return {
        x
        for x in normalize_entity(value).split()
        if len(x) >= 2
    }


def similarity(
    a: str,
    b: str,
) -> float:

    if RAPIDFUZZ_AVAILABLE:

        return (
            fuzz.WRatio(a, b)
            / 100.0
        )

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


# ============================================================
# RESOLUTION OBJECT
# ============================================================

@dataclass
class Resolution:

    entity_type: str

    input_value: str

    normalized_value: str

    status: str

    match_value: str = ""

    match_score: float = 0.0

    state: str = ""

    district: str = ""

    subdistrict: str = ""

    village_code: str = ""

    district_code: str = ""

    subdistrict_code: str = ""

    source: str = ""

    reason: str = ""


# ============================================================
# GEOGRAPHIC RESOLVER
# ============================================================

class GeographicResolver:

    """
    Context-first geographic entity resolution.

    Resolution order:

    1. Exact normalized match
    2. Geographic context
    3. PIN context
    4. Bounded fuzzy matching
    5. Ambiguity protection

    Important:
        Duplicate village/district/ULB names are NOT removed.

        They are legitimate geographic duplicates and are
        disambiguated using hierarchy/context.
    """

    def __init__(
        self,
        masters: Dict[str, pd.DataFrame],
        indexes: Dict[str, object],
    ):

        self.masters = masters

        self.indexes = indexes

        # ----------------------------------------------------
        # Main indexes
        # ----------------------------------------------------

        self.district_by_name = indexes.get(
            "district_by_name",
            {},
        )

        self.subdistrict_by_name = indexes.get(
            "subdistrict_by_name",
            {},
        )

        self.village_by_name = indexes.get(
            "village_by_name",
            {},
        )

        self.ulb_by_name = indexes.get(
            "ulb_by_name",
            {},
        )

        # ----------------------------------------------------
        # Hierarchical indexes
        # ----------------------------------------------------

        self.village_by_district = indexes.get(
            "village_by_district",
            {},
        )

        self.village_by_subdistrict = indexes.get(
            "village_by_subdistrict",
            {},
        )

        self.village_by_district_subdistrict = indexes.get(
            "village_by_district_subdistrict",
            {},
        )

        self.subdistrict_by_district = indexes.get(
            "subdistrict_by_district",
            {},
        )

        # ----------------------------------------------------
        # India Post
        # ----------------------------------------------------

        self.pin_states = indexes.get(
            "pin_states",
            {},
        )

        self.pin_districts = indexes.get(
            "pin_districts",
            {},
        )

        self.pin_post_offices = indexes.get(
            "pin_post_offices",
            {},
        )

        # ----------------------------------------------------
        # Token indexes
        #
        # Support BOTH:
        #
        # village_token_index
        # village_tokens
        #
        # This keeps the resolver compatible with both
        # versions of master_loader.py.
        # ----------------------------------------------------

        self.village_token_index = indexes.get(
            "village_token_index",
            indexes.get(
                "village_tokens",
                {},
            ),
        )

        self.district_token_index = indexes.get(
            "district_token_index",
            indexes.get(
                "district_tokens",
                {},
            ),
        )

        self.subdistrict_token_index = indexes.get(
            "subdistrict_token_index",
            indexes.get(
                "subdistrict_tokens",
                {},
            ),
        )

        self.ulb_token_index = indexes.get(
            "ulb_token_index",
            indexes.get(
                "ulb_tokens",
                {},
            ),
        )

    # ========================================================
    # DATAFRAME ROW HELPERS
    # ========================================================

    def _rows(
        self,
        master_key: str,
        indices: Iterable[int],
    ) -> List[pd.Series]:

        df = self.masters[master_key]

        rows = []

        for i in indices:

            try:

                rows.append(
                    df.loc[i]
                )

            except (KeyError, TypeError):

                continue

        return rows

    # ========================================================
    # ROW ENTITY HELPERS
    # ========================================================

    @staticmethod
    def _row_district(
        row: pd.Series,
    ) -> str:

        value = row.get(
            "district_name",
            row.get(
                "district_name_clean",
                row.get(
                    "district",
                    "",
                ),
            ),
        )

        return normalize_entity(value)

    @staticmethod
    def _row_subdistrict(
        row: pd.Series,
    ) -> str:

        value = row.get(
            "subdistrict_name",
            row.get(
                "sub_district_name",
                row.get(
                    "subdistrict_name_clean",
                    "",
                ),
            ),
        )

        return normalize_entity(value)

    @staticmethod
    def _row_state(
        row: pd.Series,
    ) -> str:

        value = row.get(
            "state_name",
            row.get(
                "statename",
                "",
            ),
        )

        return canonical_state(value)

    # ========================================================
    # PIN CONTEXT
    # ========================================================

    def _pin_context(
        self,
        pin: str,
    ) -> Tuple[set, set]:

        pin = normalize_pin(pin)

        districts = {
            normalize_entity(x)
            for x in self.pin_districts.get(
                pin,
                set(),
            )
            if x
        }

        states = {
            canonical_state(x)
            for x in self.pin_states.get(
                pin,
                set(),
            )
            if x
        }

        return districts, states

    # ========================================================
    # CONTEXT FILTER
    # ========================================================

    def _context_filter(
        self,
        rows: Sequence[pd.Series],
        state="",
        district="",
        pin="",
        subdistrict="",
    ) -> List[pd.Series]:

        if not rows:
            return []

        state = canonical_state(state)

        district = normalize_entity(
            district
        )

        subdistrict = normalize_entity(
            subdistrict
        )

        pin_districts, pin_states = (
            self._pin_context(pin)
        )

        scored = []

        for row in rows:

            row_district = self._row_district(
                row
            )

            row_state = self._row_state(
                row
            )

            row_subdistrict = (
                self._row_subdistrict(row)
            )

            score = 0

            # Supplied district.
            if (
                district
                and row_district == district
            ):
                score += 10

            # Supplied subdistrict.
            if (
                subdistrict
                and row_subdistrict == subdistrict
            ):
                score += 8

            # PIN district.
            if (
                pin_districts
                and row_district in pin_districts
            ):
                score += 6

            # Supplied state.
            if (
                state
                and row_state
                and row_state == state
            ):
                score += 4

            # PIN state.
            if (
                pin_states
                and row_state
                and row_state in pin_states
            ):
                score += 2

            scored.append(
                (
                    score,
                    row,
                )
            )

        best_score = max(
            score
            for score, _ in scored
        )

        if best_score > 0:

            return [
                row
                for score, row in scored
                if score == best_score
            ]

        return list(rows)

    # ========================================================
    # FUZZY CANDIDATES
    # ========================================================

    def _fuzzy_candidates(
        self,
        query: str,
        names: Iterable[str],
        limit: int = 3,
        cutoff: float = FUZZY_REVIEW,
    ):

        names = list(
            dict.fromkeys(
                n
                for n in names
                if n
            )
        )

        if not names:
            return []

        if RAPIDFUZZ_AVAILABLE:

            results = process.extract(
                query,
                names,
                scorer=fuzz.WRatio,
                limit=limit,
                score_cutoff=cutoff * 100,
            )

            return [
                (
                    str(name),
                    float(score) / 100.0,
                )
                for name, score, _ in results
            ]

        scored = sorted(
            (
                (
                    similarity(
                        query,
                        name,
                    ),
                    name,
                )
                for name in names
            ),
            reverse=True,
        )

        return [
            (
                name,
                score,
            )
            for score, name in scored[:limit]
            if score >= cutoff
        ]

    # ========================================================
    # TOKEN CANDIDATE NAMES
    # ========================================================

    def _token_candidate_names(
        self,
        query: str,
        token_index: Dict[str, set],
    ) -> List[str]:

        q_tokens = tokenize(query)

        if not q_tokens:
            return []

        names = set()

        for token in q_tokens:

            names.update(
                token_index.get(
                    token,
                    set(),
                )
            )

        return list(names)

    # ========================================================
    # RESOLUTION FROM ROW
    # ========================================================

    def _resolution_from_row(
        self,
        entity_type,
        raw,
        query,
        status,
        best,
        score,
        row,
        source,
        reason,
    ) -> Resolution:

        return Resolution(

            entity_type=entity_type,

            input_value=raw,

            normalized_value=query,

            status=status,

            match_value=best,

            match_score=score,

            district=self._row_district(
                row
            ),

            subdistrict=self._row_subdistrict(
                row
            ),

            village_code=str(
                row.get(
                    "village_code",
                    "",
                )
            ),

            district_code=str(
                row.get(
                    "district_code",
                    "",
                )
            ),

            subdistrict_code=str(
                row.get(
                    "sub_district_code",
                    row.get(
                        "subdistrict_code",
                        "",
                    ),
                )
            ),

            state=self._row_state(
                row
            ),

            source=source,

            reason=reason,
        )

    # ========================================================
    # DISTRICT
    # ========================================================

    def resolve_district(
        self,
        value,
        state="",
        pin="",
    ) -> Resolution:

        raw = (
            ""
            if value is None
            else str(value)
        )

        query = normalize_entity(
            value
        )

        if not query:

            return Resolution(
                "DISTRICT",
                raw,
                query,
                "NOT_CHECKED",
                reason="No district supplied.",
            )

        # ----------------------------------------------------
        # Exact
        # ----------------------------------------------------

        rows = self._rows(
            "district",
            self.district_by_name.get(
                query,
                [],
            ),
        )

        rows = self._context_filter(
            rows,
            state,
            query,
            pin,
        )

        if len(rows) == 1:

            return self._resolution_from_row(
                "DISTRICT",
                raw,
                query,
                "PASS",
                query,
                1.0,
                rows[0],
                "LGD_EXACT",
                "District matched LGD exactly after normalization.",
            )

        if len(rows) > 1:

            return Resolution(
                "DISTRICT",
                raw,
                query,
                "AMBIGUOUS",
                query,
                1.0,
                source="LGD_EXACT",
                reason=(
                    "Multiple LGD districts share this "
                    "name and context did not uniquely "
                    "resolve it."
                ),
            )

        # ----------------------------------------------------
        # Fuzzy
        # ----------------------------------------------------

        pin_districts, _ = (
            self._pin_context(pin)
        )

        candidate_names = list(
            self.district_by_name.keys()
        )

        if pin_districts:

            contextual = (
                pin_districts
                & set(candidate_names)
            )

            if contextual:

                candidate_names = list(
                    contextual
                )

        candidates = self._fuzzy_candidates(
            query,
            candidate_names,
        )

        if not candidates:

            return Resolution(
                "DISTRICT",
                raw,
                query,
                "FAIL",
                source="LGD",
                reason=(
                    "District could not be resolved "
                    "in LGD."
                ),
            )

        best, score = candidates[0]

        rows = self._context_filter(
            self._rows(
                "district",
                self.district_by_name.get(
                    best,
                    [],
                ),
            ),
            state,
            best,
            pin,
        )

        if (
            len(rows) == 1
            and score >= FUZZY_PASS
        ):

            return self._resolution_from_row(
                "DISTRICT",
                raw,
                query,
                "PASS",
                best,
                score,
                rows[0],
                "LGD_FUZZY",
                (
                    "District resolved with a strong "
                    "fuzzy match and geographic context."
                ),
            )

        return Resolution(
            "DISTRICT",
            raw,
            query,
            "AMBIGUOUS",
            best,
            score,
            source="LGD_FUZZY",
            reason=(
                "District has a plausible fuzzy match "
                "but needs review."
            ),
        )

    # ========================================================
    # SUBDISTRICT
    # ========================================================

    def resolve_subdistrict(
        self,
        value,
        state="",
        district="",
        pin="",
    ) -> Resolution:

        raw = (
            ""
            if value is None
            else str(value)
        )

        query = normalize_entity(
            value
        )

        district = normalize_entity(
            district
        )

        if not query:

            return Resolution(
                "SUBDISTRICT",
                raw,
                query,
                "NOT_CHECKED",
                reason="No subdistrict supplied.",
            )

        # ----------------------------------------------------
        # Exact
        # ----------------------------------------------------

        indices = self.subdistrict_by_name.get(
            query,
            [],
        )

        rows = self._rows(
            "subdistrict",
            indices,
        )

        rows = self._context_filter(
            rows,
            state,
            district,
            pin,
        )

        if len(rows) == 1:

            return self._resolution_from_row(
                "SUBDISTRICT",
                raw,
                query,
                "PASS",
                query,
                1.0,
                rows[0],
                "LGD_EXACT",
                (
                    "Subdistrict matched LGD exactly "
                    "after normalization."
                ),
            )

        if len(rows) > 1:

            return Resolution(
                "SUBDISTRICT",
                raw,
                query,
                "AMBIGUOUS",
                query,
                1.0,
                source="LGD_EXACT",
                reason=(
                    "Multiple LGD subdistrict records "
                    "share this name and context did "
                    "not uniquely resolve it."
                ),
            )

        # ----------------------------------------------------
        # Context-bounded fuzzy candidates
        # ----------------------------------------------------

        candidate_names = []

        if district:

            district_candidates = (
                self.subdistrict_by_district.get(
                    district,
                    {},
                )
            )

            if isinstance(
                district_candidates,
                dict,
            ):

                candidate_names = list(
                    district_candidates.keys()
                )

        if not candidate_names:

            candidate_names = (
                self._token_candidate_names(
                    query,
                    self.subdistrict_token_index,
                )
            )

        if not candidate_names:

            candidate_names = list(
                self.subdistrict_by_name.keys()
            )

        candidates = self._fuzzy_candidates(
            query,
            candidate_names,
        )

        if not candidates:

            return Resolution(
                "SUBDISTRICT",
                raw,
                query,
                "FAIL",
                source="LGD",
                reason=(
                    "Subdistrict could not be "
                    "resolved in LGD."
                ),
            )

        best, score = candidates[0]

        rows = self._rows(
            "subdistrict",
            self.subdistrict_by_name.get(
                best,
                [],
            ),
        )

        rows = self._context_filter(
            rows,
            state,
            district,
            pin,
        )

        if (
            len(rows) == 1
            and score >= FUZZY_PASS
        ):

            return self._resolution_from_row(
                "SUBDISTRICT",
                raw,
                query,
                "PASS",
                best,
                score,
                rows[0],
                "LGD_FUZZY",
                (
                    "Subdistrict resolved with a "
                    "strong fuzzy match and context."
                ),
            )

        return Resolution(
            "SUBDISTRICT",
            raw,
            query,
            "AMBIGUOUS",
            best,
            score,
            source="LGD_FUZZY",
            reason=(
                "Subdistrict has a plausible fuzzy "
                "match but needs review."
            ),
        )

    # ========================================================
    # VILLAGE
    # ========================================================

    def resolve_village(
        self,
        value,
        state="",
        district="",
        subdistrict="",
        pin="",
    ) -> Resolution:

        raw = (
            ""
            if value is None
            else str(value)
        )

        query = normalize_entity(
            value
        )

        district = normalize_entity(
            district
        )

        subdistrict = normalize_entity(
            subdistrict
        )

        pin = normalize_pin(
            pin
        )

        if not query:

            return Resolution(
                "VILLAGE",
                raw,
                query,
                "NOT_CHECKED",
                reason="No village supplied.",
            )

        # ----------------------------------------------------
        # PIN-derived district
        # ----------------------------------------------------

        pin_districts, _ = (
            self._pin_context(pin)
        )

        effective_district = (
            district
            or (
                next(
                    iter(pin_districts)
                )
                if len(pin_districts) == 1
                else ""
            )
        )

        # ====================================================
        # EXACT MATCH
        # ====================================================

        exact_indices = (
            self.village_by_name.get(
                query,
                [],
            )
        )

        exact_rows = self._rows(
            "village",
            exact_indices,
        )

        exact_rows = self._context_filter(
            exact_rows,
            state,
            effective_district,
            pin,
            subdistrict,
        )

        # Additional exact subdistrict check.
        if subdistrict:

            exact_sub = [
                row
                for row in exact_rows
                if self._row_subdistrict(row)
                == subdistrict
            ]

            if exact_sub:

                exact_rows = exact_sub

        if len(exact_rows) == 1:

            return self._resolution_from_row(
                "VILLAGE",
                raw,
                query,
                "PASS",
                query,
                1.0,
                exact_rows[0],
                "LGD_EXACT",
                (
                    "Village matched LGD exactly and "
                    "geographic context uniquely "
                    "resolved it."
                ),
            )

        if len(exact_rows) > 1:

            return Resolution(
                "VILLAGE",
                raw,
                query,
                "AMBIGUOUS",
                query,
                1.0,
                source="LGD_EXACT",
                reason=(
                    "Village name exists in multiple "
                    "LGD locations and context did not "
                    "uniquely resolve it."
                ),
            )

        # ====================================================
        # BOUNDED FUZZY SEARCH
        # ====================================================

        candidate_indices = None

        # ----------------------------------------------------
        # Most precise:
        #
        # District + Subdistrict
        # ----------------------------------------------------

        if (
            effective_district
            and subdistrict
        ):

            candidate_indices = (
                self.village_by_district_subdistrict.get(
                    (
                        effective_district,
                        subdistrict,
                    )
                )
            )

        # ----------------------------------------------------
        # Subdistrict
        # ----------------------------------------------------

        if (
            candidate_indices is None
            and subdistrict
        ):

            candidate_indices = (
                self.village_by_subdistrict.get(
                    subdistrict
                )
            )

        # ----------------------------------------------------
        # District
        # ----------------------------------------------------

        if (
            candidate_indices is None
            and effective_district
        ):

            candidate_indices = (
                self.village_by_district.get(
                    effective_district
                )
            )

        # ----------------------------------------------------
        # Convert row indexes to candidate names.
        #
        # IMPORTANT:
        # We only do this when the candidate structure
        # contains row indexes.
        # ----------------------------------------------------

        if candidate_indices is not None:

            candidate_names = []

            for i in candidate_indices:

                # ------------------------------------------
                # Normal hierarchical index:
                #
                # district -> village -> [row indexes]
                #
                # ------------------------------------------

                if isinstance(
                    candidate_indices,
                    dict,
                ):

                    # This branch is handled below.
                    break

                try:

                    value_name = self.masters[
                        "village"
                    ].loc[
                        i,
                        "village_name_clean",
                    ]

                    value_name = str(
                        value_name
                    ).strip()

                    if value_name:

                        candidate_names.append(
                            value_name
                        )

                except (
                    KeyError,
                    TypeError,
                ):

                    continue

        else:

            candidate_names = []

        # ----------------------------------------------------
        # Hierarchical indexes from master_loader.py are:
        #
        # village_by_district:
        # {
        #   "AGRA": {
        #       "ABC VILLAGE": [123, 456]
        #   }
        # }
        #
        # Therefore convert dictionary -> names.
        # ----------------------------------------------------

        if (
            effective_district
            and subdistrict
        ):

            hierarchical = (
                self.village_by_district_subdistrict.get(
                    (
                        effective_district,
                        subdistrict,
                    )
                )
            )

            if isinstance(
                hierarchical,
                dict,
            ):

                candidate_names = list(
                    hierarchical.keys()
                )

        elif subdistrict:

            hierarchical = (
                self.village_by_subdistrict.get(
                    subdistrict
                )
            )

            if isinstance(
                hierarchical,
                dict,
            ):

                candidate_names = list(
                    hierarchical.keys()
                )

        elif effective_district:

            hierarchical = (
                self.village_by_district.get(
                    effective_district
                )
            )

            if isinstance(
                hierarchical,
                dict,
            ):

                candidate_names = list(
                    hierarchical.keys()
                )

        # ====================================================
        # TOKEN SEARCH FALLBACK
        #
        # IMPORTANT FIX:
        #
        # Token indexes contain NAMES, not dataframe indexes.
        # Therefore we directly use the returned names.
        # ====================================================

        if not candidate_names:

            candidate_names = (
                self._token_candidate_names(
                    query,
                    self.village_token_index,
                )
            )

        if not candidate_names:

            return Resolution(
                "VILLAGE",
                raw,
                query,
                "FAIL",
                source="LGD",
                reason=(
                    "No bounded village candidate "
                    "was found from the available "
                    "geography context."
                ),
            )

        # Remove duplicates.
        candidate_names = list(
            dict.fromkeys(
                candidate_names
            )
        )

        # ====================================================
        # FUZZY MATCH
        # ====================================================

        candidates = self._fuzzy_candidates(
            query,
            candidate_names,
            limit=3,
        )

        if not candidates:

            return Resolution(
                "VILLAGE",
                raw,
                query,
                "FAIL",
                source="LGD",
                reason=(
                    "Village could not be resolved "
                    "in LGD."
                ),
            )

        best, score = candidates[0]

        second_score = (
            candidates[1][1]
            if len(candidates) > 1
            else 0.0
        )

        # ----------------------------------------------------
        # Retrieve rows for best name.
        # ----------------------------------------------------

        best_rows = self._rows(
            "village",
            self.village_by_name.get(
                best,
                [],
            ),
        )

        best_rows = self._context_filter(
            best_rows,
            state,
            effective_district,
            pin,
            subdistrict,
        )

        if subdistrict:

            sub_rows = [
                row
                for row in best_rows
                if self._row_subdistrict(row)
                == subdistrict
            ]

            if sub_rows:

                best_rows = sub_rows

        # ====================================================
        # AUTO PASS
        # ====================================================

        if (
            len(best_rows) == 1
            and score >= FUZZY_PASS
            and (
                score - second_score >= 0.03
                or score >= FUZZY_VERY_STRONG
            )
        ):

            return self._resolution_from_row(
                "VILLAGE",
                raw,
                query,
                "PASS",
                best,
                score,
                best_rows[0],
                "LGD_FUZZY",
                (
                    "Village resolved with a strong "
                    "fuzzy match and geographic context."
                ),
            )

        # ====================================================
        # AMBIGUOUS
        # ====================================================

        return Resolution(
            "VILLAGE",
            raw,
            query,
            "AMBIGUOUS",
            best,
            score,
            source="LGD_FUZZY",
            reason=(
                "Village has a plausible fuzzy match "
                "but is not sufficiently unique for "
                "automatic acceptance."
            ),
        )

    # ========================================================
    # ULB
    # ========================================================

    def resolve_ulb(
        self,
        value,
        state="",
        district="",
        pin="",
    ) -> Resolution:

        raw = (
            ""
            if value is None
            else str(value)
        )

        query = normalize_entity(
            value
        )

        if not query:

            return Resolution(
                "ULB",
                raw,
                query,
                "NOT_CHECKED",
                reason="No city/ULB supplied.",
            )

        # ----------------------------------------------------
        # Exact
        # ----------------------------------------------------

        rows = self._rows(
            "ulb",
            self.ulb_by_name.get(
                query,
                [],
            ),
        )

        if len(rows) == 1:

            return Resolution(
                "ULB",
                raw,
                query,
                "PASS",
                query,
                1.0,
                source="LGD_EXACT",
                reason=(
                    "City/ULB value matched a unique "
                    "LGD local-body name. This is treated "
                    "as a ULB match only, not proof that "
                    "every City value is a ULB."
                ),
            )

        if len(rows) > 1:

            return Resolution(
                "ULB",
                raw,
                query,
                "AMBIGUOUS",
                query,
                1.0,
                source="LGD_EXACT",
                reason=(
                    "Multiple LGD ULB records matched "
                    "this name."
                ),
            )

        # ----------------------------------------------------
        # Fuzzy
        # ----------------------------------------------------

        candidates = self._fuzzy_candidates(
            query,
            self.ulb_by_name.keys(),
            limit=3,
        )

        if not candidates:

            return Resolution(
                "ULB",
                raw,
                query,
                "NOT_CHECKED",
                source="LGD",
                reason=(
                    "City value did not resolve as an "
                    "LGD ULB. This does not mean that "
                    "the city value is invalid."
                ),
            )

        best, score = candidates[0]

        rows = self._rows(
            "ulb",
            self.ulb_by_name.get(
                best,
                [],
            ),
        )

        if (
            len(rows) == 1
            and score >= FUZZY_PASS
        ):

            return Resolution(
                "ULB",
                raw,
                query,
                "PASS",
                best,
                score,
                source="LGD_FUZZY",
                reason=(
                    "City value has a strong fuzzy "
                    "match to a unique LGD ULB."
                ),
            )

        return Resolution(
            "ULB",
            raw,
            query,
            "AMBIGUOUS",
            best,
            score,
            source="LGD_FUZZY",
            reason=(
                "City value has a plausible ULB "
                "match but should not be auto-corrected "
                "without review."
            ),
        )

    # ========================================================
    # POST OFFICE
    # ========================================================

    def resolve_post_office(
        self,
        value,
        pin="",
        state="",
    ) -> Resolution:

        raw = (
            ""
            if value is None
            else str(value)
        )

        query = normalize_entity(
            value
        )

        pin = normalize_pin(
            pin
        )

        if not query:

            return Resolution(
                "POST_OFFICE",
                raw,
                query,
                "NOT_CHECKED",
                reason="No post office supplied.",
            )

        offices = {
            normalize_entity(x)
            for x in self.pin_post_offices.get(
                pin,
                set(),
            )
            if x
        }

        # ----------------------------------------------------
        # Exact
        # ----------------------------------------------------

        if query in offices:

            return Resolution(
                "POST_OFFICE",
                raw,
                query,
                "PASS",
                query,
                1.0,
                source="INDIA_POST_EXACT",
                reason=(
                    "Post office matched an India Post "
                    "office for the supplied PIN."
                ),
            )

        if not offices:

            return Resolution(
                "POST_OFFICE",
                raw,
                query,
                "NOT_CHECKED",
                reason=(
                    "No India Post post-office "
                    "candidates were found for the "
                    "supplied PIN."
                ),
            )

        # ----------------------------------------------------
        # Fuzzy
        # ----------------------------------------------------

        candidates = self._fuzzy_candidates(
            query,
            offices,
            limit=3,
        )

        if not candidates:

            return Resolution(
                "POST_OFFICE",
                raw,
                query,
                "FAIL",
                source="INDIA_POST",
                reason=(
                    "Post office does not match the "
                    "supplied PIN candidates."
                ),
            )

        best, score = candidates[0]

        second = (
            candidates[1][1]
            if len(candidates) > 1
            else 0.0
        )

        if (
            score >= FUZZY_PASS
            and (
                score - second >= 0.03
                or score >= FUZZY_VERY_STRONG
            )
        ):

            return Resolution(
                "POST_OFFICE",
                raw,
                query,
                "PASS",
                best,
                score,
                source="INDIA_POST_FUZZY",
                reason=(
                    "Post office resolved strongly "
                    "within the supplied PIN."
                ),
            )

        return Resolution(
            "POST_OFFICE",
            raw,
            query,
            "AMBIGUOUS",
            best,
            score,
            source="INDIA_POST_FUZZY",
            reason=(
                "Post office has a plausible match "
                "within the supplied PIN but needs "
                "review."
            ),
        )

    # ========================================================
    # COMPLETE ADDRESS RESOLUTION
    # ========================================================

    def resolve_address(
        self,
        address,
        city="",
        state="",
        district="",
        subdistrict="",
        village="",
        post_office="",
        pin="",
    ):

        state = canonical_state(
            state
        )

        pin = normalize_pin(
            pin
        )

        supplied_district = normalize_entity(
            district
        )

        supplied_subdistrict = normalize_entity(
            subdistrict
        )

        supplied_village = normalize_entity(
            village
        )

        # ====================================================
        # DISTRICT
        # ====================================================

        if supplied_district:

            district_res = (
                self.resolve_district(
                    supplied_district,
                    state,
                    pin,
                )
            )

        else:

            district_res = Resolution(
                "DISTRICT",
                "",
                "",
                "NOT_CHECKED",
                reason="No district supplied.",
            )

        # ----------------------------------------------------
        # City may be used as district fallback ONLY when
        # it independently resolves as a unique district.
        #
        # We do NOT assume:
        #
        # City = District
        #
        # and we do NOT assume:
        #
        # City = ULB
        # ----------------------------------------------------

        if (
            district_res.status
            not in {
                "PASS",
                "AMBIGUOUS",
            }
            and city
        ):

            city_district = (
                self.resolve_district(
                    city,
                    state,
                    pin,
                )
            )

            if city_district.status == "PASS":

                district_res = city_district

        effective_district = (
            district_res.match_value
            if district_res.status
            in {
                "PASS",
                "AMBIGUOUS",
            }
            else supplied_district
        )

        # ====================================================
        # SUBDISTRICT
        # ====================================================

        if supplied_subdistrict:

            subdistrict_res = (
                self.resolve_subdistrict(
                    supplied_subdistrict,
                    state,
                    effective_district,
                    pin,
                )
            )

        else:

            subdistrict_res = Resolution(
                "SUBDISTRICT",
                "",
                "",
                "NOT_CHECKED",
                reason="No subdistrict supplied.",
            )

        # ====================================================
        # EFFECTIVE SUBDISTRICT
        # ====================================================

        effective_subdistrict = (
            subdistrict_res.match_value
            if subdistrict_res.status
            in {
                "PASS",
                "AMBIGUOUS",
            }
            else supplied_subdistrict
        )

        # ====================================================
        # VILLAGE
        # ====================================================

        village_res = self.resolve_village(
            supplied_village,
            state,
            effective_district,
            effective_subdistrict,
            pin,
        )

        # ====================================================
        # POST OFFICE
        # ====================================================

        post_office_res = (
            self.resolve_post_office(
                post_office,
                pin,
                state,
            )
        )

        # ====================================================
        # ULB / CITY
        # ====================================================

        if city:

            ulb_res = self.resolve_ulb(
                city,
                state,
                effective_district,
                pin,
            )

        else:

            ulb_res = Resolution(
                "ULB",
                "",
                "",
                "NOT_CHECKED",
                reason="No city supplied.",
            )

        # ====================================================
        # RETURN
        # ====================================================

        return {

            "district":
                district_res,

            "subdistrict":
                subdistrict_res,

            "village":
                village_res,

            "post_office":
                post_office_res,

            "ulb":
                ulb_res,
        }


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "entity_resolution.py loaded successfully."
    )

    print(
        f"RapidFuzz available: "
        f"{RAPIDFUZZ_AVAILABLE}"
    )