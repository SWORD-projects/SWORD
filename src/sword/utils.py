import math
import re

import pandas as pd


def group_ICSD(ICSD, sword_col="SWORD_label"):
    """Group rows by a SWORD label column and summarize matching ICSD entries."""
    ICSD = ICSD.copy()

    comp_sg_keys = []
    for label in ICSD[sword_col].astype(str):
        parts = label.split("_")
        sg_idx = next(i for i, tok in enumerate(parts) if tok.isdigit())
        sg = int(parts[sg_idx])
        elems = tuple(sorted(set(re.findall(r"[A-Z][a-z]?", "_".join(parts[sg_idx + 1 :])))))
        comp_sg_keys.append((sg, elems))

    ICSD["comp_sg_key"] = comp_sg_keys

    return (
        ICSD.groupby([sword_col], as_index=False)
        .agg(
            n_total=(sword_col, "size"),
            CollectionCode_list=("CollectionCode", list),
            StructuredFormula_list=("StructuredFormula", list),
            comp_sg_key=("comp_sg_key", "first"),
            WyckoffSequence_lists=("WyckoffSequence", list),
            degree_of_mixing_list=("degree_of_mixing", list),
            n_disorder=("is_disorder", "sum"),
        )
    )


def filter_by_elements_and_sg(
    df,
    col="comp_sg_key",
    req_elements=("Li", "Co", "O"),
    req_sgs=(166,),
    only=False,
):
    """Filter grouped rows by required elements and optional space-group numbers."""
    req_elems = {str(e) for e in req_elements}
    req_sgs = None if req_sgs is None else {int(s) for s in req_sgs}

    def _match(key):
        sg, elems = key
        elems = set(elems)

        elems_ok = elems == req_elems if only else req_elems.issubset(elems)
        sg_ok = True if req_sgs is None else sg in req_sgs
        return elems_ok and sg_ok

    return df[df[col].apply(_match)]


def find_by_disorder_label(
    df,
    pattern,
    col="SWORD_label",
    req_elements=None,
    comp_sg_col="comp_sg_key",
    sword_col="SWORD_label",
):
    """Find labels containing pattern; optionally require elements parsed from SWORD labels."""
    mask = df[col].astype(str).str.contains(pattern, regex=False, na=False)

    if req_elements is not None:
        req_elements = {str(e) for e in req_elements}

        if comp_sg_col in df.columns:
            elem_mask = df[comp_sg_col].apply(
                lambda key: req_elements.issubset(set(key[1]))
            )
        else:
            elem_mask = df[sword_col].astype(str).apply(
                lambda label: req_elements.issubset(
                    set(
                        re.findall(
                            r"[A-Z][a-z]?",
                            "_".join(label.split("_")[next(i for i, tok in enumerate(label.split("_")) if tok.isdigit()) + 1 :]),))))

        mask &= elem_mask

    return df[mask]


def dedupe_by_same_dom(deg_list, code_list, tol=0.01):
    """Deduplicate a SWORD group using only DOM proximity.

    Two entries are treated as duplicates when their degree of mixing (DOM) differs by at most ``tol``. 
    It is compact and easy to interpret, but it can over-merge entries whose DOM values are
    similar while their underlying disordered stoichiometry patterns differ.
    """
    if deg_list is None or code_list is None:
        return code_list or []

    keep_codes = []
    seen_dom = []
    seen_nan = False

    for dom, code in zip(deg_list, code_list):
        if pd.isna(dom):
            if seen_nan:
                continue
            seen_nan = True
            keep_codes.append(code)
            continue

        dom = float(dom)
        if any(abs(dom - prev) <= tol for prev in seen_dom):
            continue

        seen_dom.append(dom)
        keep_codes.append(code)

    return keep_codes


def dedupe_by_dom_projection(
    group_df,
    *,
    id_col="CollectionCode",
    dom_col="degree_of_mixing",
    elements_col="dom_site_elements",
    occupancies_col="dom_site_occupancies",
    dominant_element_col="dom_dominant_element",
    tol=0.03,
):
    """Deduplicate a SWORD group in the native ``(x, DOM)`` projection.

    This strategy keeps the physical meaning of the DOM sign while adding the
    dominant-element occupancy ``x`` as a second coordinate. ``group_df`` should
    contain one SWORD label group and precomputed DOM summary columns; this
    function does not rerun SWORD.

    Args:
        group_df: Dataframe containing entries from one SWORD label group.
        id_col: Entry identifier column to return.
        dom_col: Column containing the SWORD degree of mixing.
        elements_col: Column containing representative disordered-site element
            lists, such as ``["Li", "Mn"]``.
        occupancies_col: Column containing occupancies aligned with
            ``elements_col``.
        dominant_element_col: Column containing the dominant representative
            element used as the projection coordinate.
        tol: Euclidean distance tolerance in ``(dominant occupancy, DOM)``.

    Returns:
        List of kept entry IDs from ``id_col``.
    """
    if group_df is None or len(group_df) == 0:
        return []

    def _as_list(value):
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        if pd.isna(value):
            return []
        return [value]

    def _dominant_occupancy(row):
        elems = [str(x) for x in _as_list(row.get(elements_col))]
        occs = _as_list(row.get(occupancies_col))
        dominant = row.get(dominant_element_col)
        if pd.isna(row.get(dom_col)) or not elems or not occs or pd.isna(dominant):
            return None
        for elem, occ in zip(elems, occs):
            if elem == dominant:
                try:
                    return float(occ)
                except Exception:
                    return None
        return None

    df = group_df.copy()
    df["_dom_projection_x"] = df.apply(_dominant_occupancy, axis=1)
    df = df.dropna(subset=[id_col, dom_col, dominant_element_col, "_dom_projection_x"])

    if df.empty:
        return group_df[id_col].dropna().tolist()

    keep_codes = []
    representatives = []

    for _, row in df.sort_values([dominant_element_col, "_dom_projection_x", dom_col]).iterrows():
        matched = False
        for rep in representatives:
            if row[dominant_element_col] != rep["dominant_element"]:
                continue
            dist = (
                (float(row["_dom_projection_x"]) - rep["x"]) ** 2
                + (float(row[dom_col]) - rep["DOM"]) ** 2
            ) ** 0.5
            if dist <= tol:
                matched = True
                break

        if not matched:
            keep_codes.append(row[id_col])
            representatives.append(
                {
                    "dominant_element": row[dominant_element_col],
                    "x": float(row["_dom_projection_x"]),
                    "DOM": float(row[dom_col]),
                }
            )

    return keep_codes


def _as_occ_vector(vector):
    if not isinstance(vector, dict) or len(vector) < 2:
        return None
    try:
        vector = {str(key): float(value) for key, value in vector.items()}
    except (TypeError, ValueError):
        return None
    return vector if all(math.isfinite(value) for value in vector.values()) else None


def vector_dom_distance(vector_a, dom_a, vector_b, dom_b, *, composition_tol=0.05, dom_tol=0.03):
    """Return the normalized joint occupancy-vector/DOM distance."""
    if composition_tol <= 0 or dom_tol <= 0:
        raise ValueError("composition_tol and dom_tol must be positive.")
    vector_a = _as_occ_vector(vector_a)
    vector_b = _as_occ_vector(vector_b)
    if vector_a is None or vector_b is None or set(vector_a) != set(vector_b):
        return None
    try:
        dom_a, dom_b = float(dom_a), float(dom_b)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(dom_a) or not math.isfinite(dom_b):
        return None

    n_components = len(vector_a)
    vector_distance = math.sqrt(sum(
        (vector_a[element] - vector_b[element]) ** 2 for element in vector_a
    ))
    composition_distance = vector_distance / math.sqrt(2 * n_components / (n_components - 1))
    dom_distance = abs(abs(dom_a) - abs(dom_b))
    return math.hypot(composition_distance / composition_tol, dom_distance / dom_tol)


def dedupe_by_vector_dom(
    group_df,
    *,
    id_col="CollectionCode",
    dom_col="degree_of_mixing",
    vector_col="dom_site_occ_vector",
    composition_tol=0.05,
    dom_tol=0.03,
):
    """Deduplicate one SWORD group by occupancy-vector and absolute-DOM distance."""
    if group_df is None or len(group_df) == 0:
        return []
    if composition_tol <= 0 or dom_tol <= 0:
        raise ValueError("composition_tol and dom_tol must be positive.")

    df = group_df.dropna(subset=[id_col]).copy()
    if vector_col not in df or dom_col not in df:
        return df[id_col].tolist()
    df["_occ_vector"] = df[vector_col].apply(_as_occ_vector)
    df["_DOM"] = pd.to_numeric(df[dom_col], errors="coerce")
    comparable = df["_occ_vector"].notna() & df["_DOM"].apply(math.isfinite)
    incomparable_ids = df.loc[~comparable, id_col].tolist()
    df = df.loc[comparable].copy()
    df["_radius"] = df["_occ_vector"].apply(
        lambda vector: math.sqrt(sum(value ** 2 for value in vector.values()))
    )
    df["_abs_DOM"] = df["_DOM"].abs()

    keep_codes, representatives = [], []
    for _, row in df.sort_values(
        ["_radius", "_abs_DOM", id_col], ascending=[True, False, True]
    ).iterrows():
        matched = False
        for representative in representatives:
            distance = vector_dom_distance(
                row["_occ_vector"], row["_DOM"],
                representative["_occ_vector"], representative["_DOM"],
                composition_tol=composition_tol,
                dom_tol=dom_tol,
            )
            if distance is not None and distance <= 1.0:
                matched = True
                break
        if not matched:
            keep_codes.append(row[id_col])
            representatives.append(row)

    keep_codes.extend(sorted(incomparable_ids))
    return keep_codes
