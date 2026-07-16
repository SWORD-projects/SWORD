"""ICSD-oriented prescreening, labelling, and deduplication workflow."""

from __future__ import annotations

import inspect
import json
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import pandas as pd
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifParser

from .family import SWORDFamilyMatcher
from .label import get_sword_info, get_sword_info_for_ICSD
from .utils import dedupe_by_dom_projection, dedupe_by_vector_dom, group_ICSD


_ANOMALY_KEYS = (
    "label_errors",
    "wyck_float_warn",
    "occ_err_sites",
    "equivalent_but_distinct_sites",
    "same_valence_sites",
    "intersect_orb_errors",
)


def _element_from_token(token):
    token = str(token)
    if token.upper() == "VAC":
        return "VAC"
    match = re.match(r"[A-Z][a-z]?", token)
    return match.group(0) if match else token


def _label_elements(labels):
    return [_element_from_token(label) for label in labels if label]


def _get_first_cif_block(cif_text, *, parser_occ_tolerance=1.05):
    parser = CifParser(StringIO(cif_text), occupancy_tolerance=parser_occ_tolerance)
    cif_dict = parser.as_dict()
    return cif_dict[next(iter(cif_dict))]


def _contains_hydrogen(labels):
    return bool(set(_label_elements(labels)) & {"H", "D", "T"})


def _get_non_elements(labels, excluded_elements=None):
    periodic_table = {element.symbol for element in Element} | {"D", "T"}
    known_elements = periodic_table - set(excluded_elements or [])
    return sorted(element for element in set(_label_elements(labels)) if element not in known_elements)


def _num_or_none(token):
    token = str(token)
    token = re.sub(r"\(.*\)", "", token)
    try:
        return float(token)
    except Exception:
        return None


def _check_cif_format(block, *, parser_occ_tolerance):
    xs = [_num_or_none(x) for x in block.get("_atom_site_fract_x", [])]
    ys = [_num_or_none(y) for y in block.get("_atom_site_fract_y", [])]
    zs = [_num_or_none(z) for z in block.get("_atom_site_fract_z", [])]
    occupancies = [_num_or_none(x) for x in block.get("_atom_site_occupancy", [])]
    multiplicities = [str(x) for x in block.get("_atom_site_symmetry_multiplicity", [])]
    wyckoffs = [str(x) for x in block.get("_atom_site_Wyckoff_symbol", [])]

    reasons = []
    if not xs or not ys or not zs or any(value is None for value in xs + ys + zs):
        reasons.append("coordinate_error")
    if not wyckoffs or any(not value.isalpha() for value in wyckoffs if value):
        reasons.append("wyckoff_error")
    if not multiplicities or any(not value.isdigit() for value in multiplicities if value):
        reasons.append("multiplicity_error")
    if not occupancies or any(
        value is None or value <= 0 or value > parser_occ_tolerance
        for value in occupancies
    ):
        reasons.append("occupancy_error")
    return reasons


def _get_icsd_reject_reason(
    row,
    *,
    cif_col,
    id_col,
    parser_occ_tolerance,
    excluded_elements,
    exclude_hydrogen,
):
    try:
        block = _get_first_cif_block(
            row[cif_col],
            parser_occ_tolerance=max(float(parser_occ_tolerance), 1.0),
        )
    except Exception:
        return "parse_error"

    labels = list(block.get("_atom_site_label", []))
    reasons = []
    if exclude_hydrogen and _contains_hydrogen(labels):
        reasons.append("hydrogen")
    if _get_non_elements(labels, excluded_elements):
        reasons.append("non_element")
    reasons.extend(_check_cif_format(block, parser_occ_tolerance=parser_occ_tolerance))
    return ";".join(dict.fromkeys(reasons)) or None


def prescreen_icsd_dataframe(
    df,
    *,
    cif_col="cif",
    id_col="CollectionCode",
    parser_occ_tolerance=1.05,
    excluded_elements=("He", "Ne", "Ar", "Kr", "Es"),
    exclude_hydrogen=True,
):
    """Filter an ICSD-style dataframe before SWORD labelling.

    The input dataframe is expected to contain one CIF text column and one
    stable entry identifier column. Rows are rejected when the CIF cannot be
    parsed, contains H/D/T when requested, contains non-element site labels,
    has malformed coordinates/Wyckoff letters/multiplicities, or has site
    occupancy values <= 0 or greater than ``parser_occ_tolerance``.

    Args:
        df: Input dataframe containing ICSD-style CIF records.
        cif_col: Column containing raw CIF text.
        id_col: Column containing a stable entry identifier, such as
            ``CollectionCode``.
        parser_occ_tolerance: Maximum allowed raw CIF single-site occupancy.
        excluded_elements: Periodic-table elements to treat as invalid for this
            screen.
        exclude_hydrogen: If True, reject entries containing H, D, or T.

    Returns:
        ``(screened_df, rejected_df)``. ``screened_df`` preserves the original
        columns for rows that pass. ``rejected_df`` contains ``id_col`` and
        ``reject_reason``.
    """
    rows = []
    for idx, row in df.iterrows():
        reason = _get_icsd_reject_reason(
            row,
            cif_col=cif_col,
            id_col=id_col,
            parser_occ_tolerance=parser_occ_tolerance,
            excluded_elements=excluded_elements,
            exclude_hydrogen=exclude_hydrogen,
        )
        if reason:
            rows.append({id_col: row[id_col], "reject_reason": reason, "_idx": idx})

    if rows:
        rejected = pd.DataFrame(rows)
        screened = df.drop(index=rejected["_idx"]).copy()
        rejected = rejected.drop(columns=["_idx"])
    else:
        rejected = pd.DataFrame(columns=[id_col, "reject_reason"])
        screened = df.copy()
    return screened, rejected


def _filter_kwargs(fn, params, *, exclude=()):
    params = dict(params or {})
    allowed = set(inspect.signature(fn).parameters)
    blocked = set(exclude)
    return {key: value for key, value in params.items() if key in allowed and key not in blocked}


def _issue_from_exception(exc):
    text = repr(exc)
    parse_markers = ("CifParser", "Invalid CIF", "parse", "no structures", "StructureEntry")
    if any(marker.lower() in text.lower() for marker in parse_markers):
        return "structure_parse_failed"
    return "label_generation_failed"


def _anomaly_frame(rows, id_col):
    if not rows:
        return pd.DataFrame(columns=[id_col])
    return pd.DataFrame(rows)


def _add_anomaly_rows(anomalies, key, records, *, id_col, entry_id):
    if not records:
        return
    if not isinstance(records, list):
        records = [records]
    for record in records:
        if isinstance(record, dict):
            row = dict(record)
        else:
            row = {"value": record}
        row[id_col] = entry_id
        anomalies[key].append(row)


def _dom_summary(dom_info):
    if not dom_info:
        return None, None, None, None

    merged_occ = dom_info.get("representative_site_merged_occ") or {}
    by_element = {}
    for token, occ in merged_occ.items():
        element = _element_from_token(token)
        by_element[element] = by_element.get(element, 0.0) + float(occ)

    if not by_element:
        return None, None, None, None

    items = sorted(by_element.items())
    elements = [element for element, _ in items]
    occupancies = [round(occ, 6) for _, occ in items]
    dominant_element = max(items, key=lambda item: (item[1], item[0]))[0]
    occ_vector = dom_info.get("representative_site_occ_vector")
    return elements, occupancies, dominant_element, occ_vector


def _empty_label_columns():
    return {
        "SWORD_label": None,
        "is_disorder": None,
        "is_vac_disorder": None,
        "is_sub_disorder": None,
        "is_positional_disorder": None,
        "degree_of_mixing": None,
        "processing_issue": None,
        "dom_site_elements": None,
        "dom_site_occupancies": None,
        "dom_dominant_element": None,
        "dom_site_occ_vector": None,
    }


def _label_dataframe_impl(
    df,
    *,
    cif_col="cif",
    id_col,
    mode,
    sword_params=None,
    family_info=False,
    family_params=None,
):
    if mode not in {"from_collection_code", "from_pmg"}:
        raise ValueError("mode must be 'from_collection_code' or 'from_pmg'")

    work_df = df.copy()
    if mode == "from_collection_code":
        lookup_df = work_df.copy()
        if id_col != "CollectionCode":
            lookup_df["CollectionCode"] = lookup_df[id_col]
        if cif_col != "cif":
            lookup_df["cif"] = lookup_df[cif_col]
    else:
        lookup_df = None

    family_matcher = SWORDFamilyMatcher(**dict(family_params or {})) if family_info else None
    anomalies = {key: [] for key in _ANOMALY_KEYS}
    rows = []

    for _, row in work_df.iterrows():
        entry_id = row[id_col]
        out = row.to_dict()
        out.update(_empty_label_columns())
        if family_info:
            out["SWORD_family_dic"] = None

        try:
            if mode == "from_collection_code":
                params = _filter_kwargs(
                    get_sword_info_for_ICSD,
                    sword_params,
                    exclude=("collection_code", "ICSD_df"),
                )
                _, info = get_sword_info_for_ICSD(
                    entry_id,
                    ICSD_df=lookup_df,
                    **params,
                )
            else:
                params = _filter_kwargs(get_sword_info, sword_params)
                _, info = get_sword_info(row[cif_col], **params)

            dom_elements, dom_occs, dom_dominant, dom_occ_vector = _dom_summary(info.get("dom_info"))
            out.update(
                {
                    "SWORD_label": info.get("disorder_label"),
                    "is_disorder": info.get("is_disorder"),
                    "is_vac_disorder": info.get("is_vac_disorder"),
                    "is_sub_disorder": info.get("is_sub_disorder"),
                    "is_positional_disorder": info.get("is_positional_disorder"),
                    "degree_of_mixing": info.get("degree_of_mixing"),
                    "dom_site_elements": dom_elements,
                    "dom_site_occupancies": dom_occs,
                    "dom_dominant_element": dom_dominant,
                    "dom_site_occ_vector": dom_occ_vector,
                }
            )

            issues = []
            if info.get("wyck_float_warn"):
                issues.append("wyckoff_float_warning")
                _add_anomaly_rows(
                    anomalies, "wyck_float_warn", info.get("wyck_float_warn"), id_col=id_col, entry_id=entry_id
                )
            if info.get("occ_err_sites"):
                issues.append("occupancy_orbit_error")
                _add_anomaly_rows(
                    anomalies, "occ_err_sites", info.get("occ_err_sites"), id_col=id_col, entry_id=entry_id
                )
            if info.get("equivalent_but_distinct_sites"):
                issues.append("equivalent_sites_warning")
                _add_anomaly_rows(
                    anomalies,
                    "equivalent_but_distinct_sites",
                    info.get("equivalent_but_distinct_sites"),
                    id_col=id_col,
                    entry_id=entry_id,
                )
            if info.get("same_valence_sites"):
                issues.append("same_valence_site_warning")
                _add_anomaly_rows(
                    anomalies,
                    "same_valence_sites",
                    info.get("same_valence_sites"),
                    id_col=id_col,
                    entry_id=entry_id,
                )
            if info.get("intersect_orb_error"):
                issues.append("positional_check_failed")
                _add_anomaly_rows(
                    anomalies,
                    "intersect_orb_errors",
                    {"error": info.get("intersect_orb_error")},
                    id_col=id_col,
                    entry_id=entry_id,
                )
            out["processing_issue"] = ";".join(issues) or None

            if family_matcher is not None:
                out["SWORD_family_dic"] = family_matcher.get_sword_dic(
                    row[cif_col],
                    child_label=out["SWORD_label"],
                )

        except Exception as exc:
            issue = _issue_from_exception(exc)
            out["processing_issue"] = issue
            anomalies["label_errors"].append(
                {
                    id_col: entry_id,
                    "error_type": issue,
                    "error_message": repr(exc),
                    "mode": mode,
                }
            )

        rows.append(out)

    anomaly_tables = {
        key: _anomaly_frame(rows, id_col)
        for key, rows in anomalies.items()
    }
    return pd.DataFrame(rows), anomaly_tables


def label_dataframe(
    df,
    *,
    cif_col="cif",
    id_col="id",
    sword_params=None,
    family_info=False,
    family_params=None,
):
    """Append SWORD labels to a generic dataframe containing CIF text.

    This is the database-agnostic dataframe labelling interface. It always
    parses each row through the pymatgen/SWORD text path and therefore accepts
    any dataframe with a CIF text column and an ID column.

    Args:
        df: Input dataframe.
        cif_col: Column containing CIF text.
        id_col: Column containing a stable row identifier, such as
            ``material_id``, ``ICSD_CollectionCode``.
        sword_params: Optional keyword arguments passed to
            ``get_sword_info()``, such as ``symprec``,
            ``parser_occ_tolerance``, ``site_tolerance``, ``occ_tolerance``,
            ``vac_tolerance``, and ``frac_tolerance``.
        family_info: If True, append ``SWORD_family_dic`` using
            ``SWORDFamilyMatcher``. This is slower than label generation and is
            disabled by default.
        family_params: Optional dictionary passed to ``SWORDFamilyMatcher``,
            such as ``{"fill_vacancy": False}``.

    Returns:
        ``(label_results, anomalies)``. ``label_results`` is a copy of the
        input dataframe with compact SWORD result columns appended. ``anomalies``
        is a dictionary of detailed warning/error tables.
    """
    return _label_dataframe_impl(
        df,
        cif_col=cif_col,
        id_col=id_col,
        mode="from_pmg",
        sword_params=sword_params,
        family_info=family_info,
        family_params=family_params,
    )


def label_icsd_dataframe(
    df,
    *,
    cif_col="cif",
    id_col="CollectionCode",
    mode="from_collection_code",
    sword_params=None,
    family_info=False,
    family_params=None,
):
    """Append SWORD labels to an ICSD-style dataframe.

    By default this uses the ICSD-oriented raw CIF path keyed by
    ``CollectionCode``. Set ``mode='from_pmg'`` to parse the CIF text through
    the pymatgen/SWORD text path instead.

    Args:
        df: Input ICSD-style dataframe.
        cif_col: Column containing CIF text.
        id_col: Column containing the entry identifier. Defaults to
            ``CollectionCode``.
        mode: ``'from_collection_code'`` for the ICSD raw-CIF interface or
            ``'from_pmg'`` for the generic pymatgen/SWORD text interface.
        sword_params: Optional keyword arguments passed to the selected SWORD
            labelling function.
        family_info: If True, append ``SWORD_family_dic`` using
            ``SWORDFamilyMatcher``.
        family_params: Optional dictionary passed to ``SWORDFamilyMatcher``.

    Returns:
        ``(label_results, anomalies)``. ``label_results`` preserves the input
        columns and appends SWORD label, disorder flags, DOM, processing issue,
        and compact DOM projection fields. ``anomalies`` contains detailed
        per-entry diagnostic tables.
    """
    return _label_dataframe_impl(
        df,
        cif_col=cif_col,
        id_col=id_col,
        mode=mode,
        sword_params=sword_params,
        family_info=family_info,
        family_params=family_params,
    )


def _build_label_groups(label_results):
    labelled = label_results.dropna(subset=["SWORD_label"]).copy()
    if labelled.empty:
        return pd.DataFrame()
    return group_ICSD(labelled, sword_col="SWORD_label")


def _build_refined_results(
    label_results,
    *,
    id_col,
    disorder_refinement_policy,
    dom_distance_tol,
    composition_tol,
    dom_tol,
    drop_hard_errors=True,
    drop_positional_disorder=True,
    drop_occupancy_orbit_error=True,
    drop_wyckoff_float_warning=True,
    drop_equivalent_sites_warning=True,
    drop_same_valence_site_warning=False,
):
    if disorder_refinement_policy not in {"dom_project", "dom_vector"}:
        raise ValueError(
            "disorder_refinement_policy must be 'dom_project' or 'dom_vector'."
        )

    df = label_results.dropna(subset=["SWORD_label"]).copy()
    if df.empty:
        return df

    issues = df["processing_issue"].fillna("")
    remove = pd.Series(False, index=df.index)
    if drop_hard_errors:
        remove |= issues.str.contains("structure_parse_failed|label_generation_failed", regex=True)
    if drop_positional_disorder:
        remove |= df["is_positional_disorder"].fillna(False).astype(bool)
        remove |= issues.str.contains("positional_check_failed", regex=False)
    if drop_occupancy_orbit_error:
        remove |= issues.str.contains("occupancy_orbit_error", regex=False)
    if drop_wyckoff_float_warning:
        remove |= issues.str.contains("wyckoff_float_warning", regex=False)
    if drop_equivalent_sites_warning:
        remove |= issues.str.contains("equivalent_sites_warning", regex=False)
    if drop_same_valence_site_warning:
        remove |= issues.str.contains("same_valence_site_warning", regex=False)

    df = df[~remove].copy()
    if df.empty:
        return df

    keep_ids = []
    for _, group in df.groupby("SWORD_label", sort=False):
        ordered = group[~group["is_disorder"].fillna(False).astype(bool)]
        disorder = group[group["is_disorder"].fillna(False).astype(bool)]

        if not ordered.empty:
            keep_ids.append(ordered.sort_values(id_col).iloc[0][id_col])

        if not disorder.empty:
            if disorder_refinement_policy == "dom_project":
                disorder_keep_ids = dedupe_by_dom_projection(
                    disorder,
                    id_col=id_col,
                    dom_col="degree_of_mixing",
                    elements_col="dom_site_elements",
                    occupancies_col="dom_site_occupancies",
                    dominant_element_col="dom_dominant_element",
                    tol=dom_distance_tol,
                )
            else:
                disorder_keep_ids = dedupe_by_vector_dom(
                    disorder,
                    id_col=id_col,
                    dom_col="degree_of_mixing",
                    vector_col="dom_site_occ_vector",
                    composition_tol=composition_tol,
                    dom_tol=dom_tol,
                )
            keep_ids.extend(disorder_keep_ids)

    return df[df[id_col].isin(keep_ids)].copy()


@dataclass
class ICSDDedupResult:
    """Container returned by ``run_icsd_dedup_pipeline``.

    Attributes:
        prescreen_rejected: Small table with rejected entry IDs and reasons.
        label_results: Main labelled dataframe.
        anomalies: Dictionary of detailed warning/error dataframes.
        label_groups: Grouped SWORD-label summary table.
        refined: Labelled dataframe after configured curation filtering,
            ordered-label deduplication, and DOM-projection deduplication.
        metadata: Parameters used to produce the result.
    """

    prescreen_rejected: pd.DataFrame
    label_results: pd.DataFrame
    anomalies: dict
    label_groups: pd.DataFrame
    refined: pd.DataFrame
    metadata: dict

    def save(self, path):
        """Save all standard pipeline outputs under ``path``.

        The method writes the main tables as pickle files, the rejected
        prescreen table and anomaly tables as CSV files, and run metadata as
        JSON. It intentionally does not save a full prescreen-passed copy,
        because ``label_results`` already contains the rows that entered the
        labelling stage.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.prescreen_rejected.to_csv(path / "prescreen_rejected.csv", index=False)
        self.label_results.to_pickle(path / "label_results.pkl")
        self.label_groups.to_pickle(path / "label_groups.pkl")
        self.refined.to_pickle(path / "refined_results.pkl")

        anomaly_dir = path / "anomalies"
        anomaly_dir.mkdir(exist_ok=True)
        for key, table in self.anomalies.items():
            table.to_csv(anomaly_dir / f"{key}.csv", index=False)

        (path / "run_metadata.json").write_text(
            json.dumps(self.metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )


def run_icsd_dedup_pipeline(
    df,
    *,
    cif_col="cif",
    id_col="CollectionCode",
    mode="from_collection_code",
    prescreen_params=None,
    sword_params=None,
    family_info=False,
    family_params=None,
    output_dir=None,
    disorder_refinement_policy="dom_vector",
    dom_distance_tol=0.03,
    composition_tol=0.05,
    dom_tol=0.03,
    drop_hard_errors=True,
    drop_positional_disorder=True,
    drop_occupancy_orbit_error=True,
    drop_wyckoff_float_warning=True,
    drop_equivalent_sites_warning=True,
    drop_same_valence_site_warning=False,
):
    """Run the ICSD prescreening, labelling, grouping, and dedup workflow.

    This high-level convenience function is intended for ICSD-style dataframes.
    It performs prescreening, labels all prescreen-passed entries, builds a
    grouped SWORD-label table, and creates a refined table by applying the
    configured curation filters before deduplicating ordered entries by SWORD
    label and disordered entries by DOM projection.

    Args:
        df: Input ICSD-style dataframe.
        cif_col: Column containing CIF text.
        id_col: Column containing the entry identifier.
        mode: Labelling mode passed to ``label_icsd_dataframe``. Use
            ``'from_collection_code'`` for ICSD raw-CIF labelling or
            ``'from_pmg'`` for pymatgen/SWORD text labelling.
        prescreen_params: Optional keyword arguments for
            ``prescreen_icsd_dataframe``.
        sword_params: Optional keyword arguments for the selected SWORD
            labelling function.
        family_info: If True, append a ``SWORD_family_dic`` column to
            ``label_results`` using ``SWORDFamilyMatcher``.
        family_params: Optional dictionary passed to ``SWORDFamilyMatcher``.
        output_dir: If provided, save all standard outputs to this directory.
        disorder_refinement_policy: Disorder deduplication policy. Use
            ``'dom_vector'`` (default) or the legacy ``'dom_project'``.
        dom_distance_tol: Distance tolerance used by ``'dom_project'``.
        composition_tol: Occupancy-vector tolerance used by ``'dom_vector'``.
        dom_tol: Absolute-DOM tolerance used by ``'dom_vector'``.
        drop_hard_errors: Remove structure parse and label generation failures.
        drop_positional_disorder: Remove positional-disorder entries and
            positional-check failures.
        drop_occupancy_orbit_error: Remove entries with orbit occupancy errors.
        drop_wyckoff_float_warning: Remove entries with Wyckoff floating-point
            warnings.
        drop_equivalent_sites_warning: Remove entries with equivalent-site
            coordinate warnings.
        drop_same_valence_site_warning: Remove entries with same-valence
            warnings. Defaults to False because these are usually diagnostic
            records rather than hard failures.

    Returns:
        ``ICSDDedupResult`` containing rejected entries, labelled rows, anomaly
        tables, grouped labels, refined rows, and metadata.
    """
    prescreen_kwargs = dict(prescreen_params or {})
    screened, rejected = prescreen_icsd_dataframe(
        df,
        cif_col=cif_col,
        id_col=id_col,
        **prescreen_kwargs,
    )
    label_results, anomalies = label_icsd_dataframe(
        screened,
        cif_col=cif_col,
        id_col=id_col,
        mode=mode,
        sword_params=sword_params,
        family_info=family_info,
        family_params=family_params,
    )
    label_groups = _build_label_groups(label_results)
    curation_filters = {
        "drop_hard_errors": drop_hard_errors,
        "drop_positional_disorder": drop_positional_disorder,
        "drop_occupancy_orbit_error": drop_occupancy_orbit_error,
        "drop_wyckoff_float_warning": drop_wyckoff_float_warning,
        "drop_equivalent_sites_warning": drop_equivalent_sites_warning,
        "drop_same_valence_site_warning": drop_same_valence_site_warning,
    }
    refined = _build_refined_results(
        label_results,
        id_col=id_col,
        disorder_refinement_policy=disorder_refinement_policy,
        dom_distance_tol=dom_distance_tol,
        composition_tol=composition_tol,
        dom_tol=dom_tol,
        **curation_filters,
    )

    result = ICSDDedupResult(
        prescreen_rejected=rejected,
        label_results=label_results,
        anomalies=anomalies,
        label_groups=label_groups,
        refined=refined,
        metadata={
            "cif_col": cif_col,
            "id_col": id_col,
            "mode": mode,
            "prescreen_params": prescreen_kwargs,
            "sword_params": dict(sword_params or {}),
            "family_info": family_info,
            "family_params": dict(family_params or {}),
            "disorder_refinement_policy": disorder_refinement_policy,
            "dom_distance_tol": dom_distance_tol,
            "composition_tol": composition_tol,
            "dom_tol": dom_tol,
            "curation_filters": curation_filters,
        },
    )
    if output_dir is not None:
        result.save(output_dir)
    return result
