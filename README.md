# SWORDlib

SWORDlib provides tools for generating SWORD labels from ordered and disordered
crystal structures. A SWORD label combines symmetry, standardized Wyckoff
sequence information, and site-occupancy disorder information into a compact
representation that can be used for structure grouping, disorder curation, and
order-disorder family matching.

This package is the installable Python package form of SWORD:

## Installation

```bash
pip install SWORDlib
```

## Core SWORD Labelling

```python
import sword
```

Use `get_sword_label` when you only need the compact label.

```python
from sword import get_sword_label

label = get_sword_label("path/to/structure.cif")
print(label)
```

The input can be a CIF string, a CIF path, or a `pymatgen.core.Structure`.

Use `get_sword_info` when you also need the parsed entry and diagnostic
metadata.

```python
from sword import get_sword_info

entry, info = get_sword_info(
    cif_text,
    parser_occ_tolerance=1.05,
    occ_tolerance=1.0,
    site_tolerance=1e-4,
    vac_tolerance=1e-2,
    frac_tolerance=1e-4,
)

print(info["disorder_label"])
print(info["is_disorder"])
print(info["degree_of_mixing"])
```

The main labelling parameters are:

- `symprec`: symmetry tolerance used during structure standardization. See
  pymatgen/spglib documentation for detailed behavior.
- `angle_tolerance`: angular tolerance, in degrees, used during symmetry
  standardization. See pymatgen documentation for details.
- `symprec_scan`: optional adaptive symmetry-tolerance scan. It is disabled by
  default (`False`); when enabled, the default `symprec_scan_mode` is `"loose"`.
- `site_tolerance`: distance tolerance for deciding whether two sites should be
  treated as the same disorder site.
- `occ_tolerance`: SWORD's post-parsing occupancy tolerance for each grouped
  orbit/site. Occupancy sums above this value are reported as occupancy-orbit
  errors.
- `vac_tolerance`: minimum missing occupancy needed to record a `VAC` component
  in the disorder label. For example, an occupancy sum of 0.995 does not produce
  `VAC` when `vac_tolerance=1e-2`.
- `frac_tolerance`: rounding tolerance used in fractional-coordinate and
  disorder-label operations. Values like `1e-4` or `1e-5` are recommended.
- `conventional_struct`: whether to use pymatgen's conventionalized structure
  during the SWORD parsing path.
- `refine_struct`: whether to let pymatgen refine the structure before
  labelling. This can change labels and should be used deliberately.

Two occupancy-related parameters are intentionally separate.
`parser_occ_tolerance`: occupancy tolerance passed to pymatgen's CIF parser. Acts 
before SWORD, while the CIF is being read.
`occ_tolerance`: SWORD's post-parsing occupancy tolerance for each grouped
  orbit/site. Occupancy sums above this value are reported as occupancy-orbit
  errors.
For ICSD curation, a common choice is
`parser_occ_tolerance=1.05` and `occ_tolerance=1.0`.

## ICSD Curation Pipeline

`run_icsd_dedup_pipeline` provides a high-level workflow for ICSD
dataframes containing a CIF text column and a stable entry ID column.

```python
import pandas as pd
from sword import run_icsd_dedup_pipeline

df = pd.read_pickle("ICSD2025_summary.pkl")

result = run_icsd_dedup_pipeline(
    df,
    cif_col="cif",
    id_col="CollectionCode",
    mode="from_collection_code",
    prescreen_params={
        "parser_occ_tolerance": 1.05,
        "excluded_elements": ("He", "Ne", "Ar", "Kr", "Es"),
        "exclude_hydrogen": True,
    },
    sword_params={
        "parser_occ_tolerance": 1.05,
        "occ_tolerance": 1.0,
        "site_tolerance": 1e-4,
        "vac_tolerance": 1e-2,
        "frac_tolerance": 1e-4,
    },
    family_info=False,
    disorder_refinement_policy="dom_vector",
    composition_tol=0.05,
    dom_tol=0.03,
    output_dir="sword_icsd_results",
)
```

The result object contains:

- `result.prescreen_rejected`: entries removed before labelling, with reject reasons.
- `result.label_results`: the labelled dataframe with SWORD labels and compact diagnostics.
- `result.anomalies`: detailed warning/error tables.
- `result.label_groups`: grouped SWORD-label summary table.
- `result.refined`: curated and deduplicated entries.

Disordered entries use the `dom_vector` refinement policy by default. Its
default tolerances are `composition_tol=0.05` and `dom_tol=0.03`; smaller values
make refinement stricter. The representative-site occupancy vector used by the
policy is available in `result.label_results["dom_site_occ_vector"]`.

`result.prescreen_rejected["reject_reason"]` records why a row did not enter
the labelling stage. Possible reasons include:

- `parse_error`: the CIF block could not be read by pymatgen.
- `hydrogen`: the CIF site labels contain H, D, or T when `exclude_hydrogen=True`.
- `non_element`: the CIF site labels contain unsupported/non-element symbols or
  elements listed in `excluded_elements`.
- `coordinate_error`: fractional coordinates are missing or malformed.
- `wyckoff_error`: Wyckoff symbols are missing or malformed.
- `multiplicity_error`: site multiplicities are missing or malformed.
- `occupancy_error`: a raw site occupancy is missing, non-positive, or larger
  than `parser_occ_tolerance`.

`result.label_results["processing_issue"]` records semicolon-separated warning
or error flags generated during labelling. `None` means no issue was recorded.
The detailed rows are stored in `result.anomalies` and, when `output_dir` is
provided, saved under `anomalies/`:

- `label_errors`: hard failures during structure parsing or SWORD label
  generation. The compact flag: `structure_parse_failed` or
  `label_generation_failed`.
- `wyck_float_warn`: Wyckoff coordinate expansion gives more positions than the
  declared multiplicity, usually indicating floating-point or refinement issues
  in the CIF. The compact flag: `wyckoff_float_warning`.
- `occ_err_sites`: grouped orbit/site occupancy exceeds `occ_tolerance`. The
  compact flag: `occupancy_orbit_error`.
- `equivalent_but_distinct_sites`: sites are symmetry-equivalent but have
  distinct representative coordinates, often because the CIF coordinates are not
  exactly on the expected special position. The compact flag:
  `equivalent_sites_warning`.
- `same_valence_sites`: same-element/same-valence co-occupancy was detected and
  merged for labelling. The compact flag is `same_valence_site_warning`.
- `intersect_orb_errors`: the positional-disorder check failed. The compact flag
  is `positional_check_failed`. Positional disorder itself is recorded in
  `is_positional_disorder`.

By default, the refined table removes hard parse/label errors, positional
disorder, occupancy-orbit errors, Wyckoff floating-point warnings, and
equivalent-site warnings. Same-valence warnings are recorded but not removed by
default.

Set `family_info=True` to append a `SWORD_family_dic` column to
`result.label_results`. This is slower and usually not needed for a first ICSD
curation pass.

```python
result = run_icsd_dedup_pipeline(
    df,
    family_info=True,
    family_params={
        "parser_occ_tolerance": 10.0,
        "fill_vacancy": False,
    },
)
```

`family_params` must be a dictionary accepted by `SWORDFamilyMatcher`.
Use `fill_vacancy=False` for most large reference-table precomputations such as
ICSD/COD. Set `fill_vacancy=True` mainly for query-side ordered structure datasets/databases
that may contain vacancy ordering or missing-site defects, e.g. Materials Project, LeMat-Bulk, where filling
possible vacancies can help expose candidate disorder parents.

## Generic Database Labelling

For non-ICSD databases, use `label_dataframe`. The dataframe only needs a CIF
text column and a stable ID column.

```python
from sword import label_dataframe

labelled, anomalies = label_dataframe(
    df,
    cif_col="cif",
    id_col="material_id",
    sword_params={
        "parser_occ_tolerance": 1.05,
        "occ_tolerance": 1.0,
        "site_tolerance": 1e-4,
        "vac_tolerance": 1e-2,
        "frac_tolerance": 1e-4,
    },
    family_info=False,
)

print(labelled[["material_id", "SWORD_label", "is_disorder"]])
```

This interface is appropriate for MP, LeMat, or user-built
structure tables when each row can provide a CIF string.

Set `family_info=True` if you also want family dictionaries:

```python
labelled, anomalies = label_dataframe(
    df,
    cif_col="cif",
    id_col="material_id",
    family_info=True,
    family_params={
        "symprec_child": 1e-2,
        "symprec_search": 1.0,
        "parser_occ_tolerance": 10.0,
        "fill_vacancy": False,   #turn on to True if this dataset may contain vacancy ordering and you want to find disordered parent structure of this vacancy-type ordered structure
    },
)

print(labelled[["material_id", "SWORD_label", "SWORD_family_dic"]])
```

## Order-Disorder Family Matching

`SWORDFamilyMatcher` can generate a SWORD family dictionary for an ordered query
structure and search for possible disordered parent labels.

```python
from sword import SWORDFamilyMatcher

matcher = SWORDFamilyMatcher(
    symprec_child=1e-2,
    symprec_search=1.0,
    parser_occ_tolerance=10.0,
    fill_vacancy=False,
)

family = matcher.get_sword_dic("path/to/ordered_query.cif")

print(family["child_label"])
print(family["parent_labels"])
```

For repeated searches, precompute a family dictionary column for the reference
table. This can be slow for large datasets, so it is usually done once and saved.
For large reference tables, keep `fill_vacancy=False` unless you explicitly want
the reference-side vacancy-filling expansion.

```python
from sword import SWORDFamilyMatcher

matcher = SWORDFamilyMatcher(fill_vacancy=False)

reference_df["SWORD_family_dic"] = reference_df["cif"].apply(
    matcher.get_sword_dic
)
```

Then compare a query against the reference table:

```python
matches = matcher.fit_many(
    query_structure,
    reference_df,
    disorder_refinement_policy="dom_vector",
    composition_tol=0.30,
    dom_tol=0.30,
)

print(matches["matched_disordered_ids"])
print(matches["matched_disordered_labels"])
```

By default, `fit_many` reads `reference_df["SWORD_family_dic"]`. If the query is
a dataframe row with `SWORD_family_dic` and `SWORD_family_dic_vac`, both query
columns are used and `matches["matched_source_by_id"]` records which query
column produced each match.

For comparable direct disorder matches, `fit_many` also returns
`dom_distance_by_id` and `dom_match_by_id`, together with `matching_mode_by_id`
and `sg_relation_by_id` for concise match diagnostics. `fit_many` uses the
`dom_vector` policy by default with `composition_tol=0.30` and `dom_tol=0.30`.
Use smaller tolerances when a stricter parent-composition match is required.

`fill_vacancy=True` can be useful when ordered structures may represent vacancy
ordering variants. In practice, this is most useful for query structures that
are ordered but may be vacancy-ordered or vacancy-deficient variants. For large
ICSD-style reference precomputation, `fill_vacancy=False` is usually a better
default because it is faster and avoids generating extra vacancy-filled
candidates for every entry.

## Notes

SWORD labels are intended for symmetry- and Wyckoff-aware grouping of materials
structures. They are especially useful for comparing ordered and disordered
entries under a common structural representation, but they do not replace
manual crystallographic judgment for ambiguous or low-quality CIF records.

## Citation

If you use SWORDlib, please cite our papers:

```bibtex
@article{huang2026sword,
  title   = {SWORD: Symmetry and Wyckoff-sequence of Ordered and Disordered crystals},
  author  = {Huang, Yuyao and Nong, Wei and Yamazaki, Shuya and Petersen, Martin Hoffmann and Wang, Jianghai and Zhu, Ruiming and Hippalgaonkar, Kedar},
  journal = {arXiv preprint arXiv:2604.17994},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.17994},
  doi     = {10.48550/arXiv.2604.17994}
}
```

For family-matching functions, please cite:

```bibtex
@article{yamazaki2026orderdisorder,
  title   = {Navigating Order-(Dis)Order Family Trees via Group-Subgroup Transitions},
  author  = {Yamazaki, Shuya and Huang, Yuyao and Petersen, Martin Hoffmann and Nong, Wei and Hippalgaonkar, Kedar},
  journal = {arXiv preprint arXiv:2604.21386},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.21386},
  doi     = {10.48550/arXiv.2604.21386}
}
```
