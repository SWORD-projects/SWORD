
"""SWORDlib is a Python library based on SWORD - Symmetry and Wyckoff-sequence of Ordered and Disordered crystals."""

__version__ = "0.1.1"
__email__ = "YUYAO003@e.ntu.edu.sg"

from .disorder import (
    compute_disorder,
    get_equiv_positions,
    intersect_orb,
)
from .family import SWORDFamilyMatcher
from .icsd import (
    ICSDDedupResult,
    label_dataframe,
    label_icsd_dataframe,
    prescreen_icsd_dataframe,
    run_icsd_dedup_pipeline,
)
from .label import (
    disorder_label,
    get_sword_info,
    get_sword_info_for_ICSD,
    get_sword_label,
    get_sword_label_for_ICSD,
    get_sword_label_from_pyxtal,
)
from .structure import StructureEntry, symm_orbits_df
from .utils import (
    dedupe_by_dom_projection,
    dedupe_by_vector_dom,
    dedupe_by_same_dom,
    filter_by_elements_and_sg,
    find_by_disorder_label,
    group_ICSD,
)
from .vacancy import find_vacancy_ordered
from .wyckoff import get_canonical_wyckoff_sets

__all__ = [
    "__version__",
    "StructureEntry",
    "ICSDDedupResult",
    "SWORDFamilyMatcher",
    "compute_disorder",
    "dedupe_by_dom_projection",
    "dedupe_by_vector_dom",
    "dedupe_by_same_dom",
    "disorder_label",
    "filter_by_elements_and_sg",
    "find_by_disorder_label",
    "find_vacancy_ordered",
    "get_canonical_wyckoff_sets",
    "get_equiv_positions",
    "get_sword_info",
    "get_sword_info_for_ICSD",
    "get_sword_label",
    "get_sword_label_for_ICSD",
    "get_sword_label_from_pyxtal",
    "group_ICSD",
    "intersect_orb",
    "label_dataframe",
    "label_icsd_dataframe",
    "prescreen_icsd_dataframe",
    "run_icsd_dedup_pipeline",
    "symm_orbits_df",
]
