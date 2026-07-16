import math
import re
import warnings
from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from importlib.resources import files
from pymatgen.core.structure import Structure

_DATA_DIR = files("sword") / "data"
_RADII_CSV = _DATA_DIR / "all_radii.csv"

radius_df = pd.read_csv(_RADII_CSV)


def periodic_dist(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    diff_pbc = diff - np.round(diff)
    return np.linalg.norm(diff_pbc)

def get_equiv_positions(point, sym_ops, frac_tolerance=1e-4):
    """Computes the set of unique equivalent positions for a given fractional coordinate 
    using a list of symmetry operations {sym_ops} from CIF. The generated raw positions 
    will be clustered when periodic Euclidean distance tolerance <= frac_tolerance and 
    floating-error derived duplicated positions will be deduplicated.
    returning the coordinates as a sorted tuple."""
    p0 = np.asarray(point, dtype=float) % 1.0
    orb_raw = []
    for op in sym_ops:
        if hasattr(op, "operate"):
            p = op.operate(p0)
        else:
            p = op(p0)
        orb_raw.append(p % 1.0) 
    # display(orb_raw)

    clusters = [] 
    for p_new in orb_raw:
        found = False
        for rep, points_in_cluster in clusters:
            #print(f"p_new{p_new}, rep: {rep}, distance:{periodic_dist(p_new, rep)}")
            if periodic_dist(p_new, rep) <= frac_tolerance*2:
                points_in_cluster.append(p_new)
                found = True
                break

        if not found:
            clusters.append([p_new.copy(), [p_new.copy()]])

    # for i, cluster in enumerate(clusters):
    #     rep = cluster[0]
    #     print(f"Cluster {i+1}: Representative: {np.round(rep, 6)}")
    # print(f"total clusters: {len(clusters)}")

    final_coords = []
    ndigits = max(0, -int(math.floor(math.log10(frac_tolerance))))
    for rep, _ in clusters: 
        coord_tuple = tuple(
            float(x)
            for x in np.round(
                (np.round((rep % 1.0) / frac_tolerance) * frac_tolerance) % 1.0,
                ndigits,))

        if coord_tuple not in final_coords:
            final_coords.append(coord_tuple)
                       
    return tuple(sorted(final_coords))


def _make_occ_vector(merged_occ):
    """Return the normalized, element-keyed occupancy-polarization vector."""
    by_element = defaultdict(float)
    for token, occ in (merged_occ or {}).items():
        token = str(token)
        if token.upper() == "VAC":
            element = "VAC"
        else:
            match = re.match(r"[A-Z][a-z]?", token)
            element = match.group(0) if match else token
        by_element[element] += float(occ)

    by_element = {
        element: occ
        for element, occ in by_element.items()
        if occ > 0
    }
    total = sum(by_element.values())
    n_components = len(by_element)
    if total <= 0 or n_components < 2:
        return None

    uniform_occ = 1.0 / n_components
    scale = math.sqrt(n_components / (n_components - 1))
    return {
        element: round(scale * (occ / total - uniform_occ), 12)
        for element, occ in sorted(by_element.items())
    }

def compute_disorder(entry, disordered_list, total_sites = 0, verbose = False, return_details: bool = False):
    """
    For disorder structures, quantifies the degree of mixing(DOM) in a structure. The function utilizes 
    Shannon Entropy to calculate the mixing factor (site_mixing) for each disordered site. 

    The Degree of Mixing has sign delta: delta = (XA - XZ) / |XA - XZ|
    To establish a directional DOM, the site with the highest contribution to total mixing value is chosen as the representative. 
    A sign factor (delta) is then derived from the two most extreme components (minimum XA and maximum XZ occupancies) of this representative site.

    Returns the fraction of disordered sites and the overall degree of mixing (DOM) for the entire structure.
    """
    from collections import defaultdict
    all_labels = entry.labels
    label2mult = dict(zip(all_labels, entry.mults))
    label2type = dict(zip(all_labels, entry.type_symbols))
    label2occ = dict(zip(all_labels, entry.occ))

    disordered_sites = 0
    total_mixing = 0.0
    rep_site_mixing = float("-inf")
    rep_merge_occ = None
    rep_site_labels = None

    for sites in disordered_list:
        non_vac_labels = [lab for lab in sites if "VAC" not in lab]
        non_vac_type_symbols = [label2type[lab] for lab in non_vac_labels if label2occ[lab]>0]
        occs = [label2occ[lab] for lab in non_vac_labels if label2occ[lab]>0]

        merged_occ = defaultdict(float) #merge the occ, because some situation will include same valence element in disorder_list
        for s, o in zip(non_vac_type_symbols, occs):
            merged_occ[s] += o

        occs = list(merged_occ.values()) #should check if merged_occ > 1.0,可能需要增加一个空值occ溢出的参数

        if any("VAC" in lab for lab in sites):
            occ_vac = 1.0 - sum(occs)
            occs.append(occ_vac)
            merged_occ["VAC"] = occ_vac
        if len(occs) > 1:
            site_mixing = -sum(p * math.log(p) for p in occs)
            site_mixing = site_mixing / math.log(len(occs))
        else:
            site_mixing = 0.0

        multiplicity = label2mult[non_vac_labels[0]]

        if site_mixing * multiplicity > rep_site_mixing * multiplicity:
            rep_site_mixing = site_mixing
            rep_merge_occ = dict(merged_occ)
            rep_site_labels = list(non_vac_labels)

        disordered_sites += multiplicity
        total_mixing += site_mixing * multiplicity
        if verbose == True:
            print(f"the site is occupied with: {dict(merged_occ)}\n"
                    f"site mixing factor: {site_mixing}\n"
                    f"number of sites: {multiplicity}")

    dom_info = None
    if len(rep_merge_occ) > 1:
        kmin = min(rep_merge_occ, key=rep_merge_occ.get)
        kmax = max(rep_merge_occ, key=rep_merge_occ.get)
        a, b = sorted((kmin, kmax))
        XA = rep_merge_occ[a]
        XZ = rep_merge_occ[b]
        if XA == XZ:
            delta = 1
        else:
            delta = (XA - XZ) / abs(XA - XZ)
        dom_info = {
            "representative_site_labels": rep_site_labels,
            "representative_site_merged_occ": rep_merge_occ,
            "representative_site_occ_vector": _make_occ_vector(rep_merge_occ),
            "delta_pair_used": [a, b],
            "delta_occ_used": [XA, XZ],
            "delta_sign": delta,
        }
        if verbose == True:
            print(kmin, kmax, XA, XZ, delta)
    else:
        delta = 0
        if verbose == True:
            print(delta)

    #ordered_sites = sum(label2mult[lab] for lab in all_labels_list if lab not in all_disordered_labels)
    #total_sites = disordered_sites + ordered_sites
    if verbose == True:
            print(f"disordered_sites: {disordered_sites}, total_sites: {total_sites}, total_mixing: {total_mixing}")
    fraction_of_disordered_sites = round(disordered_sites / total_sites, 4)
    degree_of_mixing = round(delta* total_mixing / disordered_sites, 4)

    if return_details:
        return fraction_of_disordered_sites, degree_of_mixing, dom_info
    return fraction_of_disordered_sites, degree_of_mixing

def mean_site_radius(species_list):
    """
    Compute mean radius for a crystallographic site occupied by species_list.
    species_list: list of species entries. Each entry may be:
       - string: e.g. "Fe2+", "O2-", "C", "Fe3"
    radius_df: pandas.DataFrame with columns ['symbol','charge','ionic radius','empirical']
    Returns: float (mean radius)
    Raises: ValueError if any element has no data in radius_df.
    """
    radii = []
    if not species_list:
        raise ValueError("species_list is empty")

    for entry in species_list:
        symbol = None
        ox = None

        if isinstance(entry, (list, tuple)) and len(entry) >= 1:
            symbol = str(entry[0])
            ox = entry[1] if len(entry) > 1 else None

        elif isinstance(entry, str):
            m = re.match(r"^([A-Z][a-z]?)(.*)$", entry.strip())
            if not m:
                raise ValueError(f"Cannot parse species string: {entry!r}")
            symbol = m.group(1)
            rest = m.group(2)
            mnum = re.search(r"([+-]?\d+(\.\d+)?)", rest)
            ox = float(mnum.group(1)) if mnum else None

        else:
            if hasattr(entry, "element"):
                element_obj = entry.element
                symbol = getattr(element_obj, "symbol", str(element_obj))
                ox = getattr(entry, "oxi_state", None)
                if ox is None:
                    ox = getattr(entry, "oxidation_state", None)
                if ox is None:
                    ox = getattr(entry, "oxid_state", None)
            else:
                s = str(entry)
                m = re.match(r"^([A-Z][a-z]?)(.*)$", s.strip())
                if m:
                    symbol = m.group(1)
                    rest = m.group(2)
                    mnum = re.search(r"([+-]?\d+(\.\d+)?)", rest)
                    ox = float(mnum.group(1)) if mnum else None
                else:
                    raise ValueError(f"Unsupported species entry type: {entry!r}")
        # ---- round oxidation if present ----
        ox_int = None
        if ox is not None:
            try:
                ox_int = int(round(float(ox)))
            except Exception:
                ox_int = None
        # ---- lookup: ionic radius first, then empirical ----
        radius_val = None
        if ox_int is not None:
            rows = radius_df[(radius_df['symbol'] == symbol) & (radius_df['charge'] == ox_int)]
            if not rows.empty:
                radius_val = float(rows['ionic radius'].min())

        if radius_val is None:
            rows_emp = radius_df[radius_df['symbol'] == symbol]
            if not rows_emp.empty and 'empirical' in rows_emp.columns:
                radius_val = float(rows_emp['empirical'].min())

        if radius_val is None:
            raise ValueError(f"No radius data for element '{symbol}' (entry {entry!r}) in radius_df")

        radii.append(radius_val)

    return float(np.mean(radii))

def intersect_orb(entry, site_tolerance = 1e-4, occ_tolerance = 1.0):
    """
    Checks for positional disorder in a crystallographic entry by examining atom-to-atom 
    distances (not site-to-site wyckoff position distance) against a chemically-derived radius threshold
    derived from function mean_site_radius. The site species intersect_orb, 
    and flags any sites found to be closer than this threshold (or 1.0 Å, whichever is greater).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            crystal = Structure.from_str(entry.cif_str, fmt="cif",
                                        site_tolerance = site_tolerance,
                                        occupancy_tolerance = occ_tolerance
                                        )
        all_site_labels = [s.label for s in crystal]
        all_site_species = [list(s.species.keys()) for s in crystal]

        site_keys = [tuple(sorted((
                    sp.symbol, int(round(sp.oxi_state)) if getattr(sp, "oxi_state", None) is not None else None) for sp in lst))
                    for lst in all_site_species]

        key_to_sp = {}
        for k, lst in zip(site_keys, all_site_species):
            key_to_sp.setdefault(k, lst)
        positional_disorder = False
        rads = {k: mean_site_radius(key_to_sp[k]) for k in key_to_sp}
        pair_thresh = { (a,b): 0.5*(rads[a]+rads[b]) for a,b in combinations_with_replacement(rads, 2) }
        r_thresh_max = max(1.0, max(pair_thresh.values(), default=0.0))
        neighbors = crystal.get_all_neighbors(r_thresh_max)
        intersect_orbits = []
        for i, nbrs in enumerate(neighbors):
            for nbr in nbrs:
                j = nbr.index
                if i >= j: 
                    continue
                d = getattr(nbr, "nn_distance", None) or nbr.distance
                ki, kj = site_keys[i], site_keys[j]
                pk = (ki, kj) if (ki, kj) in pair_thresh else (kj, ki)
                val = pair_thresh.get(pk, None)
                if val is None:
                    raise ValueError(
                        f"No threshold found for species pair {pk} (site indices {i},{j}) in {entry.CollectionCode}. "
                    )
                thresh = max(1.0, val)
                if d < thresh:
                    intersect_orbits.append((entry.CollectionCode, all_site_labels[i], tuple(sp.name for sp in key_to_sp[ki]), all_site_labels[j], tuple(sp.name for sp in key_to_sp[kj]), round(d,5), thresh))
        positional_disorder = len(intersect_orbits) > 0
        intersect_orbits = list(set(intersect_orbits)) 

        intersect_dicts = []  
        for t in intersect_orbits:
            intersect_dicts.append({
                "CollectionCode": t[0],
                "intersect_site1": t[1], 
                "intersect_site1_elements": t[2], 
                "intersect_site2": t[3], 
                "intersect_site2_elements": t[4], 
                "distance": t[5], 
                "threshold": t[6],
            })
        
        return positional_disorder, intersect_dicts

    except Exception as e:
        raise RuntimeError(f"CollectionCode {entry.CollectionCode}: intersect_orb failed: {e}") from e

def orb_keys_max_diff(key1, key2, prune_tol=1e-2, frac_tolerance = 1e-4):
    ndigits = max(0, -int(math.floor(math.log10(frac_tolerance))))
    if len(key1) != len(key2):
        return math.inf
    max_diff = 0.0
    for p, q in zip(key1, key2):
        d0 = abs(p[0] - q[0]); d0 = min(d0, 1.0 - d0)
        d1 = abs(p[1] - q[1]); d1 = min(d1, 1.0 - d1)
        d2 = abs(p[2] - q[2]); d2 = min(d2, 1.0 - d2)
        pt_max = max(d0, d1, d2)
        if pt_max > max_diff:
            max_diff = pt_max
        if max_diff > prune_tol:
            return np.round(max_diff, ndigits)
    return np.round(max_diff,ndigits)
