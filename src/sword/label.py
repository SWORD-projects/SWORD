import math
import os
import warnings
from collections import Counter, defaultdict
from typing import Union

from pymatgen.core.structure import Structure

from .disorder import (
    compute_disorder,
    get_equiv_positions,
    intersect_orb,
    orb_keys_max_diff,
)
from .structure import StructureEntry
from .wyckoff import get_canonical_wyckoff_sets, mapping_by_sg


def disorder_label(entry, *, site_tolerance: float = 0.0001, vac_tolerance: float = 1e-2, occ_tolerance: float = 1.0, frac_tolerance: float = 0.0001, include_positional_disorder: bool = True, verbose = False):
    """
    Accepts an ICSDEntry instance.
    Determining disorder types (vacancy, positional);
    Return list of disordered;
    Computes the degree of mixing (DOM) and fractional of disordered sites against total sites; 
    Converts the Wyckoff set into a canonical standard format; 
    Assembles a disorder_label as {wyckoff_set_std}_{space_group_number}_{element_seq}
    (e.g., 'j_i2_h_a_225_A_2B_{A+B}_C') for unique disorder structural identification.
    """
    orb_none = []
    wyck_float_warn = []
    occ_err = []
    is_occ_err = False
    equivalent_but_distinct = []
    merged_same_valence = []

    space_group_number = entry.spg_num
    code = entry.CollectionCode
    all_labels = entry.labels
    all_points = entry.points
    all_type_symbols = entry.type_symbols
    all_occ = entry.occ
    all_mults = entry.mults
    wyckoffs = entry.wyckoffs
    sym_ops = entry.sym_ops

    groups = defaultdict(list)
    disordered_type_list = []
    disordered_label_list = []
    sequence_map = {}
    total_sites = 0

    #====================== group index by Wyckoff letter ======================
    for idx, key in enumerate(wyckoffs):
        groups[key].append(idx)
    #====================== process each letter group ============================
    is_disorder = False
    is_sub_disorder = False
    is_vac_disorder = False
    for letter, idxs in groups.items():
        if verbose:
            print(f">>> Processing letter group '{letter}':")        
        points = [all_points[i] for i in idxs]  # for each letter group and their index, retrieve all the wyckoff positions belonging to this letter group
        occs = [all_occ[i] for i in idxs]
        equiv_orbit_groups = {}
        for local_idx, point in enumerate(points): #local_idx means the wyckoff orbits that share same wyckoff letter are given a local index
            mult_point = all_mults[idxs[local_idx]]
            label_point = all_labels[idxs[local_idx]]
            if occs[local_idx] <= 0:
                if verbose:
                    print(f"    → skipping orbit with local_index {local_idx}: {label_point} has zero/negative occupancy")
                continue            
            try:
                orb_key = get_equiv_positions(point, sym_ops, frac_tolerance = frac_tolerance) 
                if len(orb_key) > mult_point:
                    orb_key = get_equiv_positions(point, sym_ops, frac_tolerance*10)
                    wyck_float_warn.append({
                        "CollectionCode": code,
                        "label": label_point,
                        "wyckoff_position": point,
                        "equivalent_positions": len(orb_key),
                        "multiplicity": mult_point
                        })
                    if len(orb_key) > mult_point:
                        if verbose is True:
                            warnings.warn(
                                    f"Incorrect stoichiometry: \nCollectionCode {code}: labels= {label_point}, coordinate= {point}\n"
                                    f"equivalent positions {len(orb_key)} do not match with multiplicity {mult_point} dut to float error\n",
                                    stacklevel=2)

            except Exception:
                orb_none.append({
                    "CollectionCode": code,
                    "wyckoff_position": point
                    }) 
                continue
            for existing_key in list(equiv_orbit_groups.keys()):
                if orb_keys_max_diff(existing_key, orb_key, prune_tol=site_tolerance, frac_tolerance=frac_tolerance) <= site_tolerance:
                    equiv_orbit_groups[existing_key].append(local_idx)  # if max difference between two orb_key < site_tolerance, there are wyckoff orbits occupy same site, will be added into equiv_orbit_group
                    break
            else:
                equiv_orbit_groups[orb_key] = [local_idx]
        # Skip wyckoff letters that only contained zero/negative occupancy entries
        if not equiv_orbit_groups:
            if verbose:
                print("    → all orbits in this letter group have non-positive occupancy, skipping")
            continue
    #==================section 2: identifying type of site-disorder, calculating number of total/disordered sites, and degree of mixing for each disordered position===============            
        elem_counter = Counter()
        for local_idxs in equiv_orbit_groups.values():            
            labels = [all_labels[idxs[i]] for i in local_idxs]
            if verbose:
                print(f"    → checking orbit with local_index {local_idxs}: {labels}")
            elems = [''.join(filter(str.isalpha, lab)) for lab in labels]
            vals = [all_type_symbols[idxs[i]] for i in local_idxs]
            occs = [all_occ[idxs[i]] for i in local_idxs]
            mults = [all_mults[idxs[i]] for i in local_idxs]
            ndigits = max(0, -int(math.log10(frac_tolerance))) + 1
            coords = [points[i] for i in local_idxs]
            rounded_coords = [tuple(round(v, ndigits) for v in coord) for coord in coords]
            assert all(m == mults[0] for m in mults), "Inconsistent multiplicities in same orbit"
            multiplicity = mults[0]     #<<<-----------------------
            total_sites += multiplicity  #<<<-----------------------
            #---------detecting vacancy disorder-------------------
            occ_sum = sum(occs)  
            if occ_sum > occ_tolerance:
                is_occ_err = True
                occ_err.append({
                    "CollectionCode": code,
                    "label": sorted(labels),
                    "wyckoff_positions": rounded_coords,
                    "occupancies": occs,
                    "occ_sum": occ_sum
                    })
            vacancy_amount = 1.0 - occ_sum
            vacancy_amount = max(0.0, vacancy_amount) #all occ_sum > occ_tolerance will be rescaled down to 1.0 when calculating vacancy_amount
            if verbose:
                print(f"           vacancy_component = {vacancy_amount}")
                print(f"           total occupancy = {occ_sum}")
            has_vacancy = (vacancy_amount >= vac_tolerance)

            if len(local_idxs) > 1 or has_vacancy:
                if has_vacancy:
                    labels.append('VAC')
                    elems.append('VAC')
                    vals.append('VAC')
                    occs.append(1.0 - occ_sum)
                    is_vac_disorder = True
                    is_disorder = True
            #------------------------------------------------
                if verbose:
                    print(f"     !!! mixed elements, marking disorder for global idxs {[idxs[i] for i in local_idxs]}")

                #rounded_coords = [tuple(round(v / frac_tolerance) * frac_tolerance for v in coord) for coord in coords]
                if len(set(rounded_coords)) > 1:  #record warning 1: equivalent wyckoff orbit with different representative coord
                    equivalent_but_distinct.append({
                        "CollectionCode": code,
                        "label": sorted(labels),
                        "wyckoff_positions": rounded_coords,
                        "occupancies": occs,
                        "occ_sum": occ_sum
                        })
                    # warnings.warn(
                    #         f"WARNING: CollectionCode {CollectionCode} shows equivalent wyckoff orbit on one site but with different representative coordinate"
                    #         f"labels={labels}, coordinate={rounded_coords}",
                    #         stacklevel=2
                    #         )

                occ_dict = defaultdict(float)
                for el, occ in zip(vals, occs):
                    occ_dict[el] += occ
            #---------detecting substirutional disorder-------------------
                actual_species = [key for key in occ_dict if key != 'VAC']
                if len(actual_species) > 1:
                    is_sub_disorder = True
            #---------record outlier same valence species occupy site------ 
                has_dup_type_elems = len(vals) != len(set(vals)) 
                if has_dup_type_elems:   ##record warning 2: same elem with same valence on site-disorder
                    merged_same_valence.append({
                        "CollectionCode": code,
                        "labels": sorted(labels),
                        "valences": sorted(vals),
                        "wyckoff_positions": rounded_coords
                    })
                    # warnings.warn(
                    #         f"WARNING: CollectionCode {CollectionCode} has same elements with same valence state appears multiple times on one orbit"
                    #         f"labels={labels}, vals={vals}",
                    #         stacklevel=2
                    #         )

                if len(occ_dict) > 1:  # only different type_symbols will be accounted, this exclude same-element disorder,
                    is_disorder = True
                    if verbose:
                        print(f"           occ_dict: {occ_dict}")
                    disordered_label_list.append(sorted(labels))
                     #valence disorder will be implicit merge into one element
                    if len(set(elems)) == len(elems):
                        disordered_type_list.append(sorted(labels))
                    else:
                        disordered_type_list.append(sorted(vals))

    #====================section 3: get wyckoff sequence and elemental sequence =========================
            cnts = Counter(set(elems))  # dictionary counter {"O":3, "Li":1}, element as key, number as value, note, here we use set() to normalize {Fe2+: 1, Fe3+: 1} valence disorder into counter{Fe: 1}
            if len(cnts) == 1: # no site-disorder, single‑element orbit -> "3O" or "Li"
                elem, c = cnts.most_common(1)[0]
                orbit_str = f"{c}{elem}" if c > 1 else elem
            else: # multi‑element orbit (site‑disorder) -> sorted then "Li+Mn"
                parts = []
                for el in sorted(cnts): 
                    parts.append(f"{cnts[el]}{el}" if cnts[el] > 1 else el)
                orbit_str = "+".join(parts)
            elem_counter[orbit_str] += 1
        
        elem_parts = [] #elem_part is the elemental correspondings of each wyckoff letter group
        for orbit_str, cnt in elem_counter.items():
            if cnt == 1:
                elem_parts.append(orbit_str)
            else:
                if "+" in orbit_str:
                    elem_parts.append(f"{cnt}({orbit_str})")
                else:
                    elem_parts.append(f"{cnt}{orbit_str}")
        elem_parts = sorted(elem_parts)
        if verbose:
            print(f"      elem_parts: {elem_parts}")

        # constructing wyckoff_sequence
        n_orbits = len(equiv_orbit_groups)
        wyckoff_part = letter + (str(n_orbits) if n_orbits > 1 else "")
        # single orbit with single element has no {}, otherwise with {}
        single_orbit_single_elem = (
            n_orbits == 1
            and len(elem_parts) == 1
            and "+" not in elem_parts[0]
            and not elem_parts[0][0].isdigit()
        )
        if single_orbit_single_elem:
            elem_part = elem_parts[0]
        else:
            elem_part = "{" + ",".join(elem_parts) + "}"

        sequence_map[wyckoff_part] = elem_part 
    #sequence_map = dict(sorted(sequence_map.items(), key=lambda kv: kv[0]))
    #sequence_map = dict(sorted(sequence_map.items(), key=lambda kv: (kv[1], kv[0])))
    #wyckoff_seq = "_".join(sequence_map.keys())
    #element_seq = "_".join(sequence_map.values())
    #=========================section 3.2: calculating fraction_of_disordered_sites, and degree_of_mixing==================================
    if disordered_label_list:
        fraction_of_disordered_sites, degree_of_mixing, dom_info = compute_disorder(
            entry, disordered_label_list, total_sites, return_details=True
        )
    else:
        fraction_of_disordered_sites = 0
        degree_of_mixing = None
        dom_info = None
    # DOM is only comparable when applying to similar structures (e.g. same prototype, same disordered positions, with different mixing ratio!!!)
    #=========================section 4: get canonical wyckoff sequence=======================  
    #wyckoff_set_std = get_canonical_wyckoff_sets(sequence_map, mapping_by_sg, entry.spg_num)
    #disorder_label = f"{wyckoff_set_std}_{space_group_number}_{element_seq}"      
    wyckoff_set_std, element_seq, sequence_map_std = get_canonical_wyckoff_sets(
    sequence_map, mapping_by_sg, entry.spg_num
    )
    disorder_label = f"{wyckoff_set_std}_{space_group_number}_{element_seq}"

    is_positional_disorder = None
    intersect_orbs = []
    intersect_orb_error= None
    if include_positional_disorder and not is_occ_err:
        try:
            is_positional_disorder, intersect_orbs = intersect_orb(entry, site_tolerance= site_tolerance, occ_tolerance= occ_tolerance)
        except Exception as e:
            is_positional_disorder = None
            intersect_orb_error = str(e)

    return {
        #disordered type:
        "is_disorder": is_disorder,
        "is_vac_disorder": is_vac_disorder,
        "is_sub_disorder": is_sub_disorder,
        "is_positional_disorder": is_positional_disorder,

        #disordered sites list:
        "disordered_valence_list": disordered_type_list,
        "disordered_label_list": disordered_label_list,

        #computed disordered info:
        "fraction_of_disordered_sites": fraction_of_disordered_sites,
        "degree_of_mixing": degree_of_mixing,
        "dom_info": dom_info,

        #label info:
        "sequence_map": sequence_map_std,
        "wyckoff_set_std": wyckoff_set_std,
        "disorder_label": disorder_label,

        #outliers:
        "wyck_float_warn": wyck_float_warn,
        "occ_err_sites": occ_err,
        "equivalent_but_distinct_sites": equivalent_but_distinct,
        "same_valence_sites": merged_same_valence,
        "intersect_orbs": intersect_orbs,
        "orb_none_entry": orb_none,
        "intersect_orb_error": intersect_orb_error
    } 

def get_sword_label(
    data: Union[str, Structure],  # CIF text / CIF path / pymatgen Structure
    *,
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
    parser_occ_tolerance: float = 1.05,
    site_tolerance: float = 1e-4,
    occ_tolerance: float = 1.0,
    vac_tolerance: float = 1e-2,
    frac_tolerance: float = 1e-4,
    conventional_struct: bool = True,
    refine_struct: bool = False,
    recover_cif_spg: bool = False,
    symprec_scan: bool = False,
    symprec_scan_mode: str = "loose",
) -> str:
    if isinstance(data, Structure):
        cif_txt = data
    elif isinstance(data, str):
        if os.path.isfile(data):
            with open(data, "r", encoding="utf-8") as f:
                cif_txt = f.read()
        else:
            cif_txt = data
    else:
        raise TypeError("data must be str (CIF text/path) or pymatgen.core.Structure")

    entry = StructureEntry.from_txt(
        cif_txt,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        parser_occ_tolerance=parser_occ_tolerance,
        conventional_struct=conventional_struct,
        refine_struct=refine_struct,
        recover_cif_spg=recover_cif_spg,
        symprec_scan=symprec_scan,
        symprec_scan_mode=symprec_scan_mode,
    )

    return disorder_label(
        entry,
        site_tolerance=site_tolerance,
        occ_tolerance=occ_tolerance,
        vac_tolerance=vac_tolerance,
        frac_tolerance=frac_tolerance,
        include_positional_disorder=False,
        verbose=False,
    )["disorder_label"]

def get_sword_info(
    data: Union[str, Structure],  # CIF text / CIF path / pymatgen Structure
    *,
    symprec: float = 1e-2,
    angle_tolerance: float = 5.0,
    parser_occ_tolerance: float = 1.05,
    site_tolerance: float = 1e-4,
    occ_tolerance: float = 1.0,
    vac_tolerance: float = 1e-2,
    frac_tolerance: float = 1e-4,
    conventional_struct: bool = True,
    refine_struct: bool = False,
    recover_cif_spg: bool = False,
    symprec_scan: bool = False,
    symprec_scan_mode: str = "loose",
    include_positional_disorder: bool = True,
) -> str:
    if isinstance(data, Structure):
        cif_txt = data
    elif isinstance(data, str):
        if os.path.isfile(data):
            with open(data, "r", encoding="utf-8") as f:
                cif_txt = f.read()
        else:
            cif_txt = data
    else:
        raise TypeError("data must be str (CIF text/path) or pymatgen.core.Structure")

    entry = StructureEntry.from_txt(
        cif_txt,
        source="SWORD",
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        parser_occ_tolerance=parser_occ_tolerance,
        conventional_struct=conventional_struct,
        refine_struct=refine_struct,
        recover_cif_spg=recover_cif_spg,
        symprec_scan=symprec_scan,
        symprec_scan_mode=symprec_scan_mode,
    )

    info = disorder_label(
        entry,
        site_tolerance=site_tolerance,
        occ_tolerance=occ_tolerance,
        vac_tolerance=vac_tolerance,
        frac_tolerance=frac_tolerance,
        include_positional_disorder=include_positional_disorder,
        verbose=False,
    )
    if getattr(entry, "symprec_scan", None) is not None:
        info["symprec_scan"] = entry.symprec_scan
        info["symprec_scan"]["wyck_float_warn_count"] = len(info.get("wyck_float_warn") or [])
    return entry, info

def get_sword_label_for_ICSD(
    collection_code,
    *,
    ICSD_df,
    site_tolerance: float = 1e-4,
    occ_tolerance: float = 1.0,
    vac_tolerance: float = 1e-2,
    frac_tolerance: float = 1e-4,
    meta=None,
) -> str:
    entry = StructureEntry.from_collection_code(
        collection_code,
        ICSD_df=ICSD_df,
        meta=meta,
    )
    return disorder_label(
        entry,
        site_tolerance=site_tolerance,
        occ_tolerance=occ_tolerance,
        vac_tolerance=vac_tolerance,
        frac_tolerance=frac_tolerance,
        include_positional_disorder=False,
        verbose=False,
    )["disorder_label"]

def get_sword_info_for_ICSD(
    collection_code,
    *,
    ICSD_df,
    site_tolerance: float = 1e-4,
    occ_tolerance: float = 1.0,
    vac_tolerance: float = 1e-2,
    frac_tolerance: float = 1e-4,
    meta=None,
    include_positional_disorder: bool = True,
) -> str:
    entry = StructureEntry.from_collection_code(
        collection_code,
        ICSD_df=ICSD_df,
        meta=meta,
    )
    return entry, disorder_label(
        entry,
        site_tolerance=site_tolerance,
        occ_tolerance=occ_tolerance,
        vac_tolerance=vac_tolerance,
        frac_tolerance=frac_tolerance,
        include_positional_disorder=include_positional_disorder,
        verbose=False,
    )

def get_sword_label_from_pyxtal(pyxtal_dict):
    """
    Generates a SWORD label from a PyXtal dictionary representation.

    Args:
        pyxtal_dict (dict): A dictionary containing 'group' (int), 
                            'sites' (list of lists of strings), 
                            and 'species' (list of strings).

    Returns:
        str: The standardized SWORD label.
    """
    sg_num = int(pyxtal_dict['group'])
    species_list = pyxtal_dict['species']
    sites_list = pyxtal_dict['sites']
    
    # 1. Group elements by Wyckoff letter
    # Flatten the PyXtal structure: Wyckoff Letter -> List of elements on that site
    wyck_groups = defaultdict(list)
    
    for species, site_sublist in zip(species_list, sites_list):
        for site_str in site_sublist:
            # Extract just the letter (e.g., '4h' -> 'h', '18e' -> 'e')
            letter = "".join(filter(str.isalpha, site_str))
            wyck_groups[letter].append(species)
            
    # 2. Construct the sequence map (Key: WyckoffPart, Value: ElementPart)
    sequence_map = {}
    
    for letter, elems in wyck_groups.items():
        count = len(elems) # Number of distinct orbits (sites) for this letter
        
        # Build Wyckoff Part (e.g., 'a' or 'a12')
        wyck_key = f"{letter}{count}" if count > 1 else letter
        
        # Build Element Part
        # Count element occurrences: e.g. {'Sn': 5, 'Te': 6, 'Ag': 1}
        cnts = Counter(elems)
        parts = []
        for el, n in cnts.items():
            if n > 1:
                parts.append(f"{n}{el}")
            else:
                parts.append(el)
        
        # Sort parts as strings (standard lexicographical sort matches SWORD format: 10 < 2, 5 < 6 < A)
        parts.sort()
        
        # Determine formatting based on orbit count
        # If multiple sites exist for this letter, enclose in brackets {}
        if count == 1:
            elem_val = parts[0]
        else:
            elem_val = "{" + ",".join(parts) + "}"
            
        sequence_map[wyck_key] = elem_val

    # PyXtal-generated sequences have some canonicalization issues; bypassing for now (need to investigate further)
    wyckoff_set_std, element_seq, _ = get_canonical_wyckoff_sets(
        sequence_map, 
        mapping_by_sg, 
        sg_num
    )
    
    return f"{wyckoff_set_std}_{sg_num}_{element_seq}"
