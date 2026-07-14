import re
import warnings
from collections import defaultdict
from io import StringIO
from typing import Any

import pandas as pd
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Magmom, Structure
from pymatgen.io.cif import CifBlock, CifFile, CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

_SYMPREC_SCAN_FACTORS = {
    "strict": (1 / 100, 1 / 75, 1 / 50),
    "loose": (1 / 100, 1 / 75, 1 / 50, 1 / 20, 1 / 15, 1 / 10, 1 / 8),
    "recovery": (1 / 20, 1 / 15, 1 / 10, 1 / 8, 1 / 5),
}


def _recorded_space_group_number(cif_txt):
    m = re.search(
        r"_(?:space_group_IT_number|symmetry_Int_Tables_number)\s+['\"]?(\S+?)['\"]?(?:\s|$)",
        cif_txt,
        re.IGNORECASE,
    )
    if not m:
        return None
    try:
        return int(re.sub(r"\(.*\)", "", m.group(1)))
    except Exception:
        return None


def _nearest_reasonable_neighbor_distance(structure, *, min_pair_distance=0.7, quantile=0.20):
    nearest = []
    for i in range(len(structure)):
        ds = [
            structure.get_distance(i, j)
            for j in range(len(structure))
            if i != j and structure.get_distance(i, j) >= min_pair_distance
        ]
        if ds:
            nearest.append(min(ds))
    if not nearest:
        raise ValueError("Could not determine a reasonable nearest-neighbor distance.")
    nearest = sorted(nearest)
    idx = min(len(nearest) - 1, max(0, int(round((len(nearest) - 1) * quantile))))
    return nearest[idx]


def _symprec_scan_grid(d_ref, mode, anchor):
    if mode not in _SYMPREC_SCAN_FACTORS:
        raise ValueError("symprec_scan_mode must be one of 'strict', 'loose', or 'recovery'.")
    values = [d_ref * factor for factor in _SYMPREC_SCAN_FACTORS[mode]]
    if anchor is not None:
        values.append(float(anchor))
    return sorted({round(v, 12) for v in values if v > 0})


def _wyckoff_signature(dataset):
    counts = defaultdict(int)
    for wyckoff, equiv in zip(dataset.wyckoffs, dataset.equivalent_atoms):
        counts[(str(wyckoff), int(equiv))] += 1
    return tuple(sorted((wyckoff, count) for (wyckoff, _), count in counts.items()))


def _symmetry_candidate(structure, symprec, angle_tolerance):
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
        raw_sg = int(sga.get_space_group_number())
        dataset = sga.get_symmetry_dataset()

        conv_sg = None
        prim_sg = None
        conv_error = None
        prim_error = None
        try:
            conv = sga.get_conventional_standard_structure()
            conv_sg = int(
                SpacegroupAnalyzer(conv, symprec=symprec, angle_tolerance=angle_tolerance)
                .get_space_group_number()
            )
        except Exception as exc:
            conv_error = repr(exc)

        try:
            prim = sga.get_primitive_standard_structure()
            prim_sg = int(
                SpacegroupAnalyzer(prim, symprec=symprec, angle_tolerance=angle_tolerance)
                .get_space_group_number()
            )
        except Exception as exc:
            prim_error = repr(exc)

        if structure.is_ordered:
            ok = raw_sg == conv_sg == prim_sg
        else:
            ok = True
        signature = (raw_sg, _wyckoff_signature(dataset), ok)
        return {
            "symprec": symprec,
            "tau": None,
            "sg": raw_sg,
            "conv_sg": conv_sg,
            "prim_sg": prim_sg,
            "conv_error": conv_error,
            "prim_error": prim_error,
            "ok": ok,
            "signature": signature,
            "error": None,
        }
    except Exception as exc:
        return {
            "symprec": symprec,
            "tau": None,
            "sg": None,
            "conv_sg": None,
            "prim_sg": None,
            "ok": False,
            "signature": None,
            "error": repr(exc),
        }


def _merge_symmetry_plateaus(candidates):
    plateaus = []
    current = None
    for cand in candidates:
        if not cand["ok"]:
            current = None
            continue
        key = cand["signature"]
        if current is None or current["signature"] != key:
            current = {
                "signature": key,
                "sg": cand["sg"],
                "points": [cand],
            }
            plateaus.append(current)
        else:
            current["points"].append(cand)
    return plateaus


def _select_symmetry_plateau(plateaus, mode):
    if not plateaus:
        return None
    stable = [p for p in plateaus if len(p["points"]) >= 2]
    pool = plateaus if mode == "recovery" else (stable or plateaus)
    if mode == "strict":
        return min(pool, key=lambda p: p["points"][0]["symprec"])
    return max(pool, key=lambda p: (p["points"][-1]["symprec"], len(p["points"])))


def select_symprec_by_scan(structure, *, symprec=1e-2, angle_tolerance=5.0, mode="loose"):
    d_ref = _nearest_reasonable_neighbor_distance(structure)
    candidates = []
    for sp in _symprec_scan_grid(d_ref, mode, symprec):
        cand = _symmetry_candidate(structure, sp, angle_tolerance)
        cand["tau"] = sp / d_ref
        candidates.append(cand)

    plateaus = _merge_symmetry_plateaus(candidates)
    selected = _select_symmetry_plateau(plateaus, mode)
    if selected is None:
        return symprec, {
            "enabled": True,
            "mode": mode,
            "d_ref": d_ref,
            "selected_symprec": symprec,
            "selected_sg": None,
            "plateaus": [],
            "warning": "no_self_consistent_plateau",
        }

    selected_point = selected["points"][-1] if mode != "strict" else selected["points"][0]
    metadata = {
        "enabled": True,
        "mode": mode,
        "d_ref": d_ref,
        "selected_symprec": selected_point["symprec"],
        "selected_tau": selected_point["tau"],
        "selected_sg": selected["sg"],
        "selected_conv_sg": selected_point["conv_sg"],
        "selected_prim_sg": selected_point["prim_sg"],
        "selected_cell_check": "enforced" if structure.is_ordered else "not_enforced_for_disordered",
        "plateaus": [
            {
                "sg": p["sg"],
                "n_points": len(p["points"]),
                "symprec_min": p["points"][0]["symprec"],
                "symprec_max": p["points"][-1]["symprec"],
            }
            for p in plateaus
        ],
    }
    if selected_point["tau"] >= 0.2 - 1e-8:
        metadata["warning"] = "very_loose_candidate"
    elif selected_point["tau"] >= 0.1 - 1e-8:
        metadata["warning"] = "loose_candidate"
    else:
        metadata["warning"] = None
    return selected_point["symprec"], metadata


class _SymmetrizedCifWriter:
    """A customized Pymatgen CIFwrapper to write symmetrized CIF files with wyckoff letter from CIF raw txt."""
    def __init__(
        self,
        cif_txt: str,
        struct: Structure | None = None,
        symprec: float | None = 1e-2,
        write_magmoms: bool = False,
        significant_figures: int = 8,
        angle_tolerance: float = 5,
        parser_occ_tolerance: float = 1.05,
        refine_struct: bool = False,
        conventional_struct: bool = True,
        write_site_properties: bool = False,
        spg_analyzer: SpacegroupAnalyzer = None
    ) -> None:
        """
        Args:
            struct (Structure): structure to write.
            symprec (float): If not none, finds the symmetry of the structure
                and writes the CIF with symmetry information. Passes symprec
                to the SpacegroupAnalyzer. See also refine_struct.
            write_magmoms (bool): If True, will write magCIF file. Incompatible
                with symprec
            significant_figures (int): Specifies precision for formatting of floats.
                Defaults to 8.
            angle_tolerance (float): Angle tolerance for symmetry finding. Passes
                angle_tolerance to the SpacegroupAnalyzer. Used only if symprec
                is not None.
            refine_struct: Used only if symprec is not None. If True, get_refined_structure
                is invoked to convert input structure from primitive to conventional.
            write_site_properties (bool): Whether to write the Structure.site_properties
                to the CIF as _atom_site_{property name}. Defaults to False.
        """

        if write_magmoms and symprec is not None:
            warnings.warn(
                "Magnetic symmetry cannot currently be detected by pymatgen, disabling symmetry detection.",
                stacklevel=2,
            )
            symprec = None

        if struct is None:
            if any(k in cif_txt for k in ("_atom_site", "_cell_length", "data_")):
                cif_str = StringIO(cif_txt)
                parser = CifParser(cif_str, occupancy_tolerance=parser_occ_tolerance)
                struct = parser.parse_structures(primitive=False)[0]
            else:
                struct = Structure.from_str(cif_txt, fmt="poscar")

        blocks: dict[str, Any] = {}
        spacegroup: tuple[str, int] = ("P 1", 1)
        if symprec is not None:
            if spg_analyzer is not None:
                spg_analyzer = spg_analyzer
            else:
                spg_analyzer = SpacegroupAnalyzer(struct, symprec, angle_tolerance=angle_tolerance)
            spacegroup = (
                spg_analyzer.get_space_group_symbol(),
                spg_analyzer.get_space_group_number(),
            )

            if refine_struct:
                # Need the refined structure when using symprec. This converts
                # primitive to conventional structures, the standard for CIF, 
                # with atoms moved to the expected symmetry positions.
                struct = spg_analyzer.get_refined_structure()
            if conventional_struct:
                # Get a structure with a conventional cell according to certain standards.
                # This is not necessarily the same as the standard settings within the 
                # International Tables of Crystallography, for which get_refined_structure 
                # should be used instead.
                struct = spg_analyzer.get_conventional_standard_structure()

        lattice = struct.lattice
        comp = struct.composition
        no_oxi_comp = comp.element_composition
        format_str: str = f"{{:.{significant_figures}f}}"
        blocks["_symmetry_space_group_name_H-M"] = spacegroup[0]
        for cell_attr in ("a", "b", "c"):
            blocks[f"_cell_length_{cell_attr}"] = format_str.format(getattr(lattice, cell_attr))
        for cell_attr in ("alpha", "beta", "gamma"):
            blocks[f"_cell_angle_{cell_attr}"] = format_str.format(getattr(lattice, cell_attr))
        blocks["_symmetry_Int_Tables_number"] = spacegroup[1]
        blocks["_chemical_formula_structural"] = no_oxi_comp.reduced_formula
        blocks["_chemical_formula_sum"] = no_oxi_comp.formula
        blocks["_cell_volume"] = format_str.format(lattice.volume)

        _, fu = no_oxi_comp.get_reduced_composition_and_factor()
        blocks["_cell_formula_units_Z"] = str(int(fu))

        if symprec is None:
            blocks["_symmetry_equiv_pos_site_id"] = ["1"]
            blocks["_symmetry_equiv_pos_as_xyz"] = ["x, y, z"]

        else:
            spg_analyzer = SpacegroupAnalyzer(struct, symprec)
            symm_ops: list[SymmOp] = []
            for op in spg_analyzer.get_symmetry_operations():
                v = op.translation_vector
                symm_ops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))

            ops = [op.as_xyz_str() for op in symm_ops]
            blocks["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
            blocks["_symmetry_equiv_pos_as_xyz"] = ops

        loops: list[list[str]] = [
            ["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"],
        ]

        try:
            symbol_to_oxi_num = {str(el): float(el.oxi_state or 0) for el in sorted(comp.elements)}
            blocks["_atom_type_symbol"] = list(symbol_to_oxi_num)
            blocks["_atom_type_oxidation_number"] = symbol_to_oxi_num.values()
            loops.append(["_atom_type_symbol", "_atom_type_oxidation_number"])
        except (TypeError, AttributeError):
            symbol_to_oxi_num = {el.symbol: 0 for el in sorted(comp.elements)}

        atom_site_type_symbol = []
        atom_site_symmetry_multiplicity = []
        atom_site_fract_x = []
        atom_site_fract_y = []
        atom_site_fract_z = []
        atom_site_label = []
        atom_site_occupancy = []
        atom_site_moment_label = []
        atom_site_moment_crystalaxis_x = []
        atom_site_moment_crystalaxis_y = []
        atom_site_moment_crystalaxis_z = []
        atom_site_properties: dict[str, list] = defaultdict(list)
        wyckoffs: list[str] = []  #####################
        count = 0
        if symprec is None:
            for site in struct:
                for sp, occu in sorted(site.species.items()):
                    atom_site_type_symbol.append(str(sp))
                    atom_site_symmetry_multiplicity.append("1")
                    atom_site_fract_x.append(format_str.format(site.a))
                    atom_site_fract_y.append(format_str.format(site.b))
                    atom_site_fract_z.append(format_str.format(site.c))
                    atom_site_occupancy.append(str(occu))
                    site_label = f"{sp.symbol}{count}"

                    if "magmom" in site.properties:
                        mag = site.properties["magmom"]
                    elif getattr(sp, "spin", None) is not None:
                        mag = sp.spin
                    else:
                        # Use site label if available for regular sites
                        site_label = site.label if site.label != site.species_string else site_label
                        mag = 0

                    atom_site_label.append(site_label)

                    magmom = Magmom(mag)
                    if write_magmoms and abs(magmom) > 0:
                        moment = Magmom.get_moment_relative_to_crystal_axes(magmom, lattice)
                        atom_site_moment_label.append(f"{sp.symbol}{count}")
                        atom_site_moment_crystalaxis_x.append(format_str.format(moment[0]))
                        atom_site_moment_crystalaxis_y.append(format_str.format(moment[1]))
                        atom_site_moment_crystalaxis_z.append(format_str.format(moment[2]))

                    if write_site_properties:
                        for key, val in site.properties.items():
                            atom_site_properties[key].append(format_str.format(val))

                    count += 1

        else:
            # The following just presents a deterministic ordering
            symm_struct = spg_analyzer.get_symmetrized_structure() #############
            dataset = spg_analyzer.get_symmetry_dataset()###############
            wyck_all = dataset.wyckoffs ###############
            unique_sites = [
                (
                    min(sites, key=lambda site: tuple(abs(x) for x in site.frac_coords)),
                    len(sites),
                    wyck_all[indices[0]],#############
                )
                for sites, indices in zip(symm_struct.equivalent_sites,symm_struct.equivalent_indices,)  # type: ignore[reportPossiblyUnboundVariable]
            ]#############
            for site, mult, wyck in sorted( ##########
                unique_sites,
                key=lambda t: (
                    t[0].species.average_electroneg,
                    -t[1],
                    t[0].a,
                    t[0].b,
                    t[0].c,
                ),
            ):
                for sp, occu in site.species.items():
                    atom_site_type_symbol.append(str(sp))
                    atom_site_symmetry_multiplicity.append(f"{mult}")
                    atom_site_fract_x.append(format_str.format(site.a))
                    atom_site_fract_y.append(format_str.format(site.b))
                    atom_site_fract_z.append(format_str.format(site.c))
                    site_label = site.label if site.label != site.species_string else f"{sp.symbol}{count}"
                    atom_site_label.append(site_label)
                    atom_site_occupancy.append(str(occu))
                    wyckoffs.append(wyck) ##########
                    count += 1

        if len(set(atom_site_label)) != len(atom_site_label):
            warnings.warn(
                "Site labels are not unique, which is not compliant with the CIF spec "
                "(https://www.iucr.org/__data/iucr/cifdic_html/1/cif_core.dic/Iatom_site_label.html):"
                f"`{atom_site_label}`.",
                stacklevel=2,
            )

        blocks["_atom_site_type_symbol"] = atom_site_type_symbol
        blocks["_atom_site_label"] = atom_site_label
        blocks["_atom_site_symmetry_multiplicity"] = atom_site_symmetry_multiplicity
        if wyckoffs:
            blocks["_atom_site_Wyckoff_symbol"] = wyckoffs ###########
        blocks["_atom_site_fract_x"] = atom_site_fract_x
        blocks["_atom_site_fract_y"] = atom_site_fract_y
        blocks["_atom_site_fract_z"] = atom_site_fract_z
        blocks["_atom_site_occupancy"] = atom_site_occupancy
        loop_labels = [
            "_atom_site_type_symbol",
            "_atom_site_label",
            "_atom_site_symmetry_multiplicity",]
        if wyckoffs:
            loop_labels.append("_atom_site_Wyckoff_symbol") ##########
        loop_labels += [
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy",
        ]

        if write_site_properties:
            for key, vals in atom_site_properties.items():
                blocks[f"_atom_site_{key}"] = vals
                loop_labels += [f"_atom_site_{key}"]
        loops.append(loop_labels)

        if write_magmoms:
            blocks["_atom_site_moment_label"] = atom_site_moment_label
            blocks["_atom_site_moment_crystalaxis_x"] = atom_site_moment_crystalaxis_x
            blocks["_atom_site_moment_crystalaxis_y"] = atom_site_moment_crystalaxis_y
            blocks["_atom_site_moment_crystalaxis_z"] = atom_site_moment_crystalaxis_z
            loops.append(
                [
                    "_atom_site_moment_label",
                    "_atom_site_moment_crystalaxis_x",
                    "_atom_site_moment_crystalaxis_y",
                    "_atom_site_moment_crystalaxis_z",
                ]
            )
        dct = {comp.reduced_formula: CifBlock(blocks, loops, comp.reduced_formula)}
        self._cf = CifFile(dct)

def is_symmetrized_CIF(raw_txt, sym_txt):
    if not any(k in raw_txt for k in ("_atom_site", "_cell_length", "data_")):
        return False

    def count_sites(txt):
        lines = txt.splitlines()
        count, in_block = 0, False
        for ln in lines:
            if ln.strip().startswith("loop_"):
                in_block = False
            if "_atom_site_Wyckoff_symbol" in ln:
                in_block = True
                continue
            if in_block:
                s = ln.strip()
                if s.startswith("_"):
                    in_block = False
                elif s:
                    count += 1
        return count

    if count_sites(raw_txt) != count_sites(sym_txt):
        return False
    
    def get_it(txt):
        for key in ("_space_group_IT_number", "_symmetry_Int_Tables_number"):
            m = re.search(rf"{key}\s+['\"]?(\S+?)['\"]?(?:\s|$)", txt)
            if m:
                return m.group(1)
        return None

    raw_it = get_it(raw_txt)
    sym_it = get_it(sym_txt)
    if raw_it is not None and sym_it is not None and raw_it != sym_it:
        warnings.warn(
            "It appears that a symmetrized CIF is provided, but the "
            "_space_group_IT_number recorded in the original file "
            f"({raw_it}) is inconsistent with the space group obtained "
            f"after symmetrization ({sym_it}) using spglib. Please double-check "
            f"the structure and its symmetry settings. Using the CIF symmetrized (space group {sym_it}) by spglib."
        )
        return False

    if "_atom_site_Wyckoff_symbol" not in raw_txt:
        return False

    return True

def find_symprec(raw_txt, *, n=2, symprec=1e-2, angle_tolerance=5.0, parser_occ_tolerance: float = 1.05, merge_tolerance:float = 0.01):
    """
    Find symprec that yields a valid symmetry analysis matching the CIF space group. Cleanup of too closed sites below tolerance.
    """
    # retrieve target space group from CIF
    m = re.search(r"_(?:space_group_IT_number|symmetry_Int_Tables_number)\s+['\"]?(\S+?)['\"]?(?:\s|$)", raw_txt, re.IGNORECASE)
    if not m:
        raise ValueError("No _space_group_IT_number found in CIF text.")
    target_sg = int(m.group(1))

    #merge_triggers = ("Incorrect stoichiometry",)
    #with warnings.catch_warnings(record=True) as warn_list:
    #        warnings.simplefilter("always")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_structure = CifParser(StringIO(raw_txt), occupancy_tolerance=parser_occ_tolerance).parse_structures(primitive=False)[0]

    merged_base = False
    # for w in warn_list:
    #     if any(k in str(w.message) for k in merge_triggers):
    #         warnings.warn((f"[Warning Triggered] {w.message}"))
    #         base_structure = base_structure.merge_sites(mode="delete", tol = merge_tolerance)
    #         merged_base = True
    #         break

    candidates = [symprec]
    for i in range(1, n + 1):
        candidates.append(symprec / (10**i))
        candidates.append(symprec * (10**i))
        
    parsed_ok = False
    best_diff = None
    best_match = None # (sp, merged, struct, sga, sgnum)
    for sp in candidates:
        struct = base_structure.copy()
        merged = merged_base  # initial state

        try:
            sga = SpacegroupAnalyzer(struct, symprec=sp, angle_tolerance=angle_tolerance)
            sgnum = sga.get_space_group_number()
            parsed_ok = True
            if target_sg == 1 or int(sgnum) == target_sg:
                return sp, merged, struct, sga
            
            if parsed_ok and sgnum is not None: #actually this condition is unnecessary
                diff = abs(int(sgnum) - target_sg)

            if best_diff is None or diff < best_diff:
                best_diff = diff
                best_match = (sp, merged, struct, sga, sgnum)
        except Exception:
            sgnum = None  # fall through to merge retry
        
        # if not merged:
        #     # retry after merging close sites
        #     struct = struct.merge_sites(mode="delete", tol = merge_tolerance) 
        #     merged = True
        #     try:
        #         sga = SpacegroupAnalyzer(struct, symprec=sp, angle_tolerance=angle_tolerance)
        #         sgnum = sga.get_space_group_number()
        #         parsed_ok = True
        #         if target_sg == "1" or str(sgnum) == target_sg:
        #             return sp, merged, struct, sga
        #     except Exception:
        #         continue

    if not parsed_ok:
        raise RuntimeError(f"No symprec setting allowed SpacegroupAnalyzer to parse the structure within {2 * (1 + 2*n)} attempts.")
    if best_match is not None:
        sp_best, merged, struct, sga, sgnum_best = best_match
        warnings.warn(f"No symprec setting could reproduced the original space group {target_sg} within {2 * (1 + 2*n)} iterations;"
                      f"returning closest match {sgnum_best} at symprec={sp_best}.")
        return sp_best, merged, struct, sga

def clean_num(token):
    if token is None:
        return None
    if isinstance(token, (int, float)):
        return float(token)
    if isinstance(token, bytes):
        token = token.decode("utf-8")
    token = re.sub(r"\(.*\)", "", token)
    try:
        return float(token)
    except Exception:
        return None

def symm_orbits_df(
    structure: Structure,
    *,
    symprec: float = 1e-2,
    angle_tolerance: float = 5,
    id_key: str = "stable_id",
) -> pd.DataFrame:
    """
    Summarize symmetry-equivalent site orbits (with Wyckoff letters) as a DataFrame.

    The returned table is meant for debugging how sites are grouped under symmetry.
    Per-member element/label/id information is taken from the *input* structure via
    equivalent_indices (i.e., it is not affected by CIF re-writing/re-parsing).
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    symm_struct = sga.get_symmetrized_structure()
    dataset = sga.get_symmetry_dataset()

    rows: list[dict[str, Any]] = []
    for sites, idxs in zip(symm_struct.equivalent_sites, symm_struct.equivalent_indices):
        idxs = list(idxs)
        rep_idx = idxs[0]
        rep_site = structure[rep_idx]

        member_sites = [structure[i] for i in idxs]
        member_ids = [s.properties.get(id_key) for s in member_sites]
        member_elements = [
            tuple(sorted(el.symbol for el in s.species.elements)) for s in member_sites
        ]

        rows.append(
            {
                "wyckoff_symbol": dataset.wyckoffs[rep_idx],
                "multiplicity": len(idxs),
                "rep_symm_frac": list(sites[0].frac_coords),
                "rep_species": rep_site.species_string,
                "rep_species_detail": str(rep_site.species),
                "rep_elements": tuple(sorted(el.symbol for el in rep_site.species.elements)),
                "member_indices": idxs,
                "member_ids": member_ids,
                "member_labels": [getattr(s, "label", None) for s in member_sites],
                "member_species": [s.species_string for s in member_sites],
                "member_elements": member_elements,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = (
            df.sort_values(["wyckoff_symbol", "multiplicity"], ascending=[True, False])
            .reset_index(drop=True)
        )
    return df

class StructureEntry:
    """
    ICSD CIF wrapper. Reads a CIF block, preserves the raw data, and optionally
    fills missing crystallographic info using pymatgen + spglib.
    """
    def __init__(self, cif_text, collection_code=None, meta=None):
        self.CollectionCode = collection_code
        parser = CifParser(StringIO(cif_text))
        cif_dict = parser.as_dict()
        block_name = next(iter(cif_dict))
        block = cif_dict[block_name]

        payload = self._payload_from_block(block, cif_dict, cif_text)
        self._apply_payload(payload, meta)

    @staticmethod
    def _payload_from_block(block, cif_dict, cif_text):
        raw_sg = block.get("_space_group_IT_number") or block.get("_symmetry_Int_Tables_number")
        sym_ops_raw = block.get("_space_group_symop_operation_xyz") or block.get("_symmetry_equiv_pos_as_xyz")
        try:
            sg_number = int(re.sub(r"\(.*\)", "", raw_sg))
        except Exception:
            sg_number = (
                block.get("_space_group_IT_number") or block.get("_symmetry_Int_Tables_number")
            )

        payload = {
            "cif_text": cif_text,
            "cif_dict": cif_dict,
            "space_group_number": sg_number,
            "lattice": (
                clean_num(block.get("_cell_length_a")),
                clean_num(block.get("_cell_length_b")),
                clean_num(block.get("_cell_length_c")),
            ),
            "labels": list(block.get("_atom_site_label", [])),
            "type_symbols": list(block.get("_atom_site_type_symbol", [])),
            "coords": list(
                zip(
                    [clean_num(x) for x in block.get("_atom_site_fract_x", [])],
                    [clean_num(y) for y in block.get("_atom_site_fract_y", [])],
                    [clean_num(z) for z in block.get("_atom_site_fract_z", [])],
                )
            ),
            "occupancies": [clean_num(x) for x in block.get("_atom_site_occupancy", [])],
            "multiplicities": [int(m) for m in block.get("_atom_site_symmetry_multiplicity", [])],
            "wyckoffs": list(block.get("_atom_site_Wyckoff_symbol", [])),
            "sym_ops": [SymmOp.from_xyz_str(op) for op in sym_ops_raw],
        }
        return payload

    def _apply_payload(self, payload, meta):
        self.meta = meta or {}
        self.cif_str = payload["cif_text"]
        self.cif_dict = payload["cif_dict"]
        self.spg_num = int(payload["space_group_number"])
        self.lattice = payload["lattice"]

        self.labels = payload["labels"]
        self.type_symbols = payload["type_symbols"]
        self.points = [tuple(pt) for pt in payload["coords"]]
        self.occ = payload["occupancies"]
        self.mults = payload["multiplicities"]
        self.wyckoffs = payload["wyckoffs"]
        self.sym_ops = [op if isinstance(op, SymmOp) else SymmOp.from_xyz_str(op) for op in payload["sym_ops"]]

        if not hasattr(self, "read_by"):
            self.read_by = "raw"

        if not hasattr(self, "source"):
            self.source = "ICSD"

        if not hasattr(self, "is_merge"):
            self.is_merge = False

        self.records = []
        for i, label in enumerate(self.labels):
            self.records.append(
                {
                    "label": label,
                    "type_symbol": self.type_symbols[i],
                    "multiplicity": self.mults[i],
                    "wyckoff_symbol": self.wyckoffs[i],
                    "coordinate": list(self.points[i]),
                    "occupancy": self.occ[i],
                }
            )
        self.records = sorted(self.records, key=lambda x: x["label"])
        self.df = pd.DataFrame(self.records)

    @classmethod
    def from_txt(
        cls,
        raw_txt,
        *,
        code=None,
        meta=None,
        symprec=1e-2,
        angle_tolerance=5,
        parser_occ_tolerance: float = 1.05,
        source: str = 'None',
        conventional_struct: bool = True,
        refine_struct: bool = False,
        recover_cif_spg: bool = False,
        symprec_scan: bool = False,
        symprec_scan_mode: str = "loose",
    ):
        #construct a ICSD-style symmetrized CIF_txt from raw CIF or POSCAR
        scan_meta = None
        if isinstance(raw_txt, Structure):
            struct = raw_txt
            raw_txt = ""
            if symprec_scan:
                symprec, scan_meta = select_symprec_by_scan(
                    struct,
                    symprec=symprec,
                    angle_tolerance=angle_tolerance,
                    mode=symprec_scan_mode,
                )
            sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
            is_merge = False
        elif any(k in raw_txt for k in ("_atom_site", "_cell_length", "data_")):
            if recover_cif_spg:
                try:
                    symprec, is_merge, struct, sga = find_symprec(raw_txt, n=2, symprec=symprec, angle_tolerance=angle_tolerance, parser_occ_tolerance=parser_occ_tolerance)
                except Exception as e:
                    warnings.warn(
                        f"[StructureEntry.from_txt] find_symprec failed: falling back to use default symprec."
                        f"symprec={symprec}, angle_tol={angle_tolerance}. "
                        f"Error: {e}")
                    symprec, is_merge, struct, sga = symprec, False, None, None
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    struct = CifParser(StringIO(raw_txt), occupancy_tolerance=parser_occ_tolerance).parse_structures(primitive=False)[0]
                if symprec_scan:
                    symprec, scan_meta = select_symprec_by_scan(
                        struct,
                        symprec=symprec,
                        angle_tolerance=angle_tolerance,
                        mode=symprec_scan_mode,
                    )
                sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
                is_merge = False
        else:
            struct = Structure.from_str(raw_txt, fmt="poscar")
            if symprec_scan:
                symprec, scan_meta = select_symprec_by_scan(
                    struct,
                    symprec=symprec,
                    angle_tolerance=angle_tolerance,
                    mode=symprec_scan_mode,
                )
            sga = SpacegroupAnalyzer(struct, symprec=symprec, angle_tolerance=angle_tolerance)
            is_merge = False
        sym_txt = str(_SymmetrizedCifWriter(raw_txt, symprec=symprec, angle_tolerance=angle_tolerance, parser_occ_tolerance=parser_occ_tolerance, struct=struct, spg_analyzer=sga, conventional_struct=conventional_struct, refine_struct=refine_struct)._cf)
        # Check completeness of input to determine whether to parse raw or symmetrized file
        #use_sym = not is_symmetrized_CIF(raw_txt, sym_txt)
        #chosed_txt = sym_txt if use_sym else raw_txt
        chosed_txt = sym_txt
        parser = CifParser(StringIO(chosed_txt))
        cif_dict = parser.as_dict()
        block_name = next(iter(cif_dict))
        block = cif_dict[block_name]

        payload = cls._payload_from_block(block, cif_dict, chosed_txt)

        entry = cls.__new__(cls)
        entry.CollectionCode = code
        entry.is_merge = is_merge
        #entry.read_by = "pmg" if use_sym else "raw"
        entry.read_by = "pmg"
        entry.source = source
        entry.symprec = symprec
        entry.symprec_scan = scan_meta
        entry._apply_payload(payload, meta)
        recorded_sg = _recorded_space_group_number(raw_txt)
        if recorded_sg is not None and recorded_sg != 1 and recorded_sg != entry.spg_num:
            warnings.warn(
                f"SWORD space group ({entry.spg_num}) is inconsistent with the CIF-recorded "
                f"space group ({recorded_sg}). CIF-recorded P1 is ignored silently; "
                "non-P1 mismatches should be checked."
            )
        return entry

    @classmethod
    def from_collection_code(cls, collection_code, ICSD_df: pd.DataFrame = None, meta=None):
        row = ICSD_df[ICSD_df["CollectionCode"] == collection_code].iloc[0]
        cif_text = row["cif"]
        if meta == "all":
            meta = row.to_dict()
        return cls(cif_text, meta=meta, collection_code=collection_code)
