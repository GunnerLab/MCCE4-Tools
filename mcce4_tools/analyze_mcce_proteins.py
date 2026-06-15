#!/usr/bin/env python3
"""
Combined MCCE analysis: buried-residue statistics, feature extraction, and plotting.

Usage:
    python analyze_mcce_proteins.py                  # analyze + plot (default)
    python analyze_mcce_proteins.py --no-plot        # analyze only, skip figures
    python analyze_mcce_proteins.py --plot-only      # regenerate figures from existing CSVs
    python analyze_mcce_proteins.py --ph 4.0         # analyze at a different pH
    python analyze_mcce_proteins.py -o results       # write all output to results/
"""
import argparse
import csv
import os
from pathlib import Path
import shutil
import sys
from collections import defaultdict


# ── Constants ────────────────────────────────────────────────────────────────
PKA0 = {
    "ASP": 4.75, "GLU": 4.75, "CYS": 9.10, "HIS": 6.98, "LYS": 10.4,
    "TYR": 10.20, "ARG": 12.5, "NTR": 8.00, "CTR": 3.75,
}

IONIZABLE_RES = set(PKA0.keys())

STANDARD_RES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
}

ALLOWED_RES = STANDARD_RES | IONIZABLE_RES

IONIZABLE_ORDER = ["ASP", "GLU", "CTR", "CYS", "HIS", "NTR", "TYR", "LYS", "ARG"]
NON_IONIZABLE_ORDER = [
    "ALA", "GLY", "ILE", "LEU", "MET", "PHE",
    "PRO", "SER", "THR", "TRP", "VAL", "ASN", "GLN",
]

ACIDS = {"ASP", "GLU", "CTR", "CYS", "TYR"}
BASES = {"ARG", "HIS", "LYS", "NTR"}
RT_LN10 = 1.364

PROTEIN_CSV = "protein_features.csv"
RESIDUE_CSV = "residue_features.csv"
FIG2_NAME = "protein_charge_analysis.png"
FIG3_NAME = "pka_energy_analysis.png"
LOG_NAME = "analysis.log"


# ── Parsing helpers ──────────────────────────────────────────────────────────
def list_pdb_dirs(topdir:str, sentinel: str = "step2_out.pdb"):
    """Using a sentinel file (any mcce run output file) allows
    for unrestricted parent folder name, e.g. frame10.
    """
    return sorted(
        fp.parent.name
        for fp in Path(topdir).glob(f"*/{sentinel}"))


def parse_acc_res(filepath):
    residues = {}
    total_sas = 0.0
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5 or parts[0] != "RES":
                continue
            restype, resid = parts[1], parts[2]
            try:
                abs_sas = float(parts[3])
                frac_acc = float(parts[4])
            except ValueError:
                continue
            residues[f"{restype}{resid}"] = (restype, resid, abs_sas, frac_acc)
            total_sas += abs_sas
    return residues, total_sas


def parse_sum_crg(filepath):
    ph_values, net_charges = [], []
    with open(filepath) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "ph":
                ph_values = [float(x) for x in parts[1:]]
            elif parts[0] == "Net_Charge":
                net_charges = [float(x) for x in parts[1:]]
    return ph_values, net_charges


def get_charge_at_ph(ph_values, net_charges, target_ph):
    if not ph_values or not net_charges:
        return None
    if target_ph in ph_values:
        return net_charges[ph_values.index(target_ph)]
    for i in range(len(ph_values) - 1):
        if ph_values[i] <= target_ph <= ph_values[i + 1]:
            frac = (target_ph - ph_values[i]) / (ph_values[i + 1] - ph_values[i])
            return net_charges[i] + frac * (net_charges[i + 1] - net_charges[i])
    return None


def calc_pi(ph_values, net_charges):
    if not ph_values or not net_charges:
        return None
    for i in range(len(net_charges) - 1):
        if net_charges[i] * net_charges[i + 1] < 0:
            frac = net_charges[i] / (net_charges[i] - net_charges[i + 1])
            return ph_values[i] + frac * (ph_values[i + 1] - ph_values[i])
        if net_charges[i] == 0.0:
            return ph_values[i]
    return None


def parse_pk_out(filepath):
    residues = []
    with open(filepath, errors="replace") as f:
        f.readline()
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            label = line[:14].strip()
            rest = line[14:].split()
            if not rest:
                continue
            sign = "+" if "+" in label else "-"
            restype = label.split(sign)[0].split("-")[0]
            resid = label.replace(restype + sign, "").rstrip("_")
            pka_str = rest[0]
            if pka_str.startswith(">") or pka_str.startswith("<"):
                pka_val = None
            else:
                try:
                    pka_val = float(pka_str)
                except ValueError:
                    pka_val = None
            residues.append((restype, resid, pka_val, pka_str))
    return residues


def compute_burial(topdir, threshold):
    total_by_type = defaultdict(int)
    buried_by_type = defaultdict(int)
    for d in list_pdb_dirs(topdir):
        acc_path = os.path.join(topdir, d, "acc.res")
        if not os.path.isfile(acc_path):
            continue
        with open(acc_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5 or parts[0] != "RES":
                    continue
                restype = parts[1]
                if restype == "NTG" or restype not in ALLOWED_RES:
                    continue
                try:
                    frac_acc = float(parts[4])
                except ValueError:
                    continue
                total_by_type[restype] += 1
                if frac_acc <= threshold:
                    buried_by_type[restype] += 1
    return {rt: 100.0 * buried_by_type.get(rt, 0) / total_by_type[rt]
            for rt in total_by_type if total_by_type[rt] > 0}


class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()


def _init_dipole():
    """Try to import MCCE4 dipole modules. Returns (parse, compute, np) or None."""
    try:
        mcce_path = shutil.which("mcce")
        if not mcce_path:
            return None
        mcce_bin_dir = os.path.normpath(
            os.path.join(os.path.dirname(os.path.realpath(mcce_path)), "..", "MCCE_bin"))
        if mcce_bin_dir not in sys.path:
            sys.path.insert(0, mcce_bin_dir)
        from mcce_dipole.parsers import parse_step2_pdb, parse_head3lst, parse_fort38
        from mcce_dipole.compute import compute_from_ensemble
        import numpy as np
        return parse_step2_pdb, parse_head3lst, parse_fort38, compute_from_ensemble, np
    except (ImportError, TypeError):
        return None


def _compute_dipole(pdb_dir, target_ph, dipole_funcs):
    """Compute backbone, ionizable, and full dipole magnitudes (Debye) at target_ph."""
    parse_step2_pdb, parse_head3lst, parse_fort38, compute_from_ensemble, np = dipole_funcs
    step2 = os.path.join(pdb_dir, "step2_out.pdb")
    head3 = os.path.join(pdb_dir, "head3.lst")
    fort38 = os.path.join(pdb_dir, "fort.38")
    if not all(os.path.isfile(f) for f in [step2, head3, fort38]):
        return None, None, None
    try:
        conformers, _ = parse_step2_pdb(step2)
        head3_data = parse_head3lst(head3)
        ph_values, conf_ids, occupancies = parse_fort38(fort38)
        results = compute_from_ensemble(conformers, head3_data, ph_values, conf_ids, occupancies)
        ph_idx = int(np.argmin(np.abs(results["ph_values"] - target_ph)))
        bb = float(np.linalg.norm(results["backbone_dipole"][ph_idx]))
        ion = float(np.linalg.norm(results["ionizable_dipole"][ph_idx]))
        full = float(np.linalg.norm(results["full_dipole"][ph_idx]))
        return bb, ion, full
    except Exception:
        return None, None, None


# ── Analysis (extract + buried stats + summary) ─────────────────────────────

def run_analysis(topdir, outdir, target_ph, burial_threshold, top_n,
                 include_ntg, all_residues):
    os.makedirs(outdir, exist_ok=True)

    log_path = os.path.join(outdir, LOG_NAME)
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    sys.stdout = _Tee(original_stdout, log_file)

    dipole_funcs = _init_dipole()
    if dipole_funcs:
        print("Dipole computation enabled (ms_dipole found)")
    else:
        print("Warning: ms_dipole not available, skipping dipole features")

    subdirs = list_pdb_dirs(topdir)

    exclude_ntg = not include_ntg
    standard_only = not all_residues

    protein_rows = []
    residue_rows = []
    errors = []

    # Per-protein burial tracking
    global_total_by_type = defaultdict(int)
    global_buried_by_type = defaultdict(int)
    burial_protein_stats = {}

    for pdb in subdirs:
        pdb_dir = os.path.join(topdir, pdb)
        acc_path = os.path.join(pdb_dir, "acc.res")
        crg_path = os.path.join(pdb_dir, "sum_crg.out")
        pk_path = os.path.join(pdb_dir, "pK.out")

        missing = [f for f in [acc_path, crg_path, pk_path] if not os.path.isfile(f)]
        if missing:
            errors.append((pdb, f"missing: {', '.join(os.path.basename(f) for f in missing)}"))
            continue

        # ── acc.res ──
        acc_residues, total_sas = parse_acc_res(acc_path)

        # burial stats from acc.res
        bur_total = defaultdict(int)
        bur_buried = defaultdict(int)
        for key, (restype, resid, abs_sas, frac_acc) in acc_residues.items():
            if exclude_ntg and restype == "NTG":
                continue
            if standard_only and restype not in ALLOWED_RES:
                continue
            bur_total[restype] += 1
            if frac_acc <= burial_threshold:
                bur_buried[restype] += 1
        burial_protein_stats[pdb] = {
            "total_by_type": dict(bur_total),
            "buried_by_type": dict(bur_buried),
        }
        for rt, c in bur_total.items():
            global_total_by_type[rt] += c
        for rt, c in bur_buried.items():
            global_buried_by_type[rt] += c

        # ── sum_crg.out ──
        ph_values, net_charges = parse_sum_crg(crg_path)
        net_charge = get_charge_at_ph(ph_values, net_charges, target_ph)
        pi = calc_pi(ph_values, net_charges)

        # ── pK.out ──
        pk_residues = parse_pk_out(pk_path)
        pka_shifts = []
        n_ionizable = 0
        for restype, resid, pka_val, pka_str in pk_residues:
            if restype not in IONIZABLE_RES:
                continue
            n_ionizable += 1
            pka0 = PKA0[restype]
            pka_shift = (pka_val - pka0) if pka_val is not None else None

            acc_key = f"{restype}{resid}"
            if acc_key in acc_residues:
                _, _, abs_sas, frac_acc = acc_residues[acc_key]
            else:
                abs_sas, frac_acc = None, None

            buried = (frac_acc <= burial_threshold) if frac_acc is not None else None
            if pka_shift is not None and restype in IONIZABLE_RES:
                if restype in ACIDS:
                    energy = pka_shift * RT_LN10
                else:
                    energy = pka_shift * (-RT_LN10)
            else:
                energy = None

            residue_rows.append({
                "pdb": pdb,
                "restype": restype,
                "resid": resid,
                "pKa": pka_str,
                "pKa0": pka0,
                "pKa_shift": f"{pka_shift:.3f}" if pka_shift is not None else pka_str,
                "energy_kcal": f"{energy:.3f}" if energy is not None else "",
                "abs_sas": f"{abs_sas:.3f}" if abs_sas is not None else "",
                "frac_acc": f"{frac_acc:.3f}" if frac_acc is not None else "",
                "buried": int(buried) if buried is not None else "",
            })
            if pka_shift is not None:
                pka_shifts.append(pka_shift)

        mean_shift = sum(pka_shifts) / len(pka_shifts) if pka_shifts else None
        max_abs_shift = max(abs(s) for s in pka_shifts) if pka_shifts else None
        n_large_shift = sum(1 for s in pka_shifts if abs(s) > 2.0)

        # ── Dipole ──
        if dipole_funcs:
            bb_dip, ion_dip, full_dip = _compute_dipole(pdb_dir, target_ph, dipole_funcs)
        else:
            bb_dip, ion_dip, full_dip = None, None, None

        n_total_res = sum(bur_total.values())
        n_buried_total = sum(bur_buried.values())
        n_ion_total = sum(v for rt, v in bur_total.items() if rt in IONIZABLE_RES)
        n_buried_ion = sum(v for rt, v in bur_buried.items() if rt in IONIZABLE_RES)

        charge_col = f"net_charge_pH{int(target_ph)}"
        row = {
            "pdb": pdb,
            "total_sas": f"{total_sas:.2f}",
            charge_col: f"{net_charge:.2f}" if net_charge is not None else "",
            "pI": f"{pi:.2f}" if pi is not None else "",
            "n_residues": n_total_res,
            "n_buried": n_buried_total,
            "frac_buried": f"{n_buried_total / n_total_res:.3f}" if n_total_res else "",
            "n_ionizable": n_ionizable,
            "n_ionizable_buried": n_buried_ion,
            "frac_ionizable_buried": f"{n_buried_ion / n_ion_total:.3f}" if n_ion_total else "",
            "n_with_pKa": len(pka_shifts),
            "mean_pKa_shift": f"{mean_shift:.3f}" if mean_shift is not None else "",
            "max_abs_pKa_shift": f"{max_abs_shift:.3f}" if max_abs_shift is not None else "",
            "n_large_shift_gt2": n_large_shift,
        }
        if dipole_funcs:
            row["backbone_dipole_D"] = f"{bb_dip:.2f}" if bb_dip is not None else ""
            row["ionizable_dipole_D"] = f"{ion_dip:.2f}" if ion_dip is not None else ""
            row["full_dipole_D"] = f"{full_dip:.2f}" if full_dip is not None else ""
        protein_rows.append(row)

    # ── Write CSVs ───────────────────────────────────────────────────────────
    if protein_rows:
        path = os.path.join(outdir, PROTEIN_CSV)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(protein_rows[0].keys()))
            w.writeheader()
            w.writerows(protein_rows)
        print(f"Wrote {len(protein_rows)} proteins to {path}")

    if residue_rows:
        path = os.path.join(outdir, RESIDUE_CSV)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(residue_rows[0].keys()))
            w.writeheader()
            w.writerows(residue_rows)
        print(f"Wrote {len(residue_rows)} ionizable residues to {path}")

    # ── Print summary ────────────────────────────────────────────────────────
    n_proteins = len(protein_rows)
    if n_proteins == 0:
        print("No PDB directories with complete data found.")
        sys.exit(1)

    charge_col = f"net_charge_pH{int(target_ph)}"

    print(f"\n{'=' * 70}")
    print(f"MCCE Analysis Summary  (pH = {target_ph})")
    print(f"Proteins: {n_proteins}   Skipped: {len(errors)}")
    print(f"Buried = fractional accessibility <= {burial_threshold}")
    print(f"{'=' * 70}")

    if errors:
        print(f"\nSkipped PDBs:")
        for pdb, reason in errors:
            print(f"  {pdb}: {reason}")

    # Protein-level stats
    charges_with_pdb = [(float(r[charge_col]), r["pdb"]) for r in protein_rows if r[charge_col]]
    pis_with_pdb = [(float(r["pI"]), r["pdb"]) for r in protein_rows if r["pI"]]
    sas_vals = [float(r["total_sas"]) for r in protein_rows]
    charges = [c for c, _ in charges_with_pdb]
    pis = [p for p, _ in pis_with_pdb]

    print(f"\n--- Protein-level features ---")
    if sas_vals:
        print(f"  Total SAS:     min={min(sas_vals):.1f}  max={max(sas_vals):.1f}  "
              f"mean={sum(sas_vals)/len(sas_vals):.1f}")

    # pI summary
    if pis:
        mean_pi = sum(pis) / len(pis)
        n_acidic = sum(1 for p in pis if p < 7)
        n_basic = sum(1 for p in pis if p >= 7)
        print(f"\n{'=' * 70}")
        print(f"Isoelectric Point (pI) Distribution")
        print(f"{'=' * 70}")
        print(f"  min={min(pis):.2f}  max={max(pis):.2f}  mean={mean_pi:.2f}")
        print(f"  Acidic (pI < 7): {n_acidic}   Basic (pI >= 7): {n_basic}")
        print(f"\n  {'Range':<14} {'Count':>6} {'Proteins':}")
        print(f"  {'-' * 60}")
        bins = [(0, 5, "pI < 5"), (5, 6, "5 <= pI < 6"), (6, 7, "6 <= pI < 7"),
                (7, 8, "7 <= pI < 8"), (8, 9, "8 <= pI < 9"), (9, 15, "pI >= 9")]
        for lo, hi, label in bins:
            in_bin = [(p, pdb) for p, pdb in pis_with_pdb if lo <= p < hi]
            if in_bin:
                pdbs = ", ".join(pdb for _, pdb in sorted(in_bin))
                if len(pdbs) > 50:
                    pdbs = pdbs[:50] + "..."
                print(f"  {label:<14} {len(in_bin):>6}   {pdbs}")

        print(f"\n  Most acidic:  ", end="")
        for p, pdb in sorted(pis_with_pdb)[:3]:
            print(f"{pdb} (pI={p:.2f})  ", end="")
        print(f"\n  Most basic:   ", end="")
        for p, pdb in sorted(pis_with_pdb, reverse=True)[:3]:
            print(f"{pdb} (pI={p:.2f})  ", end="")
        print()

    # Net charge summary
    if charges:
        mean_crg = sum(charges) / len(charges)
        n_pos = sum(1 for c in charges if c > 0)
        n_neg = sum(1 for c in charges if c < 0)
        n_neutral = sum(1 for c in charges if abs(c) <= 1.0)
        print(f"\n{'=' * 70}")
        print(f"Net Charge at pH {target_ph}")
        print(f"{'=' * 70}")
        print(f"  min={min(charges):.2f}  max={max(charges):.2f}  mean={mean_crg:.2f}")
        print(f"  Positive: {n_pos}   Negative: {n_neg}   Near-neutral (|q| <= 1): {n_neutral}")

        print(f"\n  {'Range':<20} {'Count':>6} {'Proteins':}")
        print(f"  {'-' * 60}")
        crg_bins = [(-999, -20, "q < -20"), (-20, -10, "-20 <= q < -10"),
                    (-10, -1, "-10 <= q < -1"), (-1, 1, "-1 <= q <= 1"),
                    (1, 10, "1 < q <= 10"), (10, 20, "10 < q <= 20"),
                    (20, 999, "q > 20")]
        for lo, hi, label in crg_bins:
            if lo == -1 and hi == 1:
                in_bin = [(c, pdb) for c, pdb in charges_with_pdb if lo <= c <= hi]
            else:
                in_bin = [(c, pdb) for c, pdb in charges_with_pdb if lo <= c < hi]
            if in_bin:
                pdbs = ", ".join(pdb for _, pdb in sorted(in_bin, key=lambda x: x[0]))
                if len(pdbs) > 50:
                    pdbs = pdbs[:50] + "..."
                print(f"  {label:<20} {len(in_bin):>6}   {pdbs}")

        print(f"\n  Most negative: ", end="")
        for c, pdb in sorted(charges_with_pdb)[:3]:
            print(f"{pdb} ({c:+.1f})  ", end="")
        print(f"\n  Most positive: ", end="")
        for c, pdb in sorted(charges_with_pdb, reverse=True)[:3]:
            print(f"{pdb} ({c:+.1f})  ", end="")
        print()

    # Dipole summary
    dipole_with_pdb = [(float(r["full_dipole_D"]), r["pdb"])
                       for r in protein_rows if r.get("full_dipole_D")]
    if dipole_with_pdb:
        dip_vals = [d for d, _ in dipole_with_pdb]
        mean_dip = sum(dip_vals) / len(dip_vals)
        bb_vals = [float(r["backbone_dipole_D"]) for r in protein_rows if r.get("backbone_dipole_D")]
        ion_vals = [float(r["ionizable_dipole_D"]) for r in protein_rows if r.get("ionizable_dipole_D")]
        print(f"\n{'=' * 70}")
        print(f"Dipole Moment at pH {target_ph}  (Debye)")
        print(f"{'=' * 70}")
        if bb_vals:
            print(f"  Backbone:   min={min(bb_vals):.1f}  max={max(bb_vals):.1f}  "
                  f"mean={sum(bb_vals)/len(bb_vals):.1f}")
        if ion_vals:
            print(f"  Ionizable:  min={min(ion_vals):.1f}  max={max(ion_vals):.1f}  "
                  f"mean={sum(ion_vals)/len(ion_vals):.1f}")
        print(f"  Full:       min={min(dip_vals):.1f}  max={max(dip_vals):.1f}  "
              f"mean={mean_dip:.1f}")
        print(f"\n  Largest dipole:   ", end="")
        for d, pdb in sorted(dipole_with_pdb, reverse=True)[:3]:
            print(f"{pdb} ({d:.1f} D)  ", end="")
        print(f"\n  Smallest dipole:  ", end="")
        for d, pdb in sorted(dipole_with_pdb)[:3]:
            print(f"{pdb} ({d:.1f} D)  ", end="")
        print()

    # Buried residue tables
    all_types = sorted(global_total_by_type.keys())
    ionizable = sorted(rt for rt in all_types if rt in IONIZABLE_RES)
    non_ionizable = sorted(rt for rt in all_types if rt not in IONIZABLE_RES)

    _print_burial_table(
        "Ionizable residues  (ARG, ASP, CTR, GLU, HIS, LYS, NTR, TYR)",
        ionizable, global_total_by_type, global_buried_by_type, n_proteins
    )
    _print_burial_table(
        "Non-ionizable residues",
        non_ionizable, global_total_by_type, global_buried_by_type, n_proteins
    )

    grand_total = sum(global_total_by_type.values())
    grand_buried = sum(global_buried_by_type.values())
    grand_pct = 100.0 * grand_buried / grand_total if grand_total > 0 else 0.0
    print(f"\n{'=' * 70}")
    print(f"{'GRAND TOTAL':<12} {grand_buried:>8} / {grand_total:>8}  ({grand_pct:.1f}% buried)")
    print(f"{'=' * 70}")

    # Top proteins
    all_res = set(all_types)
    _print_top_proteins(
        f"Top {top_n} proteins by fraction of buried residues (all)",
        burial_protein_stats, all_res, top_n
    )
    _print_top_proteins(
        f"Top {top_n} proteins by fraction of buried ionizable residues",
        burial_protein_stats, IONIZABLE_RES, top_n
    )

    # pKa shift summary
    print(f"\n{'=' * 70}")
    print("pKa shift summary by residue type")
    print(f"{'=' * 70}")
    print(f"{'Type':<6} {'Count':>6} {'Mean shift':>11} {'Std dev':>9} {'Max |shift|':>12} {'# |shift|>2':>12}")
    print("-" * 60)
    shifts_by_type = defaultdict(list)
    for row in residue_rows:
        try:
            shifts_by_type[row["restype"]].append(float(row["pKa_shift"]))
        except ValueError:
            pass
    for rt in sorted(shifts_by_type.keys()):
        vals = shifts_by_type[rt]
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
        std = var ** 0.5
        mx = max(abs(v) for v in vals)
        nlarge = sum(1 for v in vals if abs(v) > 2.0)
        print(f"{rt:<6} {n:>6} {mean:>+11.3f} {std:>9.3f} {mx:>12.3f} {nlarge:>12}")
    all_shifts = [v for vals in shifts_by_type.values() for v in vals]
    if all_shifts:
        n = len(all_shifts)
        mean = sum(all_shifts) / n
        std = (sum((v - mean) ** 2 for v in all_shifts) / n) ** 0.5
        mx = max(abs(v) for v in all_shifts)
        nlarge = sum(1 for v in all_shifts if abs(v) > 2.0)
        print("-" * 60)
        print(f"{'ALL':<6} {n:>6} {mean:>+11.3f} {std:>9.3f} {mx:>12.3f} {nlarge:>12}")

    sys.stdout = original_stdout
    log_file.close()
    print(f"Saved log to {log_path}")


def _print_burial_table(title, types, global_total, global_buried, n_proteins):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    print(f"{'Residue':<8} {'Buried':>8} {'Total':>8} {'% Buried':>10}  {'Avg buried/protein':>20}")
    print("-" * 58)
    rows = []
    for rt in types:
        total = global_total.get(rt, 0)
        buried = global_buried.get(rt, 0)
        if total == 0:
            continue
        rows.append((rt, buried, total, 100.0 * buried / total))
    rows.sort(key=lambda x: x[3], reverse=True)
    total_all = buried_all = 0
    for rt, buried, total, pct in rows:
        total_all += total
        buried_all += buried
        print(f"{rt:<8} {buried:>8} {total:>8} {pct:>9.1f}%  {buried/n_proteins:>20.2f}")
    pct = 100.0 * buried_all / total_all if total_all > 0 else 0.0
    print("-" * 58)
    print(f"{'TOTAL':<8} {buried_all:>8} {total_all:>8} {pct:>9.1f}%  {buried_all/n_proteins:>20.2f}")


def _print_top_proteins(title, protein_stats, res_filter, n_top):
    print(f"\n{'=' * 70}")
    print(title)
    print(f"{'=' * 70}")
    ranked = []
    for pdb, stats in protein_stats.items():
        total = sum(v for rt, v in stats["total_by_type"].items() if rt in res_filter)
        buried = sum(v for rt, v in stats["buried_by_type"].items() if rt in res_filter)
        if total > 0:
            bt = {rt: v for rt, v in stats["buried_by_type"].items() if rt in res_filter}
            ranked.append((pdb, buried, total, 100.0 * buried / total, bt))
    ranked.sort(key=lambda x: x[3], reverse=True)
    print(f"{'PDB':<8} {'Buried':>8} {'Total':>8} {'% Buried':>10}   Top buried types")
    print("-" * 70)
    for pdb, buried, total, pct, bt in ranked[:n_top]:
        top_types = sorted(bt.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join(f"{rt}:{c}" for rt, c in top_types)
        print(f"{pdb:<8} {buried:>8} {total:>8} {pct:>9.1f}%   {top_str}")


# ── Plotting ─────────────────────────────────────────────────────────────────

def run_plot(outdir, topdir, target_ph, burial_threshold, dpi):
    protein_csv = os.path.join(outdir, PROTEIN_CSV)
    residue_csv = os.path.join(outdir, RESIDUE_CSV)

    missing = [f for f in [protein_csv, residue_csv] if not os.path.isfile(f)]
    if missing:
        print("Error: required data files not found:")
        for f in missing:
            print(f"  {f}")
        print(f"\nRun the analysis first:  python {os.path.basename(__file__)} -o {outdir}")
        sys.exit(1)

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np
    import seaborn as sns
    from matplotlib.patches import Patch

    # ── Load CSVs ────────────────────────────────────────────────────────────
    residue_data = []
    with open(residue_csv) as f:
        for row in csv.DictReader(f):
            try:
                row["pKa_shift_num"] = float(row["pKa_shift"])
            except ValueError:
                row["pKa_shift_num"] = None
            try:
                row["frac_acc_num"] = float(row["frac_acc"])
            except ValueError:
                row["frac_acc_num"] = None
            residue_data.append(row)

    protein_data = []
    with open(protein_csv) as f:
        for row in csv.DictReader(f):
            protein_data.append(row)

    burial_pct = compute_burial(topdir, threshold=burial_threshold)

    sns.set_theme(style="whitegrid", font_scale=0.95)

    colors = sns.color_palette("Set2", len(IONIZABLE_ORDER))
    palette = dict(zip(IONIZABLE_ORDER, colors))
    ion_color = sns.color_palette("Set2")[3]
    non_ion_color = sns.color_palette("Set2")[0]

    # ── Figure 1: Protein charge & dipole analysis ───────────────────────────
    charge_col = f"net_charge_pH{int(target_ph)}"
    has_dipole = any(r.get("full_dipole_D") for r in protein_data)
    if has_dipole:
        fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
        fig2.subplots_adjust(wspace=0.30, hspace=0.35, left=0.07, right=0.97,
                             top=0.94, bottom=0.08)
        axes2 = axes2.flatten()
    else:
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.5))
        fig2.subplots_adjust(wspace=0.32, left=0.05, right=0.97, top=0.90, bottom=0.15)

    # Panel A: pI histogram
    pis = [float(r["pI"]) for r in protein_data if r.get("pI")]
    axes2[0].hist(pis, bins=np.arange(2, 15, 0.5), color="teal",
                  edgecolor="white", linewidth=0.5, alpha=0.85)
    axes2[0].axvline(7.0, color="red", linestyle="--", linewidth=1, alpha=0.7, label="pH 7")
    mean_pi = np.mean(pis)
    axes2[0].axvline(mean_pi, color="orange", linestyle="--", linewidth=1, alpha=0.7,
                     label=f"mean pI = {mean_pi:.1f}")
    axes2[0].set_xlabel("Isoelectric Point (pI)")
    axes2[0].set_ylabel("Number of Proteins")
    axes2[0].set_title("Distribution of Isoelectric Points")
    axes2[0].legend(fontsize=8, framealpha=0.8)
    axes2[0].text(-0.12, 1.05, "A", transform=axes2[0].transAxes,
                  fontsize=16, fontweight="bold", va="top")

    # Panel 2: Net charge histogram
    charges = [float(r[charge_col]) for r in protein_data if r.get(charge_col)]
    charge_bins = np.arange(min(charges) - 2, max(charges) + 2, 2)
    axes2[1].hist([c for c in charges if c >= 0], bins=charge_bins, color="salmon",
                  edgecolor="white", linewidth=0.5, alpha=0.85, label="Positive")
    axes2[1].hist([c for c in charges if c < 0], bins=charge_bins, color="cornflowerblue",
                  edgecolor="white", linewidth=0.5, alpha=0.85, label="Negative")
    axes2[1].axvline(0, color="gray", linestyle="--", linewidth=1)
    mean_crg = np.mean(charges)
    axes2[1].axvline(mean_crg, color="orange", linestyle="--", linewidth=1, alpha=0.7,
                     label=f"mean = {mean_crg:.1f}")
    axes2[1].set_xlabel(f"Net Charge at pH {target_ph:.0f}")
    axes2[1].set_ylabel("Number of Proteins")
    axes2[1].set_title(f"Distribution of Net Charge (pH {target_ph:.0f})")
    axes2[1].legend(fontsize=8, framealpha=0.8)
    axes2[1].text(-0.12, 1.05, "B", transform=axes2[1].transAxes,
                  fontsize=16, fontweight="bold", va="top")

    # Panel 3: pI vs Net Charge scatter
    pi_vals, crg_vals, pdb_labels = [], [], []
    for r in protein_data:
        try:
            pi_vals.append(float(r["pI"]))
            crg_vals.append(float(r[charge_col]))
            pdb_labels.append(r["pdb"])
        except (ValueError, KeyError):
            pass
    scatter_colors = ["salmon" if c > 0 else "cornflowerblue" for c in crg_vals]
    axes2[2].scatter(pi_vals, crg_vals, c=scatter_colors, s=25, alpha=0.7,
                     edgecolors="gray", linewidth=0.3)
    axes2[2].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes2[2].axvline(target_ph, color="red", linestyle=":", linewidth=0.8, alpha=0.6,
                     label=f"pH {target_ph:.0f}")
    axes2[2].set_xlabel("Isoelectric Point (pI)")
    axes2[2].set_ylabel(f"Net Charge at pH {target_ph:.0f}")
    axes2[2].set_title(f"Net Charge vs pI (pH {target_ph:.0f})")
    axes2[2].legend(fontsize=8, framealpha=0.8)
    axes2[2].text(-0.12, 1.05, "C", transform=axes2[2].transAxes,
                  fontsize=16, fontweight="bold", va="top")
    crg_arr = np.array(crg_vals)
    for q in [0.02, 0.98]:
        cutoff = np.quantile(crg_arr, q)
        for pi, crg, lbl in zip(pi_vals, crg_vals, pdb_labels):
            if (q < 0.5 and crg <= cutoff) or (q > 0.5 and crg >= cutoff):
                axes2[2].annotate(lbl, (pi, crg), fontsize=5.5, alpha=0.7,
                                  xytext=(4, 4), textcoords="offset points")

    # Panel 4: Dipole moment histogram (if data available)
    if has_dipole:
        full_dips = [float(r["full_dipole_D"]) for r in protein_data
                     if r.get("full_dipole_D")]
        bb_dips = [float(r["backbone_dipole_D"]) for r in protein_data
                   if r.get("backbone_dipole_D")]
        ion_dips = [float(r["ionizable_dipole_D"]) for r in protein_data
                    if r.get("ionizable_dipole_D")]
        dip_max = max(full_dips) if full_dips else 100
        dip_bins = np.arange(0, dip_max + 20, 20)
        axes2[3].hist(full_dips, bins=dip_bins, color="mediumpurple",
                      edgecolor="white", linewidth=0.5, alpha=0.85, label="Full protein")
        axes2[3].hist(ion_dips, bins=dip_bins, color="salmon",
                      edgecolor="white", linewidth=0.5, alpha=0.6, label="Ionizable")
        axes2[3].hist(bb_dips, bins=dip_bins, color="steelblue",
                      edgecolor="white", linewidth=0.5, alpha=0.6, label="Backbone")
        if full_dips:
            mean_full = np.mean(full_dips)
            axes2[3].axvline(mean_full, color="orange", linestyle="--", linewidth=1,
                             alpha=0.7, label=f"mean full = {mean_full:.0f} D")
        axes2[3].set_xlabel("Dipole Moment (Debye)")
        axes2[3].set_ylabel("Number of Proteins")
        axes2[3].set_title(f"Dipole Moment Distribution (pH {target_ph:.0f})")
        axes2[3].legend(fontsize=8, framealpha=0.8)
        axes2[3].text(-0.12, 1.05, "D", transform=axes2[3].transAxes,
                      fontsize=16, fontweight="bold", va="top")

    out2 = os.path.join(outdir, FIG2_NAME)
    fig2.savefig(out2, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig2)

    # ── Figure 3: pKa shift + energy comparison (2 rows × 3 columns) ─────
    for r in residue_data:
        try:
            r["energy"] = float(r["energy_kcal"])
        except (ValueError, KeyError):
            r["energy"] = None

    fig3 = plt.figure(figsize=(18, 13))

    gs_top = gridspec.GridSpec(2, 3, height_ratios=[3, 1],
                               hspace=0.05, wspace=0.32,
                               left=0.05, right=0.97, top=0.97, bottom=0.55)
    gs_bot = gridspec.GridSpec(2, 2, height_ratios=[3, 1],
                               hspace=0.05, wspace=0.32,
                               left=0.05, right=0.97, top=0.47, bottom=0.05)

    # ── Top row: pKa shift panels ──────────────────────────────────────
    ax_pka_violin = fig3.add_subplot(gs_top[0:2, 0])
    ax_pka_burial = fig3.add_subplot(gs_top[0:2, 1])
    ax_pka_scatter = fig3.add_subplot(gs_top[0, 2])
    ax_pka_fhist = fig3.add_subplot(gs_top[1, 2], sharex=ax_pka_scatter)

    # Top Panel 1: pKa shift violin
    pka_plot_data = [(r["restype"], r["pKa_shift_num"])
                     for r in residue_data if r["pKa_shift_num"] is not None]
    pka_types_present = set(d[0] for d in pka_plot_data)
    pka_order = [rt for rt in IONIZABLE_ORDER if rt in pka_types_present]
    pka_arrays = [np.array([s for t, s in pka_plot_data if t == rt]) for rt in pka_order]

    parts_p = ax_pka_violin.violinplot(pka_arrays, positions=range(len(pka_order)),
                                        showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts_p["bodies"]):
        pc.set_facecolor(palette[pka_order[i]])
        pc.set_alpha(0.7)
    ax_pka_violin.boxplot(pka_arrays, positions=range(len(pka_order)), widths=0.15,
                           patch_artist=True,
                           boxprops=dict(facecolor="white", linewidth=0.8),
                           medianprops=dict(color="black", linewidth=1.2),
                           whiskerprops=dict(linewidth=0.8),
                           capprops=dict(linewidth=0.8),
                           flierprops=dict(marker=".", markersize=2, alpha=0.4))
    ax_pka_violin.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=0)
    ax_pka_violin.set_xticks(range(len(pka_order)))
    ax_pka_violin.set_xticklabels(pka_order)
    ax_pka_violin.set_ylabel("pKa shift  (pKa − pKa$_0$)")
    ax_pka_violin.set_title("pKa Shift Distribution by Ionizable Residue Type")
    ax_pka_violin.text(-0.10, 1.05, "A", transform=ax_pka_violin.transAxes,
                        fontsize=16, fontweight="bold", va="top")
    for i, rt in enumerate(pka_order):
        n = sum(1 for t, _ in pka_plot_data if t == rt)
        ax_pka_violin.text(i, ax_pka_violin.get_ylim()[1] * 0.95, f"n={n}",
                            ha="center", va="top", fontsize=7, color="gray")

    # Top Panel 2: Burial propensity bars
    ion_types3 = sorted([rt for rt in IONIZABLE_ORDER if rt in burial_pct],
                        key=lambda rt: burial_pct.get(rt, 0), reverse=True)
    non_ion_types3 = sorted([rt for rt in NON_IONIZABLE_ORDER if rt in burial_pct],
                            key=lambda rt: burial_pct.get(rt, 0), reverse=True)
    labels3 = ion_types3 + [""] + non_ion_types3
    values3 = ([burial_pct.get(rt, 0) for rt in ion_types3] + [0] +
               [burial_pct.get(rt, 0) for rt in non_ion_types3])
    x3 = np.arange(len(labels3))
    bar_colors3 = ([ion_color] * len(ion_types3) + ["white"] +
                   [non_ion_color] * len(non_ion_types3))
    ax_pka_burial.bar(x3, values3, color=bar_colors3, edgecolor="gray", linewidth=0.5)
    ax_pka_burial.set_xticks(x3)
    ax_pka_burial.set_xticklabels(labels3, rotation=45, ha="right")
    ax_pka_burial.set_ylabel(f"% Buried  (frac. acc. ≤ {burial_threshold})")
    ax_pka_burial.set_title("Burial Propensity: Ionizable vs Non-Ionizable Residues")
    ax_pka_burial.text(-0.10, 1.05, "B", transform=ax_pka_burial.transAxes,
                        fontsize=16, fontweight="bold", va="top")
    ax_pka_burial.legend(
        handles=[Patch(facecolor=ion_color, edgecolor="gray", label="Ionizable"),
                 Patch(facecolor=non_ion_color, edgecolor="gray", label="Non-ionizable")],
        loc="upper left", fontsize=8, framealpha=0.8)
    for i, (lbl, val) in enumerate(zip(labels3, values3)):
        if lbl:
            ax_pka_burial.text(i, val + 1.0, f"{val:.0f}%", ha="center",
                               va="bottom", fontsize=6.5)

    # Top Panel 3: pKa shift vs FSA scatter + histogram
    all_frac_p = []
    for rt in pka_order:
        subset = [(r["frac_acc_num"], r["pKa_shift_num"])
                  for r in residue_data
                  if r["restype"] == rt and r["pKa_shift_num"] is not None
                  and r["frac_acc_num"] is not None]
        if not subset:
            continue
        xs, ys = zip(*subset)
        ax_pka_scatter.scatter(xs, ys, label=rt, alpha=0.5, s=12,
                               color=palette[rt], edgecolors="none")
        all_frac_p.extend(xs)
    ax_pka_scatter.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=0)
    ax_pka_scatter.axvline(burial_threshold, color="red", linestyle=":", linewidth=0.8,
                           alpha=0.6, label=f"burial cutoff ({burial_threshold})")
    ax_pka_scatter.set_ylabel("pKa shift  (pKa − pKa$_0$)")
    ax_pka_scatter.set_title("pKa Shift vs Fractional Solvent Accessibility")
    ax_pka_scatter.text(-0.12, 1.10, "C", transform=ax_pka_scatter.transAxes,
                         fontsize=16, fontweight="bold", va="top")
    ax_pka_scatter.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.8)
    ax_pka_scatter.tick_params(axis='x', labelbottom=False)

    bins_p = np.arange(0, max(all_frac_p) + 0.1, 0.1)
    ax_pka_fhist.hist(all_frac_p, bins=bins_p, color="steelblue",
                       edgecolor="white", linewidth=0.5)
    ax_pka_fhist.axvline(burial_threshold, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_pka_fhist.set_xlabel("Fractional Solvent Accessibility")
    ax_pka_fhist.set_ylabel("Count")
    ax_pka_fhist.set_xlim(ax_pka_scatter.get_xlim())

    # ── Bottom row: Energy panels (2 columns) ────────────────────────
    ax_e_violin = fig3.add_subplot(gs_bot[0:2, 0])
    ax_e_scatter = fig3.add_subplot(gs_bot[0, 1])
    ax_e_fhist = fig3.add_subplot(gs_bot[1, 1], sharex=ax_e_scatter)

    # Bottom Panel 1: Energy violin
    e_plot_data = [(r["restype"], r["energy"])
                   for r in residue_data if r["energy"] is not None]
    e_types_present = set(d[0] for d in e_plot_data)
    e_order = [rt for rt in IONIZABLE_ORDER if rt in e_types_present]
    e_arrays = [np.array([e for t, e in e_plot_data if t == rt]) for rt in e_order]

    parts3 = ax_e_violin.violinplot(e_arrays, positions=range(len(e_order)),
                                     showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts3["bodies"]):
        pc.set_facecolor(palette[e_order[i]])
        pc.set_alpha(0.7)
    ax_e_violin.boxplot(e_arrays, positions=range(len(e_order)), widths=0.15,
                         patch_artist=True,
                         boxprops=dict(facecolor="white", linewidth=0.8),
                         medianprops=dict(color="black", linewidth=1.2),
                         whiskerprops=dict(linewidth=0.8),
                         capprops=dict(linewidth=0.8),
                         flierprops=dict(marker=".", markersize=2, alpha=0.4))
    ax_e_violin.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=0)
    e_ylim = ax_e_violin.get_ylim()
    ax_e_violin.axhspan(0, e_ylim[1], color="red", alpha=0.06, zorder=0)
    ax_e_violin.axhspan(e_ylim[0], 0, color="green", alpha=0.06, zorder=0)
    ax_e_violin.annotate("↑ Destabilized", xy=(0.02, 0.95), xycoords="axes fraction",
                          ha="left", va="top", fontsize=11, fontweight="bold", color="firebrick")
    ax_e_violin.annotate("↓ Stabilized", xy=(0.02, 0.05), xycoords="axes fraction",
                          ha="left", va="bottom", fontsize=11, fontweight="bold", color="darkgreen")
    ax_e_violin.set_ylim(e_ylim)
    ax_e_violin.set_xticks(range(len(e_order)))
    ax_e_violin.set_xticklabels(e_order)
    ax_e_violin.set_ylabel("ΔΔG (kcal/mol)")
    ax_e_violin.set_title("pKa Shift Energy by Ionizable Residue Type")
    ax_e_violin.text(-0.10, 1.05, "D", transform=ax_e_violin.transAxes,
                      fontsize=16, fontweight="bold", va="top")
    for i, rt in enumerate(e_order):
        n = sum(1 for t, _ in e_plot_data if t == rt)
        ax_e_violin.text(i, e_ylim[1] * 0.95, f"n={n}",
                          ha="center", va="top", fontsize=7, color="gray")

    # Bottom Panel 2: Energy vs FSA scatter + histogram
    all_frac_e = []
    for rt in e_order:
        subset = [(r["frac_acc_num"], r["energy"])
                  for r in residue_data
                  if r["restype"] == rt and r["energy"] is not None
                  and r["frac_acc_num"] is not None]
        if not subset:
            continue
        xs, ys = zip(*subset)
        ax_e_scatter.scatter(xs, ys, label=rt, alpha=0.5, s=12,
                             color=palette[rt], edgecolors="none")
        all_frac_e.extend(xs)
    ax_e_scatter.axhline(0, color="gray", linestyle="--", linewidth=0.8, zorder=0)
    es_ylim = ax_e_scatter.get_ylim()
    ax_e_scatter.axhspan(0, es_ylim[1], color="red", alpha=0.06, zorder=0)
    ax_e_scatter.axhspan(es_ylim[0], 0, color="green", alpha=0.06, zorder=0)
    ax_e_scatter.annotate("↑ Destabilized", xy=(0.02, 0.95),
                           xycoords="axes fraction", ha="left", va="top",
                           fontsize=11, fontweight="bold", color="firebrick")
    ax_e_scatter.annotate("↓ Stabilized", xy=(0.02, 0.05),
                           xycoords="axes fraction", ha="left", va="bottom",
                           fontsize=11, fontweight="bold", color="darkgreen")
    ax_e_scatter.set_ylim(es_ylim)
    ax_e_scatter.axvline(burial_threshold, color="red", linestyle=":", linewidth=0.8,
                         alpha=0.6, label=f"burial cutoff ({burial_threshold})")
    ax_e_scatter.set_ylabel("ΔΔG (kcal/mol)")
    ax_e_scatter.set_title("pKa Shift Energy vs Fractional Solvent Accessibility")
    ax_e_scatter.text(-0.12, 1.10, "E", transform=ax_e_scatter.transAxes,
                       fontsize=16, fontweight="bold", va="top")
    ax_e_scatter.legend(fontsize=7, ncol=2, loc="upper right", framealpha=0.8)
    ax_e_scatter.tick_params(axis='x', labelbottom=False)

    e_frac_bins = np.arange(0, max(all_frac_e) + 0.1, 0.1)
    ax_e_fhist.hist(all_frac_e, bins=e_frac_bins, color="steelblue",
                     edgecolor="white", linewidth=0.5)
    ax_e_fhist.axvline(burial_threshold, color="red", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_e_fhist.set_xlabel("Fractional Solvent Accessibility")
    ax_e_fhist.set_ylabel("Count")
    ax_e_fhist.set_xlim(ax_e_scatter.get_xlim())

    out3 = os.path.join(outdir, FIG3_NAME)
    fig3.savefig(out3, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out3}")
    plt.close(fig3)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyze MCCE output: buried residues, feature extraction, and plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python analyze_mcce_proteins.py                   # full analysis\n"
            "  python analyze_mcce_proteins.py --plot             # plot from existing CSVs\n"
            "  python analyze_mcce_proteins.py --ph 4.0           # analyze at pH 4\n"
            "  python analyze_mcce_proteins.py -o results --plot  # analyze + plot, save to results/\n"
        ),
    )
    parser.add_argument(
        "-d", "--directory", default=".",
        help="Top-level directory containing PDB subdirectories; default: %(default)s"
    )
    parser.add_argument(
        "-o", "--outdir", default="analysis_mcce_proteins",
        help="Output directory for CSVs and figures; default: %(default)s"
    )
    parser.add_argument(
        "--ph", type=float, default=7.0,
        help="pH for net charge extraction; default: %(default)s"
    )
    parser.add_argument(
        "-b", "--burial-threshold", type=float, default=0.20,
        help="Fractional accessibility threshold for buried; default: %(default)s"
    )
    parser.add_argument(
        "-n", "--top-proteins", type=int, default=10,
        help="Number of top proteins to show in ranking; default: %(default)s"
    )
    parser.add_argument(
        "--include-ntg", action="store_true", default=False,
        help="Include NTG (N-terminal, non-ionizable) entries; default: %(default)s"
    )
    parser.add_argument(
        "--all-residues", action="store_true", default=False,
        help="Include non-standard residues (ligands, metals, solvents); default: %(default)s"
    )
    parser.add_argument(
        "--plot-only", action="store_true", default=False,
        help="Skip analysis, only regenerate figures from existing CSVs; default: %(default)s"
    )
    parser.add_argument(
        "--no-plot", action="store_true", default=False,
        help="Run analysis only, skip figure generation; default: %(default)s"
    )
    parser.add_argument(
        "--dpi", type=int, default=200,
        help="Figure DPI; default: %(default)s"
    )
    args = parser.parse_args()

    if not args.plot_only:
        run_analysis(args.directory, args.outdir, args.ph, args.burial_threshold,
                     args.top_proteins, args.include_ntg, args.all_residues)
    if not args.no_plot:
        run_plot(args.outdir, args.directory, args.ph, args.burial_threshold, args.dpi)


if __name__ == "__main__":
    main()
