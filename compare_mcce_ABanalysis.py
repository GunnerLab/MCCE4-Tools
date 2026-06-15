#!/usr/bin/env python3
"""
Comparative MCCE analysis: complex vs antibody-alone vs antigen-alone.

Reads the protein_features.csv and residue_features.csv produced by
analyze_mcce_proteins.py in each of the three directories and generates:
  - A merged comparison CSV (protein-level and residue-level)
  - Comparative figures highlighting binding-induced changes

Usage:
    python compare_mcce_analysis.py
    python compare_mcce_ABanalysis.py --ph 7.0 -o AB_mcce_comparative_analysis
    python compare_mcce_analysis.py --plot-only
"""
import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ── Defaults ────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
COMPLEX_DIR = os.path.join(BASE, "mcce4_complex", "analysis_mcce_proteins")
ANTIBODY_DIR = os.path.join(BASE, "mcce4_antibody", "analysis_mcce_proteins")
ANTIGEN_DIR = os.path.join(BASE, "mcce4_antigen", "analysis_mcce_proteins")

PROTEIN_CSV = "protein_features.csv"
RESIDUE_CSV = "residue_features.csv"

IONIZABLE_ORDER = ["ASP", "GLU", "CYS", "HIS", "TYR", "LYS", "ARG"]


# ── Key extraction ──────────────────────────────────────────────────────────

def _extract_key_complex(name):
    """SEQ1_EGFR_COMPLEX -> SEQ1_EGFR"""
    return re.sub(r"_COMPLEX$", "", name)


def _extract_key_antibody(name):
    """ANTIBODY_HL_SEQ1_EGFR -> SEQ1_EGFR"""
    return re.sub(r"^ANTIBODY_HL_", "", name)


def _extract_key_antigen(name):
    """ANTIGEN_A_SEQ1_EGFR -> SEQ1_EGFR; ANTIGENS_SEQ1_EGFR_TROP2 -> SEQ1_EGFR_TROP2"""
    name = re.sub(r"^ANTIGENS_", "", name)
    name = re.sub(r"^ANTIGEN_A_", "", name)
    return name


# ── Loaders ─────────────────────────────────────────────────────────────────

def load_protein_csv(path):
    rows = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["pdb"]] = row
    return rows


def load_residue_csv(path):
    rows = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            rows[row["pdb"]].append(row)
    return rows


def _safe_float(val):
    if val is None or val == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


# ── Analysis ────────────────────────────────────────────────────────────────

def run_comparison(complex_dir, antibody_dir, antigen_dir, outdir, target_ph):
    os.makedirs(outdir, exist_ok=True)
    charge_col = f"net_charge_pH{int(target_ph)}"

    cpx_prot = load_protein_csv(os.path.join(complex_dir, PROTEIN_CSV))
    ab_prot = load_protein_csv(os.path.join(antibody_dir, PROTEIN_CSV))
    ag_prot = load_protein_csv(os.path.join(antigen_dir, PROTEIN_CSV))

    cpx_res = load_residue_csv(os.path.join(complex_dir, RESIDUE_CSV))
    ab_res = load_residue_csv(os.path.join(antibody_dir, RESIDUE_CSV))
    ag_res = load_residue_csv(os.path.join(antigen_dir, RESIDUE_CSV))

    cpx_keys = {_extract_key_complex(k): k for k in cpx_prot}
    ab_keys = {_extract_key_antibody(k): k for k in ab_prot}
    ag_keys = {_extract_key_antigen(k): k for k in ag_prot}

    matched_keys = sorted(set(cpx_keys) & set(ab_keys) & set(ag_keys))
    print(f"Matched triplets: {len(matched_keys)} / "
          f"(complex={len(cpx_keys)}, antibody={len(ab_keys)}, antigen={len(ag_keys)})")

    if not matched_keys:
        print("ERROR: No matched triplets found. Check directory paths and naming.")
        sys.exit(1)

    # ── Protein-level comparison CSV ────────────────────────────────────
    protein_comp_rows = []
    for key in matched_keys:
        c = cpx_prot[cpx_keys[key]]
        a = ab_prot[ab_keys[key]]
        g = ag_prot[ag_keys[key]]

        c_sas = _safe_float(c["total_sas"])
        a_sas = _safe_float(a["total_sas"])
        g_sas = _safe_float(g["total_sas"])
        bsa = None
        if all(v is not None for v in [c_sas, a_sas, g_sas]):
            bsa = (a_sas + g_sas) - c_sas

        c_crg = _safe_float(c.get(charge_col))
        a_crg = _safe_float(a.get(charge_col))
        g_crg = _safe_float(g.get(charge_col))
        charge_add = None
        charge_delta = None
        if a_crg is not None and g_crg is not None:
            charge_add = a_crg + g_crg
        if c_crg is not None and charge_add is not None:
            charge_delta = c_crg - charge_add

        c_pi = _safe_float(c["pI"])
        a_pi = _safe_float(a["pI"])
        g_pi = _safe_float(g["pI"])

        c_fb = _safe_float(c["frac_buried"])
        a_fb = _safe_float(a["frac_buried"])
        g_fb = _safe_float(g["frac_buried"])

        c_fib = _safe_float(c["frac_ionizable_buried"])
        a_fib = _safe_float(a["frac_ionizable_buried"])
        g_fib = _safe_float(g["frac_ionizable_buried"])

        c_ms = _safe_float(c["mean_pKa_shift"])
        a_ms = _safe_float(a["mean_pKa_shift"])
        g_ms = _safe_float(g["mean_pKa_shift"])

        c_mx = _safe_float(c["max_abs_pKa_shift"])
        a_mx = _safe_float(a["max_abs_pKa_shift"])
        g_mx = _safe_float(g["max_abs_pKa_shift"])

        def _fmt(v, f=".2f"):
            return format(v, f) if v is not None else ""

        target_type = key.split("_", 1)[1] if "_" in key else key

        protein_comp_rows.append({
            "key": key,
            "target": target_type,
            "complex_sas": _fmt(c_sas),
            "antibody_sas": _fmt(a_sas),
            "antigen_sas": _fmt(g_sas),
            "buried_surface_area": _fmt(bsa),
            f"complex_{charge_col}": _fmt(c_crg),
            f"antibody_{charge_col}": _fmt(a_crg),
            f"antigen_{charge_col}": _fmt(g_crg),
            "charge_additivity_delta": _fmt(charge_delta, ".3f"),
            "complex_pI": _fmt(c_pi),
            "antibody_pI": _fmt(a_pi),
            "antigen_pI": _fmt(g_pi),
            "complex_frac_buried": _fmt(c_fb, ".3f"),
            "antibody_frac_buried": _fmt(a_fb, ".3f"),
            "antigen_frac_buried": _fmt(g_fb, ".3f"),
            "complex_frac_ion_buried": _fmt(c_fib, ".3f"),
            "antibody_frac_ion_buried": _fmt(a_fib, ".3f"),
            "antigen_frac_ion_buried": _fmt(g_fib, ".3f"),
            "complex_mean_pKa_shift": _fmt(c_ms, ".3f"),
            "antibody_mean_pKa_shift": _fmt(a_ms, ".3f"),
            "antigen_mean_pKa_shift": _fmt(g_ms, ".3f"),
            "complex_max_abs_pKa_shift": _fmt(c_mx, ".3f"),
            "antibody_max_abs_pKa_shift": _fmt(a_mx, ".3f"),
            "antigen_max_abs_pKa_shift": _fmt(g_mx, ".3f"),
            "complex_n_large_shift": c["n_large_shift_gt2"],
            "antibody_n_large_shift": a["n_large_shift_gt2"],
            "antigen_n_large_shift": g["n_large_shift_gt2"],
        })

    prot_path = os.path.join(outdir, "protein_comparison.csv")
    with open(prot_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(protein_comp_rows[0].keys()))
        w.writeheader()
        w.writerows(protein_comp_rows)
    print(f"Wrote {prot_path}")

    # ── Residue-level comparison CSV ────────────────────────────────────
    residue_comp_rows = []
    interface_residues = []

    for key in matched_keys:
        cpx_name = cpx_keys[key]
        ab_name = ab_keys[key]
        ag_name = ag_keys[key]

        cpx_by_id = {(r["restype"], r["resid"]): r for r in cpx_res.get(cpx_name, [])}

        for source_label, source_name, source_rows in [
            ("antibody", ab_name, ab_res.get(ab_name, [])),
            ("antigen", ag_name, ag_res.get(ag_name, [])),
        ]:
            for iso_row in source_rows:
                rt = iso_row["restype"]
                rid = iso_row["resid"]
                cpx_row = cpx_by_id.get((rt, rid))
                if cpx_row is None:
                    continue

                iso_shift = _safe_float(iso_row["pKa_shift"])
                cpx_shift = _safe_float(cpx_row["pKa_shift"])
                delta_shift = None
                if iso_shift is not None and cpx_shift is not None:
                    delta_shift = cpx_shift - iso_shift

                iso_frac = _safe_float(iso_row["frac_acc"])
                cpx_frac = _safe_float(cpx_row["frac_acc"])
                delta_frac = None
                if iso_frac is not None and cpx_frac is not None:
                    delta_frac = cpx_frac - iso_frac

                iso_buried = _safe_float(iso_row["buried"])
                cpx_buried = _safe_float(cpx_row["buried"])
                newly_buried = ""
                if iso_buried is not None and cpx_buried is not None:
                    newly_buried = 1 if (iso_buried == 0 and cpx_buried == 1) else 0

                iso_energy = _safe_float(iso_row["energy_kcal"])
                cpx_energy = _safe_float(cpx_row["energy_kcal"])
                delta_energy = None
                if iso_energy is not None and cpx_energy is not None:
                    delta_energy = cpx_energy - iso_energy

                def _fmt(v, f=".3f"):
                    return format(v, f) if v is not None else ""

                row = {
                    "key": key,
                    "component": source_label,
                    "restype": rt,
                    "resid": rid,
                    "isolated_pKa": iso_row["pKa"],
                    "complex_pKa": cpx_row["pKa"],
                    "isolated_pKa_shift": iso_row["pKa_shift"],
                    "complex_pKa_shift": cpx_row["pKa_shift"],
                    "delta_pKa_shift": _fmt(delta_shift),
                    "isolated_frac_acc": iso_row["frac_acc"],
                    "complex_frac_acc": cpx_row["frac_acc"],
                    "delta_frac_acc": _fmt(delta_frac),
                    "isolated_buried": iso_row["buried"],
                    "complex_buried": cpx_row["buried"],
                    "newly_buried": newly_buried,
                    "isolated_energy_kcal": iso_row["energy_kcal"],
                    "complex_energy_kcal": cpx_row["energy_kcal"],
                    "delta_energy_kcal": _fmt(delta_energy),
                }
                residue_comp_rows.append(row)

                if newly_buried == 1:
                    interface_residues.append(row)

    res_path = os.path.join(outdir, "residue_comparison.csv")
    with open(res_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(residue_comp_rows[0].keys()))
        w.writeheader()
        w.writerows(residue_comp_rows)
    print(f"Wrote {res_path}  ({len(residue_comp_rows)} matched residues)")

    # ── Summary report ──────────────────────────────────────────────────
    log_path = os.path.join(outdir, "comparison_summary.log")
    with open(log_path, "w") as log:
        _tee = lambda s: (print(s), log.write(s + "\n"))

        _tee(f"{'=' * 75}")
        _tee(f"COMPARATIVE MCCE ANALYSIS: Complex vs Antibody vs Antigen  (pH {target_ph})")
        _tee(f"Matched triplets: {len(matched_keys)}")
        _tee(f"{'=' * 75}")

        # BSA summary
        bsa_vals = [_safe_float(r["buried_surface_area"]) for r in protein_comp_rows]
        bsa_vals = [v for v in bsa_vals if v is not None]
        if bsa_vals:
            _tee(f"\n--- Buried Surface Area upon binding (Ab + Ag SAS - Complex SAS) ---")
            _tee(f"  mean = {sum(bsa_vals)/len(bsa_vals):.1f} A^2")
            _tee(f"  min  = {min(bsa_vals):.1f}   max = {max(bsa_vals):.1f}")

        # Charge additivity
        cdelta = [_safe_float(r["charge_additivity_delta"]) for r in protein_comp_rows]
        cdelta = [v for v in cdelta if v is not None]
        if cdelta:
            _tee(f"\n--- Charge additivity (complex - [Ab + Ag]) ---")
            _tee(f"  mean delta = {sum(cdelta)/len(cdelta):.3f}")
            _tee(f"  max |delta| = {max(abs(v) for v in cdelta):.3f}")
            if all(abs(v) < 1.0 for v in cdelta):
                _tee(f"  Charge is approximately additive for all systems.")

        # pI comparison
        _tee(f"\n--- Isoelectric Point (pI) ---")
        _tee(f"  {'System':<20} {'mean pI':>8}  {'min':>6}  {'max':>6}")
        _tee(f"  {'-'*50}")
        for label, col in [("Complex", "complex_pI"), ("Antibody", "antibody_pI"),
                           ("Antigen", "antigen_pI")]:
            vals = [_safe_float(r[col]) for r in protein_comp_rows]
            vals = [v for v in vals if v is not None]
            if vals:
                _tee(f"  {label:<20} {sum(vals)/len(vals):>8.2f}  {min(vals):>6.2f}  {max(vals):>6.2f}")

        # Burial comparison
        _tee(f"\n--- Fraction buried (all residues) ---")
        _tee(f"  {'System':<20} {'mean':>8}  {'min':>6}  {'max':>6}")
        _tee(f"  {'-'*50}")
        for label, col in [("Complex", "complex_frac_buried"),
                           ("Antibody", "antibody_frac_buried"),
                           ("Antigen", "antigen_frac_buried")]:
            vals = [_safe_float(r[col]) for r in protein_comp_rows]
            vals = [v for v in vals if v is not None]
            if vals:
                _tee(f"  {label:<20} {sum(vals)/len(vals):>8.3f}  {min(vals):>6.3f}  {max(vals):>6.3f}")

        _tee(f"\n--- Fraction ionizable buried ---")
        _tee(f"  {'System':<20} {'mean':>8}  {'min':>6}  {'max':>6}")
        _tee(f"  {'-'*50}")
        for label, col in [("Complex", "complex_frac_ion_buried"),
                           ("Antibody", "antibody_frac_ion_buried"),
                           ("Antigen", "antigen_frac_ion_buried")]:
            vals = [_safe_float(r[col]) for r in protein_comp_rows]
            vals = [v for v in vals if v is not None]
            if vals:
                _tee(f"  {label:<20} {sum(vals)/len(vals):>8.3f}  {min(vals):>6.3f}  {max(vals):>6.3f}")

        # pKa shift comparison
        _tee(f"\n--- Mean |pKa shift| by residue type ---")
        _tee(f"  {'Type':<6} {'Complex':>10} {'Antibody':>10} {'Antigen':>10}  {'Delta(C-iso)':>12}")
        _tee(f"  {'-'*55}")
        for rt in IONIZABLE_ORDER:
            cpx_shifts = []
            iso_shifts = []
            for r in residue_comp_rows:
                if r["restype"] != rt:
                    continue
                cs = _safe_float(r["complex_pKa_shift"])
                iss = _safe_float(r["isolated_pKa_shift"])
                if cs is not None:
                    cpx_shifts.append(abs(cs))
                if iss is not None:
                    iso_shifts.append(abs(iss))
            if not cpx_shifts:
                continue
            ab_shifts = [abs(_safe_float(r["isolated_pKa_shift"])) for r in residue_comp_rows
                         if r["restype"] == rt and r["component"] == "antibody"
                         and _safe_float(r["isolated_pKa_shift"]) is not None]
            ag_shifts = [abs(_safe_float(r["isolated_pKa_shift"])) for r in residue_comp_rows
                         if r["restype"] == rt and r["component"] == "antigen"
                         and _safe_float(r["isolated_pKa_shift"]) is not None]
            c_mean = sum(cpx_shifts)/len(cpx_shifts) if cpx_shifts else 0
            a_mean = sum(ab_shifts)/len(ab_shifts) if ab_shifts else 0
            g_mean = sum(ag_shifts)/len(ag_shifts) if ag_shifts else 0
            i_mean = sum(iso_shifts)/len(iso_shifts) if iso_shifts else 0
            delta = c_mean - i_mean
            _tee(f"  {rt:<6} {c_mean:>10.3f} {a_mean:>10.3f} {g_mean:>10.3f}  {delta:>+12.3f}")

        # Interface residues
        _tee(f"\n--- Interface residues (exposed isolated -> buried in complex) ---")
        _tee(f"  Total newly buried ionizable residues: {len(interface_residues)}")
        if interface_residues:
            by_type = defaultdict(int)
            by_component = defaultdict(int)
            large_shift = []
            for r in interface_residues:
                by_type[r["restype"]] += 1
                by_component[r["component"]] += 1
                ds = _safe_float(r["delta_pKa_shift"])
                if ds is not None and abs(ds) > 1.0:
                    large_shift.append(r)
            _tee(f"  By component: " + ", ".join(f"{k}={v}" for k, v in sorted(by_component.items())))
            _tee(f"  By type:      " + ", ".join(f"{k}={v}" for k, v in sorted(by_type.items())))
            _tee(f"  With |delta pKa shift| > 1.0: {len(large_shift)}")
            if large_shift:
                _tee(f"\n  {'Key':<25} {'Comp':<9} {'Res':<10} {'iso_shift':>10} {'cpx_shift':>10} {'delta':>8}")
                _tee(f"  {'-'*75}")
                large_shift.sort(key=lambda r: abs(_safe_float(r["delta_pKa_shift"]) or 0), reverse=True)
                for r in large_shift[:20]:
                    _tee(f"  {r['key']:<25} {r['component']:<9} "
                         f"{r['restype']}{r['resid']:<7} "
                         f"{r['isolated_pKa_shift']:>10} {r['complex_pKa_shift']:>10} "
                         f"{r['delta_pKa_shift']:>8}")

        _tee(f"\n{'=' * 75}")

    print(f"Wrote {log_path}")
    return outdir


# ── Plotting ────────────────────────────────────────────────────────────────

def run_plot(outdir, target_ph, dpi):
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.patches import Patch

    prot_csv = os.path.join(outdir, "protein_comparison.csv")
    res_csv = os.path.join(outdir, "residue_comparison.csv")
    for f in [prot_csv, res_csv]:
        if not os.path.isfile(f):
            print(f"ERROR: {f} not found. Run analysis first.")
            sys.exit(1)

    prot_data = []
    with open(prot_csv) as f:
        for row in csv.DictReader(f):
            prot_data.append(row)

    res_data = []
    with open(res_csv) as f:
        for row in csv.DictReader(f):
            row["delta_pKa_shift_num"] = _safe_float(row["delta_pKa_shift"])
            row["delta_frac_acc_num"] = _safe_float(row["delta_frac_acc"])
            row["delta_energy_num"] = _safe_float(row["delta_energy_kcal"])
            row["iso_shift_num"] = _safe_float(row["isolated_pKa_shift"])
            row["cpx_shift_num"] = _safe_float(row["complex_pKa_shift"])
            row["newly_buried_num"] = _safe_float(row["newly_buried"])
            res_data.append(row)

    charge_col = f"net_charge_pH{int(target_ph)}"
    sns.set_theme(style="whitegrid", font_scale=0.95)
    colors_3 = ["#4C72B0", "#DD8452", "#55A868"]
    labels_3 = ["Complex", "Antibody", "Antigen"]

    # ================================================================
    # FIGURE 1: Protein-level comparison (2x3 grid)
    # ================================================================
    fig1, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig1.suptitle(f"Protein-Level Comparison: Complex vs Antibody vs Antigen  (pH {target_ph:.0f})",
                  fontsize=14, fontweight="bold", y=0.98)
    fig1.subplots_adjust(hspace=0.35, wspace=0.30, left=0.06, right=0.97, top=0.93, bottom=0.08)
    panel_labels = "ABCDEF"

    # Panel A: BSA bar chart
    ax = axes[0, 0]
    keys = [r["key"] for r in prot_data]
    bsa_vals = [_safe_float(r["buried_surface_area"]) for r in prot_data]

    target_types = sorted(set(r["target"] for r in prot_data))
    target_colors = dict(zip(target_types, sns.color_palette("Set2", len(target_types))))
    bar_colors = [target_colors[r["target"]] for r in prot_data]

    x = np.arange(len(keys))
    ax.bar(x, bsa_vals, color=bar_colors, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([k.split("_", 1)[0] for k in keys], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Buried Surface Area ($\\AA^2$)")
    ax.set_title("Buried Surface Area upon Binding")
    ax.axhline(np.mean([v for v in bsa_vals if v]), color="red", ls="--", lw=0.8,
               label=f"mean={np.mean([v for v in bsa_vals if v]):.0f}")
    ax.legend(fontsize=7)
    handles = [Patch(facecolor=target_colors[t], edgecolor="gray", label=t) for t in target_types]
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=handles, fontsize=7, loc="upper right", title="Target", title_fontsize=7)
    ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel B: pI comparison
    ax = axes[0, 1]
    for i, (label, col, color) in enumerate(zip(labels_3,
            ["complex_pI", "antibody_pI", "antigen_pI"], colors_3)):
        vals = [_safe_float(r[col]) for r in prot_data]
        ax.hist(vals, bins=np.arange(3, 13, 0.5), alpha=0.5, color=color,
                edgecolor="white", linewidth=0.5, label=label)
    ax.axvline(7.0, color="red", ls="--", lw=0.8, label="pH 7")
    ax.set_xlabel("Isoelectric Point (pI)")
    ax.set_ylabel("Count")
    ax.set_title("pI Distribution by System")
    ax.legend(fontsize=7)
    ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel C: Net charge comparison
    ax = axes[0, 2]
    for i, (label, col, color) in enumerate(zip(labels_3,
            [f"complex_{charge_col}", f"antibody_{charge_col}", f"antigen_{charge_col}"],
            colors_3)):
        vals = [_safe_float(r[col]) for r in prot_data if _safe_float(r[col]) is not None]
        if vals:
            ax.hist(vals, bins=15, alpha=0.5, color=color,
                    edgecolor="white", linewidth=0.5, label=label)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel(f"Net Charge (pH {target_ph:.0f})")
    ax.set_ylabel("Count")
    ax.set_title("Net Charge Distribution by System")
    ax.legend(fontsize=7)
    ax.text(-0.12, 1.05, "C", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel D: Fraction buried comparison
    ax = axes[1, 0]
    width = 0.25
    x = np.arange(len(prot_data))
    for i, (label, col, color) in enumerate(zip(labels_3,
            ["complex_frac_buried", "antibody_frac_buried", "antigen_frac_buried"],
            colors_3)):
        vals = [_safe_float(r[col]) or 0 for r in prot_data]
        ax.bar(x + i * width, vals, width, color=color, edgecolor="gray",
               linewidth=0.3, label=label)
    ax.set_xticks(x + width)
    ax.set_xticklabels([k.split("_", 1)[0] for k in keys], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Fraction Buried")
    ax.set_title("Residue Burial: Complex vs Isolated Components")
    ax.legend(fontsize=7)
    ax.text(-0.12, 1.05, "D", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel E: Mean pKa shift comparison
    ax = axes[1, 1]
    for i, (label, col, color) in enumerate(zip(labels_3,
            ["complex_mean_pKa_shift", "antibody_mean_pKa_shift", "antigen_mean_pKa_shift"],
            colors_3)):
        vals = [_safe_float(r[col]) for r in prot_data]
        vals = [v for v in vals if v is not None]
        if vals:
            ax.hist(vals, bins=15, alpha=0.5, color=color,
                    edgecolor="white", linewidth=0.5, label=label)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Mean pKa Shift")
    ax.set_ylabel("Count")
    ax.set_title("Mean pKa Shift Distribution by System")
    ax.legend(fontsize=7)
    ax.text(-0.12, 1.05, "E", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel F: Charge additivity
    ax = axes[1, 2]
    cdeltas = [_safe_float(r["charge_additivity_delta"]) for r in prot_data]
    cdeltas = [v for v in cdeltas if v is not None]
    ax.hist(cdeltas, bins=15, color="mediumpurple", edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="red", ls="--", lw=1)
    ax.set_xlabel("Complex charge - (Ab + Ag charge)")
    ax.set_ylabel("Count")
    ax.set_title("Charge Additivity Check")
    mean_cd = np.mean(cdeltas) if cdeltas else 0
    ax.axvline(mean_cd, color="orange", ls="--", lw=0.8,
               label=f"mean={mean_cd:.2f}")
    ax.legend(fontsize=7)
    ax.text(-0.12, 1.05, "F", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    out1 = os.path.join(outdir, "protein_comparison.png")
    fig1.savefig(out1, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out1}")
    plt.close(fig1)

    # ================================================================
    # FIGURE 2: Residue-level comparison (2x3 grid)
    # ================================================================
    fig2, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig2.suptitle("Residue-Level Comparison: Binding-Induced Changes in Ionizable Residues",
                  fontsize=14, fontweight="bold", y=0.98)
    fig2.subplots_adjust(hspace=0.35, wspace=0.30, left=0.06, right=0.97, top=0.93, bottom=0.08)

    palette = dict(zip(IONIZABLE_ORDER, sns.color_palette("Set2", len(IONIZABLE_ORDER))))

    # Panel A: Isolated vs Complex pKa shift scatter
    ax = axes[0, 0]
    for rt in IONIZABLE_ORDER:
        subset = [(r["iso_shift_num"], r["cpx_shift_num"])
                  for r in res_data
                  if r["restype"] == rt and r["iso_shift_num"] is not None
                  and r["cpx_shift_num"] is not None]
        if not subset:
            continue
        xs, ys = zip(*subset)
        ax.scatter(xs, ys, label=rt, alpha=0.4, s=10, color=palette[rt], edgecolors="none")
    lim = ax.get_xlim()
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5, label="y=x")
    ax.set_xlabel("Isolated pKa Shift")
    ax.set_ylabel("Complex pKa Shift")
    ax.set_title("pKa Shift: Complex vs Isolated")
    ax.legend(fontsize=6, ncol=2, loc="upper left")
    ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel B: Delta pKa shift violin by residue type
    ax = axes[0, 1]
    delta_plot = [(r["restype"], r["delta_pKa_shift_num"])
                  for r in res_data if r["delta_pKa_shift_num"] is not None]
    types_present = set(d[0] for d in delta_plot)
    d_order = [rt for rt in IONIZABLE_ORDER if rt in types_present]
    d_arrays = [np.array([s for t, s in delta_plot if t == rt]) for rt in d_order]

    if d_arrays and any(len(a) > 0 for a in d_arrays):
        parts = ax.violinplot([a for a in d_arrays if len(a) > 0],
                              positions=[i for i, a in enumerate(d_arrays) if len(a) > 0],
                              showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            valid_order = [rt for rt, a in zip(d_order, d_arrays) if len(a) > 0]
            pc.set_facecolor(palette[valid_order[i]])
            pc.set_alpha(0.7)
        valid_arrays = [a for a in d_arrays if len(a) > 0]
        valid_pos = [i for i, a in enumerate(d_arrays) if len(a) > 0]
        ax.boxplot(valid_arrays, positions=valid_pos, widths=0.15,
                   patch_artist=True,
                   boxprops=dict(facecolor="white", linewidth=0.8),
                   medianprops=dict(color="black", linewidth=1.2),
                   whiskerprops=dict(linewidth=0.8),
                   capprops=dict(linewidth=0.8),
                   flierprops=dict(marker=".", markersize=2, alpha=0.4))
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xticks(range(len(d_order)))
    ax.set_xticklabels(d_order)
    ax.set_ylabel("Delta pKa Shift (complex - isolated)")
    ax.set_title("Binding-Induced pKa Shift Change by Type")
    ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel C: Delta pKa shift by antibody vs antigen
    ax = axes[0, 2]
    for comp_label, color in [("antibody", colors_3[1]), ("antigen", colors_3[2])]:
        vals = [r["delta_pKa_shift_num"] for r in res_data
                if r["component"] == comp_label and r["delta_pKa_shift_num"] is not None]
        if vals:
            ax.hist(vals, bins=np.arange(-6, 6.5, 0.5), alpha=0.5, color=color,
                    edgecolor="white", linewidth=0.5, label=comp_label.capitalize())
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Delta pKa Shift (complex - isolated)")
    ax.set_ylabel("Count")
    ax.set_title("Binding-Induced pKa Shift: Antibody vs Antigen Side")
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.05, "C", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel D: Delta energy violin by type
    ax = axes[1, 0]
    e_plot = [(r["restype"], r["delta_energy_num"])
              for r in res_data if r["delta_energy_num"] is not None]
    e_types = set(d[0] for d in e_plot)
    e_order = [rt for rt in IONIZABLE_ORDER if rt in e_types]
    e_arrays = [np.array([e for t, e in e_plot if t == rt]) for rt in e_order]

    if e_arrays and any(len(a) > 0 for a in e_arrays):
        valid_arrays = [a for a in e_arrays if len(a) > 0]
        valid_pos = [i for i, a in enumerate(e_arrays) if len(a) > 0]
        valid_order = [rt for rt, a in zip(e_order, e_arrays) if len(a) > 0]
        parts = ax.violinplot(valid_arrays, positions=valid_pos,
                              showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(palette[valid_order[i]])
            pc.set_alpha(0.7)
        ax.boxplot(valid_arrays, positions=valid_pos, widths=0.15,
                   patch_artist=True,
                   boxprops=dict(facecolor="white", linewidth=0.8),
                   medianprops=dict(color="black", linewidth=1.2),
                   whiskerprops=dict(linewidth=0.8),
                   capprops=dict(linewidth=0.8),
                   flierprops=dict(marker=".", markersize=2, alpha=0.4))
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ylim = ax.get_ylim()
    ax.axhspan(0, ylim[1], color="red", alpha=0.04, zorder=0)
    ax.axhspan(ylim[0], 0, color="green", alpha=0.04, zorder=0)
    ax.set_ylim(ylim)
    ax.set_xticks(range(len(e_order)))
    ax.set_xticklabels(e_order)
    ax.set_ylabel("Delta Energy (kcal/mol)")
    ax.set_title("Binding-Induced Energy Change by Type")
    ax.text(-0.12, 1.05, "D", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel E: Fractional accessibility change (delta_frac_acc)
    ax = axes[1, 1]
    for rt in IONIZABLE_ORDER:
        subset = [(r["delta_frac_acc_num"],)
                  for r in res_data if r["restype"] == rt and r["delta_frac_acc_num"] is not None]
        if not subset:
            continue
        vals = [s[0] for s in subset]
        ax.hist(vals, bins=np.arange(-1.0, 0.55, 0.05), alpha=0.4,
                color=palette[rt], edgecolor="none", label=rt)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Delta Fractional Accessibility (complex - isolated)")
    ax.set_ylabel("Count")
    ax.set_title("Solvent Accessibility Change upon Binding")
    ax.legend(fontsize=6, ncol=2)
    ax.text(-0.12, 1.05, "E", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel F: Newly buried interface residues by type & component
    ax = axes[1, 2]
    newly_buried = [r for r in res_data if r.get("newly_buried_num") == 1]
    if newly_buried:
        nb_by_type_comp = defaultdict(lambda: defaultdict(int))
        for r in newly_buried:
            nb_by_type_comp[r["restype"]][r["component"]] += 1
        nb_types = [rt for rt in IONIZABLE_ORDER if rt in nb_by_type_comp]
        ab_counts = [nb_by_type_comp[rt].get("antibody", 0) for rt in nb_types]
        ag_counts = [nb_by_type_comp[rt].get("antigen", 0) for rt in nb_types]
        x = np.arange(len(nb_types))
        ax.bar(x - 0.15, ab_counts, 0.3, color=colors_3[1], edgecolor="gray",
               linewidth=0.5, label="Antibody")
        ax.bar(x + 0.15, ag_counts, 0.3, color=colors_3[2], edgecolor="gray",
               linewidth=0.5, label="Antigen")
        ax.set_xticks(x)
        ax.set_xticklabels(nb_types)
        ax.set_ylabel("Count")
        ax.set_title("Newly Buried Interface Residues by Type")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "No newly buried\ninterface residues", ha="center",
                va="center", transform=ax.transAxes, fontsize=12, color="gray")
        ax.set_title("Newly Buried Interface Residues")
    ax.text(-0.12, 1.05, "F", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    out2 = os.path.join(outdir, "residue_comparison.png")
    fig2.savefig(out2, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close(fig2)

    # ================================================================
    # FIGURE 3: Per-target breakdown (grouped by EGFR / TROP2 / EGFR_TROP2)
    # ================================================================
    target_types = sorted(set(r["target"] for r in prot_data))
    if len(target_types) > 1:
        fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5.5))
        fig3.suptitle("Per-Target Comparison", fontsize=14, fontweight="bold", y=1.02)
        fig3.subplots_adjust(wspace=0.30, left=0.05, right=0.97, top=0.88, bottom=0.15)

        # Panel A: BSA by target
        ax = axes3[0]
        targ_bsa = defaultdict(list)
        for r in prot_data:
            v = _safe_float(r["buried_surface_area"])
            if v is not None:
                targ_bsa[r["target"]].append(v)
        positions = []
        data = []
        tick_labels = []
        for i, t in enumerate(target_types):
            if targ_bsa[t]:
                positions.append(i)
                data.append(targ_bsa[t])
                tick_labels.append(t)
        if data:
            bp = ax.boxplot(data, positions=positions, widths=0.5, patch_artist=True,
                            medianprops=dict(color="black", linewidth=1.5))
            for patch, t in zip(bp["boxes"], tick_labels):
                patch.set_facecolor(target_colors[t])
                patch.set_alpha(0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels(tick_labels, rotation=20, ha="right")
        ax.set_ylabel("Buried Surface Area ($\\AA^2$)")
        ax.set_title("BSA by Target")
        ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

        # Panel B: Fraction buried by target
        ax = axes3[1]
        for i, (label, col, color) in enumerate(zip(labels_3,
                ["complex_frac_buried", "antibody_frac_buried", "antigen_frac_buried"],
                colors_3)):
            targ_fb = defaultdict(list)
            for r in prot_data:
                v = _safe_float(r[col])
                if v is not None:
                    targ_fb[r["target"]].append(v)
            means = [np.mean(targ_fb[t]) if targ_fb[t] else 0 for t in target_types]
            x = np.arange(len(target_types))
            ax.bar(x + i * 0.25 - 0.25, means, 0.25, color=color, edgecolor="gray",
                   linewidth=0.5, label=label, alpha=0.8)
        ax.set_xticks(np.arange(len(target_types)))
        ax.set_xticklabels(target_types, rotation=20, ha="right")
        ax.set_ylabel("Mean Fraction Buried")
        ax.set_title("Burial by Target and System")
        ax.legend(fontsize=7)
        ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

        # Panel C: Mean |pKa shift| by target
        ax = axes3[2]
        for i, (label, col, color) in enumerate(zip(labels_3,
                ["complex_mean_pKa_shift", "antibody_mean_pKa_shift", "antigen_mean_pKa_shift"],
                colors_3)):
            targ_ms = defaultdict(list)
            for r in prot_data:
                v = _safe_float(r[col])
                if v is not None:
                    targ_ms[r["target"]].append(abs(v))
            means = [np.mean(targ_ms[t]) if targ_ms[t] else 0 for t in target_types]
            x = np.arange(len(target_types))
            ax.bar(x + i * 0.25 - 0.25, means, 0.25, color=color, edgecolor="gray",
                   linewidth=0.5, label=label, alpha=0.8)
        ax.set_xticks(np.arange(len(target_types)))
        ax.set_xticklabels(target_types, rotation=20, ha="right")
        ax.set_ylabel("Mean |pKa Shift|")
        ax.set_title("Mean pKa Perturbation by Target")
        ax.legend(fontsize=7)
        ax.text(-0.12, 1.05, "C", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

        out3 = os.path.join(outdir, "target_comparison.png")
        fig3.savefig(out3, dpi=dpi, bbox_inches="tight")
        print(f"Saved {out3}")
        plt.close(fig3)

    # ================================================================
    # FIGURE 4: Antigen-focused binding impact (2x3)
    # ================================================================
    ag_data = [r for r in res_data if r["component"] == "antigen"]

    fig4, axes4 = plt.subplots(2, 3, figsize=(20, 12))
    fig4.suptitle("Antigen Impact upon Antibody Binding",
                  fontsize=15, fontweight="bold", y=0.98)
    fig4.subplots_adjust(hspace=0.38, wspace=0.32, left=0.06, right=0.97, top=0.92, bottom=0.08)

    palette = dict(zip(IONIZABLE_ORDER, sns.color_palette("Set2", len(IONIZABLE_ORDER))))

    # Panel A: Summary counts — stacked bar showing affected vs unaffected
    ax = axes4[0, 0]
    total_ag = len(ag_data)
    n_newly_buried = sum(1 for r in ag_data if r.get("newly_buried_num") == 1)
    n_shift_gt1 = sum(1 for r in ag_data if r["delta_pKa_shift_num"] is not None
                      and abs(r["delta_pKa_shift_num"]) > 1.0)
    n_shift_gt2 = sum(1 for r in ag_data if r["delta_pKa_shift_num"] is not None
                      and abs(r["delta_pKa_shift_num"]) > 2.0)
    n_acc_loss = sum(1 for r in ag_data if r["delta_frac_acc_num"] is not None
                     and r["delta_frac_acc_num"] < -0.1)

    categories = ["Newly\nBuried", "|ΔpKa| > 1", "|ΔpKa| > 2", "Acc. loss\n> 10%"]
    counts = [n_newly_buried, n_shift_gt1, n_shift_gt2, n_acc_loss]
    bar_colors = ["#E07B54", "#5B9BD5", "#4472C4", "#70AD47"]
    bars = ax.bar(categories, counts, color=bar_colors, edgecolor="gray", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Antigen Ionizable Residues")
    ax.set_title(f"Antigen Residues Affected by Binding\n(out of {total_ag} total)")
    ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel B: Mean |delta pKa shift| by residue type (antigen only)
    ax = axes4[0, 1]
    ag_by_type = defaultdict(list)
    for r in ag_data:
        if r["delta_pKa_shift_num"] is not None:
            ag_by_type[r["restype"]].append(abs(r["delta_pKa_shift_num"]))
    ag_types = [rt for rt in IONIZABLE_ORDER if rt in ag_by_type]
    ag_means = [np.mean(ag_by_type[rt]) for rt in ag_types]
    ag_stds = [np.std(ag_by_type[rt]) for rt in ag_types]
    x = np.arange(len(ag_types))
    bars = ax.bar(x, ag_means, color=[palette[rt] for rt in ag_types],
                  edgecolor="gray", linewidth=0.5)
    ax.errorbar(x, ag_means, yerr=ag_stds, fmt="none", ecolor="black",
                capsize=3, linewidth=0.8)
    for i, (m, n) in enumerate(zip(ag_means, [len(ag_by_type[rt]) for rt in ag_types])):
        ax.text(i, m + ag_stds[i] + 0.05, f"n={n}", ha="center", fontsize=7, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(ag_types)
    ax.set_ylabel("Mean |ΔpKa Shift| (complex − isolated)")
    ax.set_title("Binding-Induced pKa Perturbation\nof Antigen Residues by Type")
    ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel C: Top 20 most perturbed antigen residues (horizontal bar)
    ax = axes4[0, 2]
    ranked = [(r, abs(r["delta_pKa_shift_num"]))
              for r in ag_data if r["delta_pKa_shift_num"] is not None]
    ranked.sort(key=lambda x: x[1], reverse=True)
    top_n = 20
    top = ranked[:top_n]
    labels_top = [f"{r['restype']}{r['resid']} ({r['key'].split('_',1)[0]})"
                  for r, _ in top]
    vals_top = [r["delta_pKa_shift_num"] for r, _ in top]
    colors_top = ["#C0504D" if v < 0 else "#4F81BD" for v in vals_top]
    y = np.arange(len(top))
    ax.barh(y, vals_top, color=colors_top, edgecolor="gray", linewidth=0.3, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_top, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔpKa Shift (complex − isolated)")
    ax.set_title("Top 20 Most Perturbed Antigen Residues\n(Candidate Epitope Hotspots)")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#C0504D", label="pKa suppressed"),
                       Patch(facecolor="#4F81BD", label="pKa elevated")],
              fontsize=7, loc="lower right")
    ax.text(-0.18, 1.05, "C", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel D: Antigen accessibility change — isolated vs complex scatter
    ax = axes4[1, 0]
    for rt in IONIZABLE_ORDER:
        subset = [(r, float(r["isolated_frac_acc"]), float(r["complex_frac_acc"]))
                  for r in ag_data
                  if r["restype"] == rt
                  and r.get("isolated_frac_acc") and r.get("complex_frac_acc")
                  and r["isolated_frac_acc"] != "" and r["complex_frac_acc"] != ""]
        if not subset:
            continue
        iso_vals = [s[1] for s in subset]
        cpx_vals = [s[2] for s in subset]
        ax.scatter(iso_vals, cpx_vals, label=rt, alpha=0.4, s=12,
                   color=palette[rt], edgecolors="none")
    lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax.fill_between(lim, [0, 0], lim, color="red", alpha=0.05)
    ax.set_xlabel("Fractional Accessibility (isolated antigen)")
    ax.set_ylabel("Fractional Accessibility (in complex)")
    ax.set_title("Antigen Solvent Accessibility:\nIsolated vs Antibody-Bound")
    ax.legend(fontsize=6, ncol=2)
    ax.annotate("Buried by\nantibody", xy=(0.7, 0.15), xycoords="axes fraction",
                fontsize=10, color="firebrick", fontstyle="italic", ha="center")
    ax.text(-0.12, 1.05, "D", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel E: Newly buried vs not — compare delta pKa shift distributions
    ax = axes4[1, 1]
    buried_shifts = [r["delta_pKa_shift_num"] for r in ag_data
                     if r.get("newly_buried_num") == 1 and r["delta_pKa_shift_num"] is not None]
    not_buried_shifts = [r["delta_pKa_shift_num"] for r in ag_data
                         if r.get("newly_buried_num") == 0 and r["delta_pKa_shift_num"] is not None]
    bins = np.arange(-8, 8.5, 0.5)
    if not_buried_shifts:
        ax.hist(not_buried_shifts, bins=bins, alpha=0.5, color="#7F7F7F",
                edgecolor="white", linewidth=0.5, label=f"Not newly buried (n={len(not_buried_shifts)})",
                density=True)
    if buried_shifts:
        ax.hist(buried_shifts, bins=bins, alpha=0.6, color="#E07B54",
                edgecolor="white", linewidth=0.5, label=f"Newly buried (n={len(buried_shifts)})",
                density=True)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    if buried_shifts:
        ax.axvline(np.mean(buried_shifts), color="#E07B54", ls="--", lw=1,
                   label=f"buried mean={np.mean(buried_shifts):.2f}")
    if not_buried_shifts:
        ax.axvline(np.mean(not_buried_shifts), color="#7F7F7F", ls="--", lw=1,
                   label=f"other mean={np.mean(not_buried_shifts):.2f}")
    ax.set_xlabel("ΔpKa Shift (complex − isolated)")
    ax.set_ylabel("Density")
    ax.set_title("pKa Perturbation: Newly Buried\nvs Non-Interface Antigen Residues")
    ax.legend(fontsize=7)
    ax.text(-0.12, 1.05, "E", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel F: Epitope consistency heatmap — top antigen positions across sequences
    ax = axes4[1, 2]
    from collections import Counter
    resid_shifts = defaultdict(dict)
    for r in ag_data:
        if r["delta_pKa_shift_num"] is not None:
            label = f"{r['restype']}{r['resid']}"
            seq = r["key"].split("_", 1)[0]
            resid_shifts[label][seq] = r["delta_pKa_shift_num"]

    mean_abs = {lab: np.mean([abs(v) for v in seqs.values()])
                for lab, seqs in resid_shifts.items() if len(seqs) >= 3}
    top_residues = sorted(mean_abs, key=mean_abs.get, reverse=True)[:25]
    all_seqs = sorted(set(s for lab in top_residues for s in resid_shifts[lab]))

    heatmap_data = np.full((len(top_residues), len(all_seqs)), np.nan)
    for i, lab in enumerate(top_residues):
        for j, seq in enumerate(all_seqs):
            if seq in resid_shifts[lab]:
                heatmap_data[i, j] = resid_shifts[lab][seq]

    vmax = np.nanmax(np.abs(heatmap_data))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(all_seqs)))
    ax.set_xticklabels(all_seqs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_residues)))
    ax.set_yticklabels(top_residues, fontsize=7)
    ax.set_xlabel("Antibody Sequence")
    ax.set_title("Top 25 Most Perturbed Antigen Positions\nacross Antibody Sequences")
    cbar = fig4.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("ΔpKa Shift", fontsize=8)
    ax.text(-0.22, 1.05, "F", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    out4 = os.path.join(outdir, "antigen_binding_impact.png")
    fig4.savefig(out4, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out4}")
    plt.close(fig4)

    # ================================================================
    # FIGURE 5: Antigen energetics and charge impact (1x3)
    # ================================================================
    fig5, axes5 = plt.subplots(1, 3, figsize=(18, 5.5))
    fig5.suptitle("Antigen Energetic & Charge Impact of Antibody Binding",
                  fontsize=14, fontweight="bold", y=1.02)
    fig5.subplots_adjust(wspace=0.30, left=0.06, right=0.97, top=0.87, bottom=0.15)

    # Panel A: Delta energy by type (antigen)
    ax = axes5[0]
    e_by_type = defaultdict(list)
    for r in ag_data:
        if r["delta_energy_num"] is not None:
            e_by_type[r["restype"]].append(r["delta_energy_num"])
    e_types = [rt for rt in IONIZABLE_ORDER if rt in e_by_type]
    e_arrays = [np.array(e_by_type[rt]) for rt in e_types]
    if e_arrays and any(len(a) > 0 for a in e_arrays):
        valid_arrays = [a for a in e_arrays if len(a) > 0]
        valid_pos = [i for i, a in enumerate(e_arrays) if len(a) > 0]
        valid_types = [rt for rt, a in zip(e_types, e_arrays) if len(a) > 0]
        parts = ax.violinplot(valid_arrays, positions=valid_pos,
                              showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(palette[valid_types[i]])
            pc.set_alpha(0.7)
        ax.boxplot(valid_arrays, positions=valid_pos, widths=0.15,
                   patch_artist=True,
                   boxprops=dict(facecolor="white", linewidth=0.8),
                   medianprops=dict(color="black", linewidth=1.2),
                   whiskerprops=dict(linewidth=0.8),
                   capprops=dict(linewidth=0.8),
                   flierprops=dict(marker=".", markersize=2, alpha=0.4))
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ylim = ax.get_ylim()
    ax.axhspan(0, ylim[1], color="red", alpha=0.04, zorder=0)
    ax.axhspan(ylim[0], 0, color="green", alpha=0.04, zorder=0)
    ax.set_ylim(ylim)
    ax.annotate("Destabilized\nby binding", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="firebrick", fontstyle="italic")
    ax.annotate("Stabilized\nby binding", xy=(0.98, 0.05), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, color="darkgreen", fontstyle="italic")
    ax.set_xticks(range(len(e_types)))
    ax.set_xticklabels(e_types)
    ax.set_ylabel("ΔΔG (kcal/mol)  (complex − isolated)")
    ax.set_title("Antigen Energetic Cost of Binding\nby Residue Type")
    ax.text(-0.12, 1.08, "A", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel B: Per-system BSA vs net charge change
    ax = axes5[1]
    bsa_v = [_safe_float(r["buried_surface_area"]) for r in prot_data]
    cdelta_v = [_safe_float(r["charge_additivity_delta"]) for r in prot_data]
    targets = [r["target"] for r in prot_data]
    target_types = sorted(set(targets))
    target_colors = dict(zip(target_types, sns.color_palette("Set2", len(target_types))))
    for t in target_types:
        idx = [i for i, tg in enumerate(targets) if tg == t]
        bx = [bsa_v[i] for i in idx if bsa_v[i] is not None and cdelta_v[i] is not None]
        cy = [cdelta_v[i] for i in idx if bsa_v[i] is not None and cdelta_v[i] is not None]
        ax.scatter(bx, cy, color=target_colors[t], label=t, s=50, edgecolors="gray",
                   linewidth=0.5, alpha=0.8)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Buried Surface Area ($\\AA^2$)")
    ax.set_ylabel("Charge Change (complex − [Ab+Ag])")
    ax.set_title("Interface Size vs Charge\nRedistribution upon Binding")
    ax.legend(fontsize=8, title="Target", title_fontsize=8)
    ax.text(-0.12, 1.08, "B", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel C: Fraction of antigen ionizable residues buried — isolated vs complex
    ax = axes5[2]
    iso_fb = [_safe_float(r["antigen_frac_ion_buried"]) for r in prot_data]
    cpx_fb = [_safe_float(r["complex_frac_ion_buried"]) for r in prot_data]
    keys = [r["key"] for r in prot_data]
    x = np.arange(len(keys))
    width = 0.35
    ax.bar(x - width/2, iso_fb, width, color="#55A868", edgecolor="gray",
           linewidth=0.3, label="Antigen isolated", alpha=0.8)
    ax.bar(x + width/2, cpx_fb, width, color="#4C72B0", edgecolor="gray",
           linewidth=0.3, label="In complex", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([k.split("_", 1)[0] for k in keys], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Fraction Ionizable Residues Buried")
    ax.set_title("Antigen Ionizable Burial:\nIsolated vs Antibody-Bound")
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.08, "C", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    out5 = os.path.join(outdir, "antigen_energetics.png")
    fig5.savefig(out5, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out5}")
    plt.close(fig5)

    # ================================================================
    # FIGURE 6: Antibody-focused binding impact (2x3)
    # ================================================================
    ab_data = [r for r in res_data if r["component"] == "antibody"]

    fig6, axes6 = plt.subplots(2, 3, figsize=(20, 12))
    fig6.suptitle("Antibody Impact upon Antigen Binding",
                  fontsize=15, fontweight="bold", y=0.98)
    fig6.subplots_adjust(hspace=0.38, wspace=0.32, left=0.06, right=0.97, top=0.92, bottom=0.08)

    # Panel A: Summary counts
    ax = axes6[0, 0]
    total_ab = len(ab_data)
    n_nb_ab = sum(1 for r in ab_data if r.get("newly_buried_num") == 1)
    n_s1_ab = sum(1 for r in ab_data if r["delta_pKa_shift_num"] is not None
                  and abs(r["delta_pKa_shift_num"]) > 1.0)
    n_s2_ab = sum(1 for r in ab_data if r["delta_pKa_shift_num"] is not None
                  and abs(r["delta_pKa_shift_num"]) > 2.0)
    n_al_ab = sum(1 for r in ab_data if r["delta_frac_acc_num"] is not None
                  and r["delta_frac_acc_num"] < -0.1)

    categories = ["Newly\nBuried", "|ΔpKa| > 1", "|ΔpKa| > 2", "Acc. loss\n> 10%"]
    counts = [n_nb_ab, n_s1_ab, n_s2_ab, n_al_ab]
    bar_colors = ["#E07B54", "#5B9BD5", "#4472C4", "#70AD47"]
    bars = ax.bar(categories, counts, color=bar_colors, edgecolor="gray", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("Number of Antibody Ionizable Residues")
    ax.set_title(f"Antibody Residues Affected by Binding\n(out of {total_ab} total)")
    ax.text(-0.12, 1.05, "A", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel B: Mean |delta pKa shift| by residue type (antibody only)
    ax = axes6[0, 1]
    ab_by_type = defaultdict(list)
    for r in ab_data:
        if r["delta_pKa_shift_num"] is not None:
            ab_by_type[r["restype"]].append(abs(r["delta_pKa_shift_num"]))
    ab_types = [rt for rt in IONIZABLE_ORDER if rt in ab_by_type]
    ab_means = [np.mean(ab_by_type[rt]) for rt in ab_types]
    ab_stds = [np.std(ab_by_type[rt]) for rt in ab_types]
    x = np.arange(len(ab_types))
    bars = ax.bar(x, ab_means, color=[palette[rt] for rt in ab_types],
                  edgecolor="gray", linewidth=0.5)
    ax.errorbar(x, ab_means, yerr=ab_stds, fmt="none", ecolor="black",
                capsize=3, linewidth=0.8)
    for i, (m, n) in enumerate(zip(ab_means, [len(ab_by_type[rt]) for rt in ab_types])):
        ax.text(i, m + ab_stds[i] + 0.05, f"n={n}", ha="center", fontsize=7, color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels(ab_types)
    ax.set_ylabel("Mean |ΔpKa Shift| (complex − isolated)")
    ax.set_title("Binding-Induced pKa Perturbation\nof Antibody Residues by Type")
    ax.text(-0.12, 1.05, "B", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel C: Top 20 most perturbed antibody residues (horizontal bar)
    ax = axes6[0, 2]
    ranked_ab = [(r, abs(r["delta_pKa_shift_num"]))
                 for r in ab_data if r["delta_pKa_shift_num"] is not None]
    ranked_ab.sort(key=lambda x: x[1], reverse=True)
    top_ab = ranked_ab[:20]
    labels_top = [f"{r['restype']}{r['resid']} ({r['key'].split('_',1)[0]})"
                  for r, _ in top_ab]
    vals_top = [r["delta_pKa_shift_num"] for r, _ in top_ab]
    colors_top = ["#C0504D" if v < 0 else "#4F81BD" for v in vals_top]
    y = np.arange(len(top_ab))
    ax.barh(y, vals_top, color=colors_top, edgecolor="gray", linewidth=0.3, height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels_top, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("ΔpKa Shift (complex − isolated)")
    ax.set_title("Top 20 Most Perturbed Antibody Residues\n(Paratope Hotspots)")
    ax.legend(handles=[Patch(facecolor="#C0504D", label="pKa suppressed"),
                       Patch(facecolor="#4F81BD", label="pKa elevated")],
              fontsize=7, loc="lower right")
    ax.text(-0.18, 1.05, "C", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel D: Antibody accessibility change — isolated vs complex scatter
    ax = axes6[1, 0]
    for rt in IONIZABLE_ORDER:
        subset = [(r, float(r["isolated_frac_acc"]), float(r["complex_frac_acc"]))
                  for r in ab_data
                  if r["restype"] == rt
                  and r.get("isolated_frac_acc") and r.get("complex_frac_acc")
                  and r["isolated_frac_acc"] != "" and r["complex_frac_acc"] != ""]
        if not subset:
            continue
        iso_vals = [s[1] for s in subset]
        cpx_vals = [s[2] for s in subset]
        ax.scatter(iso_vals, cpx_vals, label=rt, alpha=0.4, s=12,
                   color=palette[rt], edgecolors="none")
    lim = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.5)
    ax.fill_between(lim, [0, 0], lim, color="red", alpha=0.05)
    ax.set_xlabel("Fractional Accessibility (isolated antibody)")
    ax.set_ylabel("Fractional Accessibility (in complex)")
    ax.set_title("Antibody Solvent Accessibility:\nIsolated vs Antigen-Bound")
    ax.legend(fontsize=6, ncol=2)
    ax.annotate("Buried by\nantigen", xy=(0.7, 0.15), xycoords="axes fraction",
                fontsize=10, color="firebrick", fontstyle="italic", ha="center")
    ax.text(-0.12, 1.05, "D", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel E: Newly buried vs not — compare delta pKa shift distributions
    ax = axes6[1, 1]
    buried_shifts_ab = [r["delta_pKa_shift_num"] for r in ab_data
                        if r.get("newly_buried_num") == 1 and r["delta_pKa_shift_num"] is not None]
    not_buried_shifts_ab = [r["delta_pKa_shift_num"] for r in ab_data
                            if r.get("newly_buried_num") == 0 and r["delta_pKa_shift_num"] is not None]
    bins = np.arange(-8, 8.5, 0.5)
    if not_buried_shifts_ab:
        ax.hist(not_buried_shifts_ab, bins=bins, alpha=0.5, color="#7F7F7F",
                edgecolor="white", linewidth=0.5,
                label=f"Not newly buried (n={len(not_buried_shifts_ab)})", density=True)
    if buried_shifts_ab:
        ax.hist(buried_shifts_ab, bins=bins, alpha=0.6, color="#E07B54",
                edgecolor="white", linewidth=0.5,
                label=f"Newly buried (n={len(buried_shifts_ab)})", density=True)
    ax.axvline(0, color="gray", ls="--", lw=0.8)
    if buried_shifts_ab:
        ax.axvline(np.mean(buried_shifts_ab), color="#E07B54", ls="--", lw=1,
                   label=f"buried mean={np.mean(buried_shifts_ab):.2f}")
    if not_buried_shifts_ab:
        ax.axvline(np.mean(not_buried_shifts_ab), color="#7F7F7F", ls="--", lw=1,
                   label=f"other mean={np.mean(not_buried_shifts_ab):.2f}")
    ax.set_xlabel("ΔpKa Shift (complex − isolated)")
    ax.set_ylabel("Density")
    ax.set_title("pKa Perturbation: Newly Buried\nvs Non-Interface Antibody Residues")
    ax.legend(fontsize=7)
    ax.text(-0.12, 1.05, "E", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel F: Paratope consistency heatmap — top antibody positions across targets
    ax = axes6[1, 2]
    resid_shifts_ab = defaultdict(dict)
    for r in ab_data:
        if r["delta_pKa_shift_num"] is not None:
            label = f"{r['restype']}{r['resid']}"
            target = r["key"].split("_", 1)[1] if "_" in r["key"] else r["key"]
            resid_shifts_ab[label][target] = r["delta_pKa_shift_num"]

    mean_abs_ab = {lab: np.mean([abs(v) for v in targs.values()])
                   for lab, targs in resid_shifts_ab.items() if len(targs) >= 2}
    top_res_ab = sorted(mean_abs_ab, key=mean_abs_ab.get, reverse=True)[:25]
    all_targs = sorted(set(t for lab in top_res_ab for t in resid_shifts_ab[lab]))

    hm_ab = np.full((len(top_res_ab), len(all_targs)), np.nan)
    for i, lab in enumerate(top_res_ab):
        for j, targ in enumerate(all_targs):
            if targ in resid_shifts_ab[lab]:
                hm_ab[i, j] = resid_shifts_ab[lab][targ]

    vmax_ab = np.nanmax(np.abs(hm_ab))
    im = ax.imshow(hm_ab, aspect="auto", cmap="RdBu_r", vmin=-vmax_ab, vmax=vmax_ab,
                   interpolation="nearest")
    ax.set_xticks(range(len(all_targs)))
    ax.set_xticklabels(all_targs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(top_res_ab)))
    ax.set_yticklabels(top_res_ab, fontsize=7)
    ax.set_xlabel("Target Antigen")
    ax.set_title("Top 25 Most Perturbed Antibody Positions\nacross Target Antigens")
    cbar = fig6.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("ΔpKa Shift", fontsize=8)
    ax.text(-0.22, 1.05, "F", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    out6 = os.path.join(outdir, "antibody_binding_impact.png")
    fig6.savefig(out6, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out6}")
    plt.close(fig6)

    # ================================================================
    # FIGURE 7: Antibody energetics (1x3)
    # ================================================================
    fig7, axes7 = plt.subplots(1, 3, figsize=(18, 5.5))
    fig7.suptitle("Antibody Energetic & Charge Impact of Antigen Binding",
                  fontsize=14, fontweight="bold", y=1.02)
    fig7.subplots_adjust(wspace=0.30, left=0.06, right=0.97, top=0.87, bottom=0.15)

    # Panel A: Delta energy by type (antibody)
    ax = axes7[0]
    e_by_type_ab = defaultdict(list)
    for r in ab_data:
        if r["delta_energy_num"] is not None:
            e_by_type_ab[r["restype"]].append(r["delta_energy_num"])
    e_types_ab = [rt for rt in IONIZABLE_ORDER if rt in e_by_type_ab]
    e_arrays_ab = [np.array(e_by_type_ab[rt]) for rt in e_types_ab]
    if e_arrays_ab and any(len(a) > 0 for a in e_arrays_ab):
        valid_arrays = [a for a in e_arrays_ab if len(a) > 0]
        valid_pos = [i for i, a in enumerate(e_arrays_ab) if len(a) > 0]
        valid_types = [rt for rt, a in zip(e_types_ab, e_arrays_ab) if len(a) > 0]
        parts = ax.violinplot(valid_arrays, positions=valid_pos,
                              showmeans=False, showmedians=False, showextrema=False)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(palette[valid_types[i]])
            pc.set_alpha(0.7)
        ax.boxplot(valid_arrays, positions=valid_pos, widths=0.15,
                   patch_artist=True,
                   boxprops=dict(facecolor="white", linewidth=0.8),
                   medianprops=dict(color="black", linewidth=1.2),
                   whiskerprops=dict(linewidth=0.8),
                   capprops=dict(linewidth=0.8),
                   flierprops=dict(marker=".", markersize=2, alpha=0.4))
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ylim = ax.get_ylim()
    ax.axhspan(0, ylim[1], color="red", alpha=0.04, zorder=0)
    ax.axhspan(ylim[0], 0, color="green", alpha=0.04, zorder=0)
    ax.set_ylim(ylim)
    ax.annotate("Destabilized\nby binding", xy=(0.98, 0.95), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="firebrick", fontstyle="italic")
    ax.annotate("Stabilized\nby binding", xy=(0.98, 0.05), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, color="darkgreen", fontstyle="italic")
    ax.set_xticks(range(len(e_types_ab)))
    ax.set_xticklabels(e_types_ab)
    ax.set_ylabel("ΔΔG (kcal/mol)  (complex − isolated)")
    ax.set_title("Antibody Energetic Cost of Binding\nby Residue Type")
    ax.text(-0.12, 1.08, "A", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel B: Antibody vs Antigen side-by-side mean |delta pKa shift|
    ax = axes7[1]
    all_types_both = sorted(set(
        [rt for rt in IONIZABLE_ORDER if rt in ab_by_type or rt in ag_by_type]))
    x = np.arange(len(all_types_both))
    width = 0.35
    ab_m = [np.mean(ab_by_type.get(rt, [0])) for rt in all_types_both]
    ag_m = [np.mean(ag_by_type.get(rt, [0])) for rt in all_types_both]
    ax.bar(x - width/2, ab_m, width, color="#DD8452", edgecolor="gray",
           linewidth=0.5, label="Antibody", alpha=0.8)
    ax.bar(x + width/2, ag_m, width, color="#55A868", edgecolor="gray",
           linewidth=0.5, label="Antigen", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(all_types_both)
    ax.set_ylabel("Mean |ΔpKa Shift|")
    ax.set_title("Antibody vs Antigen: Binding-Induced\npKa Perturbation by Type")
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.08, "B", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    # Panel C: Fraction of antibody ionizable residues buried — isolated vs complex
    ax = axes7[2]
    iso_fb_ab = [_safe_float(r["antibody_frac_ion_buried"]) for r in prot_data]
    cpx_fb_ab = [_safe_float(r["complex_frac_ion_buried"]) for r in prot_data]
    keys = [r["key"] for r in prot_data]
    x = np.arange(len(keys))
    width = 0.35
    ax.bar(x - width/2, iso_fb_ab, width, color="#DD8452", edgecolor="gray",
           linewidth=0.3, label="Antibody isolated", alpha=0.8)
    ax.bar(x + width/2, cpx_fb_ab, width, color="#4C72B0", edgecolor="gray",
           linewidth=0.3, label="In complex", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([k.split("_", 1)[0] for k in keys], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Fraction Ionizable Residues Buried")
    ax.set_title("Antibody Ionizable Burial:\nIsolated vs Antigen-Bound")
    ax.legend(fontsize=8)
    ax.text(-0.12, 1.08, "C", transform=ax.transAxes, fontsize=16, fontweight="bold", va="top")

    out7 = os.path.join(outdir, "antibody_energetics.png")
    fig7.savefig(out7, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out7}")
    plt.close(fig7)


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comparative MCCE analysis: complex vs antibody vs antigen.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--complex-dir", default=COMPLEX_DIR,
                        help=f"Analysis output dir for complex; default: {COMPLEX_DIR}")
    parser.add_argument("--antibody-dir", default=ANTIBODY_DIR,
                        help=f"Analysis output dir for antibody; default: {ANTIBODY_DIR}")
    parser.add_argument("--antigen-dir", default=ANTIGEN_DIR,
                        help=f"Analysis output dir for antigen; default: {ANTIGEN_DIR}")
    parser.add_argument("-o", "--outdir", default="AB_mcce_comparative_analysis",
                        help="Output directory; default: %(default)s")
    parser.add_argument("--ph", type=float, default=7.0,
                        help="pH used in the original analysis; default: %(default)s")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip analysis, regenerate figures from existing CSVs")
    parser.add_argument("--no-plot", action="store_true",
                        help="Run analysis only, skip figures")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Figure DPI; default: %(default)s")
    args = parser.parse_args()

    if not args.plot_only:
        run_comparison(args.complex_dir, args.antibody_dir, args.antigen_dir,
                       args.outdir, args.ph)
    if not args.no_plot:
        run_plot(args.outdir, args.ph, args.dpi)


if __name__ == "__main__":
    main()
