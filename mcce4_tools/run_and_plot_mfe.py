#!/usr/bin/env python

import subprocess
import argparse
import sys
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORBLIND_PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#CC79A7",
    "#56B4E9", "#D55E00", "#F0E442", "#000000",
    "#332288", "#AA4499", "#44AA99",
]

# --- Command Line Arguments ---
EXAMPLE_RESIDUES = ["FRLAA3101_", "FRLAA3103_", "FRLAB3002_", "FRLAB3003_"]
ALL_TERMS = ["vdw0", "vdw1", "tors", "ebkb", "dsol", "offset", "pH&pK0", "Eh&Em0", "-TS", "residues", "TOTAL"]

parser = argparse.ArgumentParser(
    description="Plot MFE terms for residues of interest.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=f"""Examples:
  python run_and_plot_mfe.py
  python run_and_plot_mfe.py -u Kcal -p 7 -c 0.05 -r {" ".join(EXAMPLE_RESIDUES)}
  python run_and_plot_mfe.py -u meV -p 7 -c 0.05 -r {" ".join(EXAMPLE_RESIDUES)}
  python run_and_plot_mfe.py -u pH -p 7 -c 0.05 -r {" ".join(EXAMPLE_RESIDUES)}
  python run_and_plot_mfe.py -ignore-terms "vdw0,dsol,pH&pK0,Eh&Em0"
  python run_and_plot_mfe.py -o my_output_dir
""",
)
parser.add_argument("-u", "--unit", choices=["pH", "meV", "Kcal"], default="Kcal",
                    help="Units to plot: pH, meV, or Kcal (default: Kcal)")
parser.add_argument("-p", "--pH", type=float, default=7.0, help="pH for mfe.py")
parser.add_argument("-c", "--cut", type=float, default=0.05, help="Cutoff for mfe.py")
parser.add_argument("-r", "--residues", nargs="+",
                    default=EXAMPLE_RESIDUES,
                    metavar="RESIDUE",
                    help="Residue IDs to run through mfe.py")
parser.add_argument("-ignore-terms", type=str, default="",
                    metavar="TERM,TERM,...",
                    help=f"Comma-separated list of MFE terms to exclude from the plot. "
                         f"Available terms: {', '.join(ALL_TERMS)}")
parser.add_argument("-o", "--outdir", type=str, default="mfe_plots",
                    help="Output directory for figures (default: mfe_plots)")
args = parser.parse_args()

MFE_COLUMNS = {
    "label": (0, 10),
    "pH": (10, 17),
    "meV": (17, 25),
    "Kcal": (25, 34),
}

residues_to_run = args.residues
os.makedirs(args.outdir, exist_ok=True)

def format_residue_label(res_id):
    """Convert e.g. 'FRLAA3101_' -> 'FRL A 3101'."""
    raw = res_id.rstrip("_")
    name = raw[:3]
    chain = raw[3]
    resnum = raw[4:]
    return f"{name} {chain} {resnum}"

def parse_mfe_line(line):
    """Parse one fixed-width mfe.py table row."""
    label_start, label_end = MFE_COLUMNS["label"]
    val_start, val_end = MFE_COLUMNS[args.unit]
    label = line[label_start:label_end].strip()
    value = line[val_start:val_end].strip()

    if not label or not value:
        return None, None

    try:
        return label, float(value)
    except ValueError:
        return None, None

def run_mfe(res_id):
    """Run mfe.py and extract the selected unit from the MFE term table."""
    cmd = ["mfe.py", res_id, "-p", str(args.pH), "-c", str(args.cut)]
    try:
        result = subprocess.check_output(cmd, text=True)
    except subprocess.CalledProcessError as err:
        print(f"Warning: mfe.py failed for {res_id}: {err}", file=sys.stderr)
        return None

    data = {}
    capture = False

    for line in result.splitlines():
        if not line.strip():
            continue

        if "Terms" in line and "pH" in line:
            capture = True
            continue

        if not capture or "---" in line or "***" in line:
            continue

        label, value = parse_mfe_line(line)
        if label is None:
            continue

        data[label] = value
        if label == "TOTAL":
            break

    return data

# --- Data Processing ---
all_results = {}
for res in residues_to_run:
    print(f"Processing {res}...")
    mfe_vals = run_mfe(res)
    if mfe_vals:
        all_results[res] = mfe_vals

if not all_results:
    print("Error: No data extracted.")
    sys.exit(1)

df = pd.DataFrame(all_results).T
ignore = set(t.strip() for t in args.ignore_terms.split(",") if t.strip())
unknown = ignore - set(ALL_TERMS)
if unknown:
    print(f"Warning: unknown terms ignored: {', '.join(sorted(unknown))}", file=sys.stderr)
term_order = [t for t in ALL_TERMS if t not in ignore]
df = df.reindex(columns=term_order).fillna(0)
df.index = [format_residue_label(r) for r in df.index]

# --- Plotting ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 16,
    "axes.titlesize": 18,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
})

colors = COLORBLIND_PALETTE[:len(term_order)]
fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
df.plot(kind='bar', width=0.8, ax=ax, color=colors, edgecolor="black", linewidth=0.5)

ax.axhline(0, color='black', linewidth=1)
ax.set_ylabel(f"MFE Contribution ({args.unit})")
ax.set_xlabel("Site")
ax.tick_params(axis='x', rotation=0)
ax.grid(axis='y', linestyle=':', alpha=0.5)
ax.legend(title="Terms", bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)

for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

for p in ax.patches:
    val = p.get_height()
    if abs(val) < 0.01:
        continue
    ax.annotate(f'{val:.2f}',
                (p.get_x() + p.get_width() / 2., val),
                ha='center', va='bottom' if val > 0 else 'top',
                fontsize=8, rotation=90,
                xytext=(0, 5 if val > 0 else -5),
                textcoords='offset points')

for i in range(len(df) - 1):
    ax.axvline(i + 0.5, color='grey', linestyle='--', alpha=0.3)

basename = f"mfe_comparison_{args.unit}"
for ext in ("png", "pdf", "svg"):
    outpath = os.path.join(args.outdir, f"{basename}.{ext}")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
print(f"Saved figures to {args.outdir}/ ({basename}.png, .pdf, .svg)")
