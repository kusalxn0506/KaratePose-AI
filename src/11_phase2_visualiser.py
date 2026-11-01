import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional

from src.config import PLOT_CONFIG, PATHS

plt.style.use(PLOT_CONFIG["style"])

# Phase 2 colour palette
_FILTER_COLORS = {
    "raw":           "#9E9E9E",
    "skf":           "#FF9800",
    "linear_interp": "#2196F3",
    "ekf":           "#4CAF50",
}
_FILTER_LABELS = {
    "raw":           "Raw (No Filter)",
    "skf":           "Savitzky–Golay",
    "linear_interp": "Linear Interp.",
    "ekf":           "Event-Triggered EKF",
}


# ── 1. Trajectory Comparison ─────────────────────────────────────────────────

def plot_trajectory_comparison(
    diagnostics_dir: Path,
    joint: str = "right_wrist",
    model: str = "mediapipe",
    axis: str = "x",
    frame_range: tuple = None,
    save_path: Optional[Path] = None,
):
    """Overlay raw vs all filters for a single joint coordinate over time."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for mode, color in _FILTER_COLORS.items():
        csvs = sorted(diagnostics_dir.glob(f"diagnostics_{mode}_*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        sub = df[(df["joint_name"] == joint) & (df["model"] == model)].copy()
        if sub.empty:
            continue
        if frame_range:
            sub = sub[(sub["frame"] >= frame_range[0]) & (sub["frame"] <= frame_range[1])]

        col = f"filtered_{axis}" if mode != "raw" else f"raw_{axis}"
        if mode == "raw":
            ax.plot(sub["frame"], sub[f"raw_{axis}"], color=color,
                    linewidth=1.0, alpha=0.5, label=_FILTER_LABELS[mode])
        else:
            ax.plot(sub["frame"], sub[col], color=color,
                    linewidth=1.5, label=_FILTER_LABELS[mode])

    ax.set_xlabel("Frame", fontsize=PLOT_CONFIG["font_size"])
    ax.set_ylabel(f"{axis.upper()} coordinate (normalised)", fontsize=PLOT_CONFIG["font_size"])
    ax.set_title(f"Trajectory Comparison — {joint} ({model})", fontsize=PLOT_CONFIG["font_size"] + 2)
    ax.legend(fontsize=PLOT_CONFIG["font_size"] - 1, loc="best")
    ax.tick_params(labelsize=PLOT_CONFIG["font_size"])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG["figure_dpi"], bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 2. Velocity Profile (Kime Preservation) ─────────────────────────────────

def plot_velocity_profiles(
    diagnostics_dir: Path,
    joint: str = "right_wrist",
    model: str = "mediapipe",
    save_path: Optional[Path] = None,
):
    """Compare velocity magnitude across filters — shows Kime preservation."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for mode, color in _FILTER_COLORS.items():
        csvs = sorted(diagnostics_dir.glob(f"diagnostics_{mode}_*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        sub = df[(df["joint_name"] == joint) & (df["model"] == model)].copy()
        if sub.empty:
            continue

        if mode == "raw":
            dx = np.diff(sub["raw_x"].values, prepend=sub["raw_x"].values[0])
            dy = np.diff(sub["raw_y"].values, prepend=sub["raw_y"].values[0])
        else:
            dx = np.diff(sub["filtered_x"].values, prepend=sub["filtered_x"].values[0])
            dy = np.diff(sub["filtered_y"].values, prepend=sub["filtered_y"].values[0])

        speed = np.sqrt(dx**2 + dy**2)
        lw = 1.0 if mode == "raw" else 1.5
        alpha = 0.4 if mode == "raw" else 0.9
        ax.plot(sub["frame"].values, speed, color=color,
                linewidth=lw, alpha=alpha, label=_FILTER_LABELS[mode])

    ax.set_xlabel("Frame", fontsize=PLOT_CONFIG["font_size"])
    ax.set_ylabel("Velocity magnitude (Δpx/frame)", fontsize=PLOT_CONFIG["font_size"])
    ax.set_title(f"Velocity Profile — {joint} ({model}) — Kime Preservation", fontsize=PLOT_CONFIG["font_size"] + 2)
    ax.legend(fontsize=PLOT_CONFIG["font_size"] - 1, loc="best")
    ax.tick_params(labelsize=PLOT_CONFIG["font_size"])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG["figure_dpi"], bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 3. MPJPE Bar Chart ──────────────────────────────────────────────────────

def plot_mpjpe_bars(
    benchmark_csv: Path,
    save_path: Optional[Path] = None,
):
    """Grouped bar chart: MPJPE per filter mode, grouped by model."""
    df = pd.read_csv(benchmark_csv)
    df["filter_label"] = df["filter_mode"].map(_FILTER_LABELS)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df, x="filter_label", y="mpjpe_mean", hue="model",
        palette=[PLOT_CONFIG["color_set_a"], PLOT_CONFIG["color_set_b"]],
        ax=ax, edgecolor="black", linewidth=0.5,
    )
    ax.set_xlabel("Filter Mode", fontsize=PLOT_CONFIG["font_size"])
    ax.set_ylabel("Mean MPJPE (normalised)", fontsize=PLOT_CONFIG["font_size"])
    ax.set_title("Filter Comparison — Mean Per-Joint Position Error", fontsize=PLOT_CONFIG["font_size"] + 2)
    ax.tick_params(axis="x", rotation=15, labelsize=PLOT_CONFIG["font_size"])
    ax.tick_params(axis="y", labelsize=PLOT_CONFIG["font_size"])
    ax.legend(title="Model", fontsize=PLOT_CONFIG["font_size"] - 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG["figure_dpi"], bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 4. PCK@0.2 Per-Joint Radar Chart ────────────────────────────────────────

def plot_pck_radar(
    detailed_csv: Path,
    model: str = "mediapipe",
    save_path: Optional[Path] = None,
):
    """Radar chart of PCK@0.2 per joint for each filter mode."""
    df = pd.read_csv(detailed_csv)
    df = df[df["model"] == model]

    joints = sorted(df["joint_name"].unique())
    n = len(joints)
    if n == 0:
        return

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for mode, color in _FILTER_COLORS.items():
        sub = df[df["filter_mode"] == mode]
        if sub.empty:
            continue
        values = [sub[sub["joint_name"] == j]["pck_at_0.2"].values[0]
                  if j in sub["joint_name"].values else 0 for j in joints]
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=1.5, label=_FILTER_LABELS[mode])
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([j.replace("_", " ").title() for j in joints],
                       fontsize=PLOT_CONFIG["font_size"] - 2)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"PCK@0.2 Per Joint — {model}", fontsize=PLOT_CONFIG["font_size"] + 2, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=PLOT_CONFIG["font_size"] - 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG["figure_dpi"], bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 5. PVR (Velocity Preservation Ratio) Bar Chart ──────────────────────────

def plot_pvr_bars(
    benchmark_csv: Path,
    save_path: Optional[Path] = None,
):
    """Bar chart of Peak Velocity Preservation Ratio — the Kime metric."""
    df = pd.read_csv(benchmark_csv)
    df["filter_label"] = df["filter_mode"].map(_FILTER_LABELS)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(
        data=df, x="filter_label", y="pvr_mean", hue="model",
        palette=[PLOT_CONFIG["color_set_a"], PLOT_CONFIG["color_set_b"]],
        ax=ax, edgecolor="black", linewidth=0.5,
    )
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, label="Ideal PVR = 1.0")
    ax.set_xlabel("Filter Mode", fontsize=PLOT_CONFIG["font_size"])
    ax.set_ylabel("Peak Velocity Preservation Ratio", fontsize=PLOT_CONFIG["font_size"])
    ax.set_title("Kime Fidelity — Velocity Preservation by Filter", fontsize=PLOT_CONFIG["font_size"] + 2)
    ax.tick_params(axis="x", rotation=15, labelsize=PLOT_CONFIG["font_size"])
    ax.tick_params(axis="y", labelsize=PLOT_CONFIG["font_size"])
    ax.legend(fontsize=PLOT_CONFIG["font_size"] - 1)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG["figure_dpi"], bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 6. CoG Trajectory Plot ──────────────────────────────────────────────────

def plot_cog_trajectory(
    cog_dir: Path,
    model: str = "mediapipe",
    save_path: Optional[Path] = None,
):
    """CoG X-Y trajectory in sagittal plane, one line per filter mode."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for mode, color in _FILTER_COLORS.items():
        csvs = sorted(cog_dir.glob(f"cog_{mode}_*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        lw = 1.0 if mode == "raw" else 1.5
        alpha = 0.4 if mode == "raw" else 0.9
        ax.plot(sub["cog_x"], sub["cog_y"], color=color,
                linewidth=lw, alpha=alpha, label=_FILTER_LABELS[mode])

    ax.set_xlabel("CoG X (normalised)", fontsize=PLOT_CONFIG["font_size"])
    ax.set_ylabel("CoG Y (normalised)", fontsize=PLOT_CONFIG["font_size"])
    ax.set_title(f"Center of Gravity Trajectory — {model}", fontsize=PLOT_CONFIG["font_size"] + 2)
    ax.invert_yaxis()
    ax.legend(fontsize=PLOT_CONFIG["font_size"] - 1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=PLOT_CONFIG["font_size"])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG["figure_dpi"], bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 7. CoG Stance Depth Over Time ────────────────────────────────────────────

def plot_stance_depth(
    cog_dir: Path,
    model: str = "mediapipe",
    save_path: Optional[Path] = None,
):
    """Stance depth (ankle-to-ankle horizontal distance) over frames."""
    fig, ax = plt.subplots(figsize=(12, 4))

    for mode, color in _FILTER_COLORS.items():
        csvs = sorted(cog_dir.glob(f"cog_{mode}_*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0])
        sub = df[df["model"] == model]
        if sub.empty:
            continue
        col = "stance_depth_m" if "stance_depth_m" in sub.columns else "stance_depth"
        if col not in sub.columns:
            continue
        lw = 1.0 if mode == "raw" else 1.5
        alpha = 0.4 if mode == "raw" else 0.9
        ax.plot(sub["frame"], sub[col], color=color,
                linewidth=lw, alpha=alpha, label=_FILTER_LABELS[mode])

    ax.set_xlabel("Frame", fontsize=PLOT_CONFIG["font_size"])
    ax.set_ylabel("Stance Depth", fontsize=PLOT_CONFIG["font_size"])
    ax.set_title(f"Stance Depth Over Time — {model}", fontsize=PLOT_CONFIG["font_size"] + 2)
    ax.legend(fontsize=PLOT_CONFIG["font_size"] - 1, loc="best")
    ax.tick_params(labelsize=PLOT_CONFIG["font_size"])
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG["figure_dpi"], bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── 8. Benchmark Summary Table ───────────────────────────────────────────────

def plot_benchmark_table(
    benchmark_csv: Path,
    save_path: Optional[Path] = None,
):
    """Render the benchmark summary as a publication-quality table figure."""
    df = pd.read_csv(benchmark_csv)
    df["filter_mode"] = df["filter_mode"].map(_FILTER_LABELS)

    display_cols = ["model", "filter_mode", "mpjpe_mean", "pck_mean", "phase_lag_mean", "pvr_mean"]
    display_cols = [c for c in display_cols if c in df.columns]
    dt = df[display_cols].copy()
    dt.columns = ["Model", "Filter", "MPJPE ↓", "PCK@0.2 ↑", "Phase Lag ↓", "PVR (≈1) ↑"]

    for c in dt.columns[2:]:
        dt[c] = dt[c].apply(lambda v: f"{v:.4f}" if isinstance(v, float) else v)

    fig, ax = plt.subplots(figsize=(12, 2 + len(dt) * 0.45))
    ax.axis("off")
    table = ax.table(cellText=dt.values, colLabels=dt.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(PLOT_CONFIG["font_size"])
    table.scale(1.2, 1.6)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e0e0e0")
        elif col == 1:
            label = cell.get_text().get_text()
            if label == _FILTER_LABELS["ekf"]:
                cell.set_text_props(weight="bold", color="#2E7D32")

    ax.set_title("Phase 2 — Filter Benchmark Summary", weight="bold",
                 fontsize=PLOT_CONFIG["font_size"] + 4, pad=20)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=PLOT_CONFIG["figure_dpi"], bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ── Master Generator ─────────────────────────────────────────────────────────

def generate_all_phase2_plots(
    filtered_dir: Path = None,
    cog_dir: Path = None,
    benchmark_dir: Path = None,
    output_dir: Path = None,
):
    """Generate all Paper 2 figures and save to output_dir."""
    filtered_dir = filtered_dir or PATHS["filtered"]
    cog_dir = cog_dir or PATHS["cog"]
    benchmark_dir = benchmark_dir or PATHS["benchmark"]
    output_dir = output_dir or PATHS["plots"] / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = benchmark_dir / "benchmark_summary.csv"
    detailed_csv = benchmark_dir / "benchmark_detailed.csv"

    count = 0

    # 1. Trajectory comparison
    print("  📈 [1/8] Trajectory comparison...")
    plot_trajectory_comparison(filtered_dir, joint="right_wrist", model="mediapipe",
                               save_path=output_dir / "trajectory_wrist_mp.png")
    count += 1

    # 2. Velocity profiles
    print("  📈 [2/8] Velocity profiles (Kime)...")
    plot_velocity_profiles(filtered_dir, joint="right_wrist", model="mediapipe",
                           save_path=output_dir / "velocity_wrist_mp.png")
    count += 1

    # 3. MPJPE bars
    if summary_csv.exists():
        print("  📈 [3/8] MPJPE bar chart...")
        plot_mpjpe_bars(summary_csv, save_path=output_dir / "mpjpe_bars.png")
        count += 1

    # 4. PCK radar
    if detailed_csv.exists():
        print("  📈 [4/8] PCK@0.2 radar chart...")
        plot_pck_radar(detailed_csv, model="mediapipe",
                       save_path=output_dir / "pck_radar_mp.png")
        count += 1

    # 5. PVR bars
    if summary_csv.exists():
        print("  📈 [5/8] PVR bar chart (Kime fidelity)...")
        plot_pvr_bars(summary_csv, save_path=output_dir / "pvr_bars.png")
        count += 1

    # 6. CoG trajectory
    print("  📈 [6/8] CoG trajectory...")
    plot_cog_trajectory(cog_dir, model="mediapipe",
                        save_path=output_dir / "cog_trajectory_mp.png")
    count += 1

    # 7. Stance depth
    print("  📈 [7/8] Stance depth over time...")
    plot_stance_depth(cog_dir, model="mediapipe",
                      save_path=output_dir / "stance_depth_mp.png")
    count += 1

    # 8. Benchmark table
    if summary_csv.exists():
        print("  📈 [8/8] Benchmark summary table...")
        plot_benchmark_table(summary_csv, save_path=output_dir / "benchmark_table.png")
        count += 1

    print(f"\n  ✅ Generated {count} Phase 2 plots → {output_dir}")


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  🥋 KaratePose-AI — Phase 2 Visualiser")
    print("=" * 60)
    generate_all_phase2_plots()
