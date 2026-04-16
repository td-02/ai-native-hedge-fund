from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
MEDIA = OUT / "media"
MEDIA.mkdir(parents=True, exist_ok=True)


def _load_equity(folder: str) -> pd.DataFrame:
    p = OUT / folder / "equity_curves.csv"
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df


def _plot_equity(df: pd.DataFrame, title: str, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in ["performance_v2", "baseline", "benchmark_spy", "benchmark_60_40"]:
        if col in df.columns:
            ax.plot(df.index, df[col], label=col)
    ax.set_title(title)
    ax.set_ylabel("Equity (start=1.0)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _plot_summary_table(out_png: Path) -> None:
    rows = []
    for label, folder in [
        ("2015-2026 (full)", "final_full"),
        ("2019-2021 (bull+COVID)", "final_bull"),
        ("2021-2024 (post-COVID)", "final_post_covid"),
    ]:
        met = pd.read_csv(OUT / folder / "comparison_metrics.csv")
        v2 = met.loc[met["variant"] == "performance_v2"].iloc[0]
        rows.append(
            {
                "Period": label,
                "Sharpe": float(v2["sharpe"]),
                "CAGR %": float(v2["cagr"]) * 100.0,
                "Vol %": float(v2["annual_vol"]) * 100.0,
                "MaxDD %": float(v2["max_drawdown"]) * 100.0,
            }
        )
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    tbl = ax.table(
        cellText=[
            [
                r["Period"],
                f"{r['Sharpe']:.2f}",
                f"{r['CAGR %']:.1f}%",
                f"{r['Vol %']:.1f}%",
                f"{r['MaxDD %']:.1f}%",
            ]
            for _, r in df.iterrows()
        ],
        colLabels=["Period", "Sharpe", "CAGR", "Vol", "MaxDD"],
        cellLoc="center",
        loc="center",
    )
    tbl.scale(1, 1.4)
    ax.set_title("Performance Summary (V2)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _make_gif(df: pd.DataFrame, out_gif: Path) -> None:
    frames = []
    tmp_dir = MEDIA / "_gif_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    n = len(df)
    # Keep GIF compact while still smooth.
    points = max(18, min(60, n))
    step = max(1, n // points)
    idxs = list(range(step, n + 1, step))
    if idxs[-1] != n:
        idxs.append(n)

    for j, end in enumerate(idxs):
        fig, ax = plt.subplots(figsize=(10, 5))
        chunk = df.iloc[:end]
        for col in ["performance_v2", "baseline", "benchmark_spy", "benchmark_60_40"]:
            if col in chunk.columns:
                ax.plot(chunk.index, chunk[col], label=col)
        ax.set_title(f"Equity Curve Evolution (frame {j + 1}/{len(idxs)})")
        ax.set_ylabel("Equity (start=1.0)")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fp = tmp_dir / f"frame_{j:03d}.png"
        fig.savefig(fp, dpi=120)
        plt.close(fig)
        frames.append(imageio.imread(fp))

    imageio.mimsave(out_gif, frames, duration=0.18, loop=0)


def main() -> None:
    full = _load_equity("final_full")
    bull = _load_equity("final_bull")
    post = _load_equity("final_post_covid")

    _plot_equity(full, "Equity Curves: 2015-2026", MEDIA / "equity_full_2015_2026.png")
    _plot_equity(bull, "Equity Curves: 2019-2021 (Bull + COVID)", MEDIA / "equity_bull_2019_2021.png")
    _plot_equity(post, "Equity Curves: 2021-2024 (Post-COVID)", MEDIA / "equity_post_2021_2024.png")
    _plot_summary_table(MEDIA / "performance_summary_table.png")
    _make_gif(full, MEDIA / "equity_evolution_2015_2026.gif")

    print("Generated media:")
    for p in sorted(MEDIA.glob("*")):
        if p.is_file():
            print(f" - {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

