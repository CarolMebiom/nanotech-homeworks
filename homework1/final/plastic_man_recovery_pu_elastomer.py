"""
PU elastomer healing efficiency vs time toward full recovery.

η(t) = 100(1 − e^(−t/τ_heal)); τ_heal from 12 h Jin et al. data @ 25, 40, 60 °C.
"""

import numpy as np
import matplotlib.pyplot as plt

from plastic_man_recovery_pu_tau_heal_vs_T import (
    DATA_C,
    ETA_FRACTION,
    FIG_SIZE_SQUARE,
    HEAL_TIME_H,
    SIGMA_ORIGINAL_MPA,
    tau_heal_from_eta,
)

HEAL_12H_H = HEAL_TIME_H
TIME_H = np.linspace(0, 48, 400)

TAU_HEAL_DATA_H = {
    float(t): tau_heal_from_eta(HEAL_TIME_H, float(e))
    for t, e in zip(DATA_C, ETA_FRACTION)
}

TEMP_SERIES = (
    (25.0, "#1f77b4"),
    (40.0, "#2ca02c"),
    (60.0, "#d62728"),
)


def healing_efficiency_pct(hours: np.ndarray, tau_h: float) -> np.ndarray:
    h = np.asarray(hours, dtype=float)
    return 100.0 * (1.0 - np.exp(-h / tau_h))


def main() -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE_SQUARE)

    for temp, color in TEMP_SERIES:
        tau = TAU_HEAL_DATA_H[temp]
        eta = healing_efficiency_pct(TIME_H, tau)
        eta_frac = float(dict(zip(DATA_C, ETA_FRACTION, strict=True))[temp])
        eta_meas = eta_frac * 100.0

        ax.plot(
            TIME_H,
            eta,
            color=color,
            linewidth=2,
            label=rf"{temp:.0f} °C ($\tau_{{heal}}$ ≈ {tau:.1f} h)",
        )
        ax.scatter(
            [HEAL_12H_H],
            [eta_meas],
            s=140,
            facecolors="none",
            edgecolors=color,
            linewidths=2.5,
            zorder=5,
        )

    ax.axhline(
        100.0,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
        label=rf"Full recovery (σ₀ = {SIGMA_ORIGINAL_MPA} MPa)",
    )

    ax.set_xlim(0, 48)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Healing time after cut and rejoin (h)")
    ax.set_ylabel(
        f"Healing efficiency η (%)\n"
        f"(recovered σ = η × {SIGMA_ORIGINAL_MPA} MPa)"
    )
    ax.set_title("PU elastomer: healing toward original strength")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="center right", framealpha=0.95)

    note = (
        "Open circles @ 12 h: Jin et al. (2023) Fig. 4d. "
        "Curves: τ_heal inverted from 12 h data @ 25, 40, 60 °C."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=8.5, color="#444444")
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out = "plastic_man_pu_elastomer_healing.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()