"""
Plastic Man — PU elastomer arm cut and self-healing (regeneration only).

Healing efficiency = recovered tensile strength / original strength (9.9 MPa).
Data from Jin et al., Polymers 2023, 15, 4020 (Fig. 4d); specimens cut,
rejoined, then healed at fixed temperature before mechanical retest.
"""

import numpy as np
import matplotlib.pyplot as plt

# Original specimen (Fig. 4a, Table S2)
SIGMA_ORIGINAL_MPA = 9.9

# Measured healing efficiency (%), strength recovery relative to virgin specimen
# 25 °C: 24.7% at 12 h; longer healing → 43.2% (paper §2.4)
HEAL_25C_MEASURED_H = np.array([0.0, 12.0, 24.0])
HEAL_25C_MEASURED_PCT = np.array([0.0, 24.7, 43.2])

# 60 °C: 96.2% at 12 h (paper §2.4)
HEAL_60C_MEASURED_H = np.array([0.0, 12.0])
HEAL_60C_MEASURED_PCT = np.array([0.0, 96.2])

# Evaluation times for the plot (h)
HEALING_TIMES_H = np.arange(0, 25, 4, dtype=float)


def healing_efficiency_25c(hours: np.ndarray) -> np.ndarray:
    """Piecewise linear between reported 25 °C points."""
    return np.interp(hours, HEAL_25C_MEASURED_H, HEAL_25C_MEASURED_PCT)


def healing_efficiency_60c(hours: np.ndarray) -> np.ndarray:
    """
    Saturating kinetics fitted to 12 h @ 60 °C (96.2%).
    η(t) = 100 (1 − exp(−t/τ)); τ from the 12 h anchor.
    """
    eta_12 = HEAL_60C_MEASURED_PCT[1] / 100.0
    tau_h = 12.0 / np.log(1.0 / (1.0 - eta_12))
    return 100.0 * (1.0 - np.exp(-hours / tau_h))


def strength_recovered_mpa(efficiency_pct: np.ndarray) -> np.ndarray:
    return (efficiency_pct / 100.0) * SIGMA_ORIGINAL_MPA


def main() -> None:
    eta_25 = healing_efficiency_25c(HEALING_TIMES_H)
    eta_60 = healing_efficiency_60c(HEALING_TIMES_H)
    sigma_25 = strength_recovered_mpa(eta_25)
    sigma_60 = strength_recovered_mpa(eta_60)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(
        HEALING_TIMES_H,
        eta_25,
        "o-",
        color="#1f77b4",
        linewidth=2,
        markersize=8,
        label="25 °C",
    )
    ax.plot(
        HEALING_TIMES_H,
        eta_60,
        "s-",
        color="#d62728",
        linewidth=2,
        markersize=8,
        label="60 °C",
    )

    # Highlight directly reported efficiencies
    ax.scatter(
        [12.0],
        [24.7],
        s=140,
        facecolors="none",
        edgecolors="#1f77b4",
        linewidths=2.5,
        zorder=5,
    )
    ax.scatter(
        [24.0],
        [43.2],
        s=140,
        facecolors="none",
        edgecolors="#1f77b4",
        linewidths=2.5,
        zorder=5,
    )
    ax.scatter(
        [12.0],
        [96.2],
        s=140,
        facecolors="none",
        edgecolors="#d62728",
        linewidths=2.5,
        zorder=5,
    )

    ax.annotate(
        "2.45 MPa",
        (12, 24.7),
        textcoords="offset points",
        xytext=(-58, 10),
        fontsize=9,
        color="#1f77b4",
    )
    ax.annotate(
        "9.52 MPa",
        (12, 96.2),
        textcoords="offset points",
        xytext=(8, -18),
        fontsize=9,
        color="#d62728",
    )

    ax.set_xlim(-0.5, 24.5)
    ax.set_ylim(0, 105)
    ax.set_xticks(HEALING_TIMES_H)
    ax.set_xlabel("Healing time after cut and rejoin (h)")
    ax.set_ylabel(
        f"Healing efficiency as strength recovery (%)\n"
        f"(recovered σ = η × {SIGMA_ORIGINAL_MPA} MPa)"
    )
    ax.set_title(
        "Plastic Man arm (PU elastomer): strength recovery while healing"
    )
    ax.grid(True, alpha=0.35)
    ax.legend(loc="center right", framealpha=0.95)

    note = (
        "Big circles: measured efficiencies (25 °C: 12 h, 24 h; 60 °C: 12 h) - Jin et al., Polymers 2023 \n"
        "Other points: linear interpolation (25 °C) or exp. saturation through 12 h @ 60 °C."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=8.5, color="#444444")

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = "plastic_man_pu_elastomer_healing.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)

    print("Healing time (h) | η @ 25 °C (%) | σ @ 25 °C (MPa) | η @ 60 °C (%) | σ @ 60 °C (MPa)")
    for t, e25, s25, e60, s60 in zip(HEALING_TIMES_H, eta_25, sigma_25, eta_60, sigma_60):
        print(f"{t:14.0f} | {e25:13.2f} | {s25:15.2f} | {e60:13.2f} | {s60:15.2f}")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()