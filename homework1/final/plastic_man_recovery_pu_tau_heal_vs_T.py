"""
τ_heal vs temperature from 12 h strength-recovery data only (no network Arrhenius).

η = σ_healed / σ₀ with σ₀ = 9.9 MPa (Jin et al., Polymers 2023).
η(t) = 1 − exp(−t/τ_heal)  →  τ_heal = −t / ln(1 − η)   at fixed t = 12 h.
"""

import numpy as np
import matplotlib.pyplot as plt

FIG_SIZE_SQUARE = (7.5, 7.5)

SIGMA_ORIGINAL_MPA = 9.9
HEAL_TIME_H = 12.0

# Measured recovered strength after 12 h heal, then tensile test
DATA_C = np.array([25.0, 40.0, 60.0])
ETA_FRACTION = np.array([
    24.7 / 100.0,
    68.1 / 100.0,
    96.2 / 100.0,
])
SIGMA_HEALED_MPA = ETA_FRACTION * SIGMA_ORIGINAL_MPA


def tau_heal_from_eta(hours: float, eta: float) -> float:
    eta = float(np.clip(eta, 1e-6, 1.0 - 1e-9))
    return -hours / np.log(1.0 - eta)


def save_tau_plot(
    x: np.ndarray,
    tau_h: np.ndarray,
    *,
    xlabel: str,
    xlim: tuple[float, float],
    outfile: str,
    annotate_x_unit: str,
) -> None:
    fig, ax = plt.subplots(figsize=FIG_SIZE_SQUARE)

    ax.plot(
        x,
        tau_h,
        "o-",
        color="#9467bd",
        linewidth=2,
        markersize=11,
        markeredgecolor="black",
        markeredgewidth=0.8,
        label=rf"$\tau_{{heal}}$ from 12 h η (σ₀ = {SIGMA_ORIGINAL_MPA} MPa)",
        zorder=3,
    )

    for xi, tc, tau, eta, sig in zip(
        x, DATA_C, tau_h, ETA_FRACTION, SIGMA_HEALED_MPA
    ):
        if annotate_x_unit == "K":
            temp_line = f"{xi:.2f} K ({tc:.0f} °C)\n"
        else:
            temp_line = f"{tc:.0f} °C\n"
        ax.annotate(
            temp_line
            + f"η = {100 * eta:.1f}%\n"
            + rf"$\sigma$ = {sig:.2f} MPa" + "\n"
            + rf"$\tau_{{heal}}$ ≈ {tau:.1f} h",
            (xi, tau),
            textcoords="offset points",
            xytext=(12, 0),
            fontsize=9,
            ha="left",
            va="center",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Healing time constant $\tau_{\mathrm{heal}}$ (h)")
    ax.set_title(
        "Recovery PU elastomer: $\\tau_{\\mathrm{heal}}$ from 12 h strength recovery"
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(0, max(tau_h) * 1.15)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper right", framealpha=0.95)

    note = (
        "Points: η = (healed σ / 9.9 MPa) after 12 h @ 25, 40, 60 °C \n (Jin et al., Fig. 4d)."
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=8.5, color="#444444")

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(outfile, dpi=160)
    plt.close(fig)


def main() -> None:
    tau_h = np.array([tau_heal_from_eta(HEAL_TIME_H, e) for e in ETA_FRACTION])
    data_k = DATA_C + 273.15

    out_c = "plastic_man_recovery_pu_tau_heal_vs_T.png"
    save_tau_plot(
        DATA_C,
        tau_h,
        xlabel="Healing temperature (°C)",
        xlim=(20, 65),
        outfile=out_c,
        annotate_x_unit="°C",
    )

    out_k = "plastic_man_recovery_pu_tau_heal_vs_TK.png"
    save_tau_plot(
        data_k,
        tau_h,
        xlabel="Healing temperature $T$ (K)",
        xlim=(data_k.min() - 10, data_k.max() + 10),
        outfile=out_k,
        annotate_x_unit="K",
    )

    print(f"σ₀ = {SIGMA_ORIGINAL_MPA} MPa, t_heal = {HEAL_TIME_H} h\n")
    print("T (°C) | T (K)   | η (%)  | σ healed (MPa) | τ_heal (h)")
    for tc, tk, eta, sig, tau in zip(
        DATA_C, data_k, ETA_FRACTION, SIGMA_HEALED_MPA, tau_h
    ):
        print(
            f"{tc:6.1f} | {tk:7.2f} | {100 * eta:6.1f} | "
            f"{sig:13.2f} | {tau:10.2f}"
        )
    print(f"\nSaved {out_c}\nSaved {out_k}")


if __name__ == "__main__":
    main()