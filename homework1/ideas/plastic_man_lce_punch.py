"""
Plastic Man arm kinematics: polydomain LCE vs elite fighter.
Length vs time for extension and shape recovery.
"""

import numpy as np
import matplotlib.pyplot as plt

# Geometry (cm)
L_REST = 106.5
L_MAX = 340.8
DELTA = L_MAX - L_REST

# Shared axes for LCE and elite-fighter panels
Y_LIM_MIN = 50
Y_LIM_MAX = L_MAX * 1.05
Y_LIM_MAX_ELITE = 115  # shared ymin; zoomed ymax so 50→100 cm punch is visible

# Extension at constant rate (user scenarios)
T_EXT_FAST = 22.0    # s, 5 MPa, ~0.1 s^-1 strain rate regime
T_EXT_SLOW = 220.0   # s, 3 MPa

# Shape recovery: gamma -> 0 at these times (NLCE1 / ILCE1, T = 25 C)
T_REC_FAST = 1.7e3   # s
T_REC_SLOW = 6.0e3   # s

# Elite athlete punch (flexed guard → extended arm, human scale)
L_FLEXED = 50.0  # cm, bent-arm guard
L_ATHLETE_EXT = 100.0  # cm, full extension at impact
V_ATHLETE = 9.14  # m/s
V_ATHLETE_CM = V_ATHLETE * 100  # cm/s
T_GUARD = 0.04  # s, brief guard before extension
DELTA_PUNCH = L_ATHLETE_EXT - L_FLEXED
T_PUNCH_EXT = DELTA_PUNCH / V_ATHLETE_CM
T_PUNCH_ATHLETE = T_GUARD + T_PUNCH_EXT


def linear_extension(t, t_start, duration):
    """Length during constant-rate stretch."""
    u = np.clip((t - t_start) / duration, 0.0, 1.0)
    return L_REST + DELTA * u


def exponential_recovery(t, t_start, tau):
    """Return from L_MAX toward L_REST (strain-like exponential decay)."""
    dt = np.maximum(t - t_start, 0.0)
    return L_REST + DELTA * np.exp(-dt / tau)


def recovery_time_constant(t_to_zero, fraction=1e-2):
    """tau so remaining deformation is ~1% of initial at t_to_zero."""
    return t_to_zero / (-np.log(fraction))


def build_lce_cycle(t_ext, t_rec_char, t_total, n=4000):
    """Extension then exponential recovery."""
    tau = recovery_time_constant(t_rec_char)
    t = np.linspace(0, t_total, n)
    L = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti <= t_ext:
            L[i] = linear_extension(ti, 0.0, t_ext)
        else:
            L[i] = exponential_recovery(ti, t_ext, tau)
    return t, L, tau


def build_athlete_punch(t_after=0.05, n=200):
    """Flexed guard, then constant hand speed to full reach; return to guard."""
    t_guard = np.linspace(0, T_GUARD, 25)
    L_guard = np.full_like(t_guard, L_FLEXED)

    t_up = np.linspace(T_GUARD, T_GUARD + T_PUNCH_EXT, n)
    L_up = L_FLEXED + V_ATHLETE_CM * (t_up - T_GUARD)

    t_end_ext = T_GUARD + T_PUNCH_EXT
    t_dn = np.linspace(t_end_ext, t_end_ext + t_after, 50)
    L_dn = np.linspace(L_ATHLETE_EXT, L_FLEXED, len(t_dn))

    t = np.concatenate([t_guard, t_up[1:], t_dn[1:]])
    L = np.concatenate([L_guard, L_up[1:], L_dn[1:]])
    return t, L


def plot_lce_panel():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "legend.fontsize": 9,
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    scenarios = [
        (T_EXT_FAST, T_REC_FAST, "Fast stretch (22 s, 5 MPa) + NLCE recovery (~1.7×10³ s)", "#1f77b4", "-"),
        (T_EXT_FAST, T_REC_SLOW, "Fast stretch (22 s, 5 MPa) + ILCE recovery (~6×10³ s)", "#1f77b4", "--"),
        (T_EXT_SLOW, T_REC_FAST, "Slow stretch (220 s, 3 MPa) + NLCE recovery", "#d62728", "-"),
        (T_EXT_SLOW, T_REC_SLOW, "Slow stretch (220 s, 3 MPa) + ILCE recovery", "#d62728", "--"),
    ]

    t_max = 7500
    for t_ext, t_rec, label, color, ls in scenarios:
        t, L, _ = build_lce_cycle(t_ext, t_rec, t_max)
        ax.plot(t, L, color=color, linestyle=ls, linewidth=2, label=label)

    ax.axhline(L_REST, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.axhline(L_MAX, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(t_max * 0.02, L_REST + 8, f"Rest length = {L_REST} cm", color="gray", fontsize=9)
    ax.text(t_max * 0.02, L_MAX - 15, f"λ = 3.2 → {L_MAX} cm", color="gray", fontsize=9)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Arm length (cm)")
    ax.set_title("Plastic Man (polydomain LCE) — extension and shape recovery")
    ax.set_xlim(0, t_max)
    ax.set_ylim(Y_LIM_MIN, Y_LIM_MAX)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", framealpha=0.95)

    note = (
        f"LCE peak extension velocity (fast case): {DELTA/T_EXT_FAST:.1f} cm/s "
        f"≈ {DELTA/T_EXT_FAST/100:.2f} m/s"
    )
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="#333333")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = "/home/carolinda/Desktop/nn/homework1/plastic_man_length_vs_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_elite_fighter_panel():
    fig, ax = plt.subplots(figsize=(6, 4))

    t_a, L_a = build_athlete_punch()
    ax.plot(t_a, L_a, color="#2ca02c", linewidth=2.5, label=f"Hand speed {V_ATHLETE} m/s")
    ax.axhline(L_FLEXED, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.axhline(L_ATHLETE_EXT, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    ax.text(0.01, L_FLEXED + 2, f"Flexed ≈ {L_FLEXED:.0f} cm", color="gray", fontsize=8)
    ax.text(0.01, L_ATHLETE_EXT + 2, f"Extended ≈ {L_ATHLETE_EXT:.0f} cm", color="gray", fontsize=8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Arm length (cm)")
    ax.set_title(
        f"Elite fighter — flexed to extended in ≈ {T_PUNCH_EXT*1000:.0f} ms"
    )
    ax.set_xlim(0, max(t_a) * 1.1)
    ax.set_ylim(Y_LIM_MIN, Y_LIM_MAX_ELITE)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    note = "F ≈ 3427 N on ~5 cm² hand area (for pressure comparison)"
    fig.text(0.5, 0.02, note, ha="center", fontsize=9, color="#333333")

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = "/home/carolinda/Desktop/nn/homework1/elite_fighter_length_vs_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    lce_out = plot_lce_panel()
    athlete_out = plot_elite_fighter_panel()
    print(f"Saved: {lce_out}")
    print(f"Saved: {athlete_out}")
    print(f"Athlete extension (flexed → max): {T_PUNCH_EXT:.4f} s")
    print(f"LCE fast extension mean velocity: {DELTA/T_EXT_FAST/100:.3f} m/s")


if __name__ == "__main__":
    main()