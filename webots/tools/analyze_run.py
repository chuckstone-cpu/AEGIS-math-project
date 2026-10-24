"""Turn a Webots run's logs into a report figure + metrics.

    python webots/tools/analyze_run.py webots/output [title]
Reads ego_engine.csv (controller) and supervisor.csv (ground truth pose,
obstacle-clear events); writes run_report.png and metrics.txt.
"""
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for k in rows[0].keys():
        col = []
        for r in rows:
            try:
                col.append(float(r[k]))
            except (TypeError, ValueError):
                col.append(r[k])
        out[k] = col
    return out


def main():
    d = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "APEX in Webots"
    e = load(os.path.join(d, "ego_engine.csv"))
    s = load(os.path.join(d, "supervisor.csv"))
    t = e["t"]
    dt = t[1] - t[0]
    steer = e["steer"]
    accel_cmd = e["accel"]

    # metrics
    tv_per_min = sum(abs(steer[i] - steer[i - 1]) for i in range(1, len(steer))) / (t[-1] / 60.0)
    v = e["speed_kmh"]
    jerk = [abs(accel_cmd[i] - accel_cmd[i - 1]) / dt for i in range(1, len(accel_cmd))]
    jerk_sorted = sorted(jerk)
    jerk_p99 = jerk_sorted[int(0.99 * (len(jerk_sorted) - 1))]
    gaps = [g for g, h in zip(e["gap"], e["has_lead"]) if h > 0]
    clears = [ev for ev in s["event"] if isinstance(ev, str) and ev.startswith("clear")]
    lane_ok = sum(1 for x in e["lane_ok"] if x > 0) / len(t)
    conv = sum(1 for x in e["admm_conv"] if x > 0) / len(t)
    dist = sum(math.hypot(s["x"][i] - s["x"][i - 1], s["y"][i] - s["y"][i - 1])
               for i in range(1, len(s["x"])))
    m = {
        "sim time [s]": t[-1],
        "distance driven [m]": dist,
        "mean / max speed [km/h]": f"{sum(v) / len(v):.1f} / {max(v):.1f}",
        "steering total variation [rad/min]": tv_per_min,
        "commanded jerk p99 [m/s^3]": jerk_p99,
        "min lidar gap [m]": min(gaps) if gaps else float("nan"),
        "obstacles handled (stop & go)": len(clears),
        "frames with lane fit": f"{100 * lane_ok:.1f} %",
        "ADMM converged frames": f"{100 * conv:.1f} %",
        "maths time mean / max [ms]": f"{sum(e['math_ms']) / len(t):.2f} / {max(e['math_ms']):.2f}",
    }
    with open(os.path.join(d, "metrics.txt"), "w") as f:
        for k, val in m.items():
            line = f"{k:38s} {val:.3f}" if isinstance(val, float) else f"{k:38s} {val}"
            print(line)
            f.write(line + "\n")

    # figure
    fig = plt.figure(figsize=(15, 9), facecolor="#16120f")
    plt.rcParams.update({"text.color": "#e8e8e8", "axes.labelcolor": "#bbbbbb",
                         "xtick.color": "#999999", "ytick.color": "#999999"})
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 1.4], hspace=0.45, wspace=0.18)

    def style(ax, ttl):
        ax.set_facecolor("#1f1a16")
        ax.set_title(ttl, color="#e8e8e8", fontsize=11, loc="left")
        for sp in ax.spines.values():
            sp.set_color("#444")
        ax.grid(color="#333", lw=0.6)

    ax = fig.add_subplot(gs[:, 0])
    style(ax, "Trajectory (supervisor ground truth)")
    sc = ax.scatter(s["x"], s["y"], c=s["speed_kmh"], cmap="viridis", s=4)
    for i, ev in enumerate(s["event"]):
        if isinstance(ev, str) and ev.startswith("clear"):
            ax.plot(s["x"][i], s["y"][i], "o", color="#ff5050", ms=8)
            ax.annotate(ev.split()[1].replace("OBST_", ""), (s["x"][i], s["y"][i]),
                        color="#ff8080", fontsize=8, xytext=(5, 5), textcoords="offset points")
    ax.plot(s["x"][0], s["y"][0], "s", color="white", ms=7)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    cb = fig.colorbar(sc, ax=ax, fraction=0.04)
    cb.set_label("speed [km/h]")

    ax = fig.add_subplot(gs[0, 1])
    style(ax, "Speed: measured vs ADMM command")
    ax.plot(t, v, color="#5ad06e", lw=1.4, label="measured")
    ax.plot(t, e["cmd_kmh"], color="#3cdcff", lw=1, alpha=0.8, label="ADMM command")
    for i, h in enumerate(e["brake"]):
        pass
    ax.fill_between(t, 0, [b * 40 for b in e["brake"]], color="#ff5050", alpha=0.35, label="brake x40")
    ax.set_ylabel("km/h")
    ax.legend(loc="upper right", fontsize=8, facecolor="#1f1a16", edgecolor="#444")

    ax = fig.add_subplot(gs[1, 1])
    style(ax, "Obstacle gap (lidar) and MDP action")
    ax.plot(t, [g if h else math.nan for g, h in zip(e["gap"], e["has_lead"])],
            color="#ff9a3c", lw=1.2, label="gap [m]")
    ax.set_ylabel("gap [m]")
    ax.set_ylim(0, 46)
    ax2 = ax.twinx()
    ax2.step(t, e["action"], color="#dd88dd", lw=0.8, where="post")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["ACCEL", "HOLD", "BRAKE"], color="#dd88dd", fontsize=8)
    ax2.set_ylim(-0.5, 2.5)

    ax = fig.add_subplot(gs[2, 1])
    style(ax, f"Steering command (total variation {tv_per_min:.2f} rad/min)")
    ax.plot(t, e["raw_steer"], color="#7070c0", lw=0.6, label="raw steer law")
    ax.plot(t, steer, color="#3cdcff", lw=1.3, label="filtered + rate-limited")
    ax.set_xlabel("simulation time [s]")
    ax.set_ylabel("rad")
    ax.legend(loc="upper right", fontsize=8, facecolor="#1f1a16", edgecolor="#444")

    fig.suptitle(title, color="#3cdcff", fontsize=15, x=0.02, ha="left")
    fig.savefig(os.path.join(d, "run_report.png"), dpi=110, facecolor=fig.get_facecolor())
    print("wrote", os.path.join(d, "run_report.png"))


if __name__ == "__main__":
    main()
