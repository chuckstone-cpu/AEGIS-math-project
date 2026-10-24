"""Generate webots/worlds/apex_city.wbt from Webots' stock city.wbt.

Every change to the stock world is applied here, so the world is reproducible:
    python webots/tools/build_world.py [--stage calib|full] [--duration S]
"""
import argparse
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WEBOTS_HOME = os.environ.get(
    "WEBOTS_HOME", os.path.expanduser(r"~\AppData\Local\Programs\Webots"))
STOCK = os.path.join(WEBOTS_HOME, "projects", "vehicles", "worlds", "city.wbt")
PROTO = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/"


def axis_angle_from_yaw_pitch(yaw, pitch):
    """Viewpoint orientation (axis-angle) looking along `yaw`, tilted down by `pitch`."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    # R = Rz(yaw) * Ry(pitch)
    R = [[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0.0, cp]]
    angle = math.acos(max(-1.0, min(1.0, (R[0][0] + R[1][1] + R[2][2] - 1) / 2)))
    if abs(angle) < 1e-9:
        return (0, 0, 1, 0)
    if abs(angle - math.pi) < 1e-6:
        # axis from the diagonal for a 180 degree rotation
        x = math.sqrt(max(0, (R[0][0] + 1) / 2))
        y = math.sqrt(max(0, (R[1][1] + 1) / 2))
        z = math.sqrt(max(0, (R[2][2] + 1) / 2))
        y = math.copysign(y, R[0][1] if x else 1)
        z = math.copysign(z, R[0][2] if x else R[1][2])
        return (x, y, z, angle)
    s = 2 * math.sin(angle)
    return ((R[2][1] - R[1][2]) / s, (R[0][2] - R[2][0]) / s, (R[1][0] - R[0][1]) / s, angle)


def replace_block(text, header, new_block):
    """Replace the top-level node starting with `header` (e.g. 'Viewpoint {')."""
    start = text.index(header)
    depth, i = 0, start
    while True:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[:start] + new_block + text[i + 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="full", choices=["calib", "full"])
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--movie", default="")
    ap.add_argument("--run", default="run", help="output subfolder under webots/output")
    ap.add_argument("--start-s", type=float, default=0.0, help="spawn ego at this arc length")
    ap.add_argument("--out", default=os.path.join(ROOT, "worlds", "apex_city.wbt"))
    ap.add_argument("--ego-record", action="store_true",
                    help="let the ego controller record its dashboard video")
    args = ap.parse_args()

    w = open(STOCK, encoding="utf-8").read()

    # --- physics / timing: 16 ms step (sensors and controller run at 16 ms) ---
    w = w.replace("basicTimeStep 10", "basicTimeStep 16")

    # --- smooth chase camera: no rigid follow, so physics wobble is filtered ---
    ego_x, ego_y = -45.0, 45.88
    ax = axis_angle_from_yaw_pitch(math.pi, 0.32)
    w = replace_block(w, "Viewpoint {", f"""Viewpoint {{
  orientation {ax[0]:.6f} {ax[1]:.6f} {ax[2]:.6f} {ax[3]:.6f}
  position {ego_x + 13.0:.2f} {ego_y:.2f} 5.5
  near 1
  follow "vehicle"
  followType "Mounted Shot"
  followSmoothness 0.85
}}""")

    # --- ego car: our controller, 4:3 camera, GPS kept ---
    out_dir = "../../output/" + args.run
    ego_args = ['"--log"', f'"{out_dir}/ego_engine.csv"', '"--ui-every"', '"2"']
    if args.ego_record:
        ego_args = ['"--log"', f'"{out_dir}/ego_engine.csv"', '"--ui-every"', '"6"',
                    '"--record-jpg"', f'"{out_dir}/dash"', '"--dump-scans"', f'"{out_dir}/scans"']
    car_hdr = w.index("BmwX5 {")
    w = w[:car_hdr] + "DEF EGO " + w[car_hdr:]
    if args.start_s > 0:  # spawn the ego further along the reference path
        import scenarios
        x, y, a = scenarios.RefPath(os.path.join(ROOT, "worlds", "ref_path.csv")).pose(args.start_s)
        w = w.replace("DEF EGO BmwX5 {\n  translation -45 45.88 0.4\n  rotation 0 0 1 3.14159\n",
                      f"DEF EGO BmwX5 {{\n  translation {x:.3f} {y:.3f} 0.4\n"
                      f"  rotation 0 0 1 {a:.5f}\n", 1)
        assert f"translation {x:.3f}" in w
    w = w.replace('controller "autonomous_vehicle"',
                  'name "vehicle"\n  controller "apex_engine"\n'
                  f'  controllerArgs [ {" ".join(ego_args)} ]', 1)

    # drop the stock sample's on-car Display (it shows as a black overlay)
    w = re.sub(r"\n    Display \{\n      width 200\n      height 150\n    \}", "", w, count=1)

    # mount the SICK low and level (0.40 m above the road instead of 0.85 m):
    # a 0.7 m traffic cone or a 0.88 m barrel is otherwise ABOVE/BELOW nothing -
    # it simply is not in the scan plane. Ground strikes caused by body pitch
    # are rejected in software (lateral-extent test in apex_engine.cpp).
    sick = "SickLms291 {\n      translation 0.06 0 0\n"
    assert sick in w
    # x +0.25 m: in FRONT of the bumper mesh (inside it, every forward ray hits
    # the car's own body); z -0.42 m: 0.43 m above the road
    w = w.replace(sick, "SickLms291 {\n      translation 0.25 0 -0.42\n", 1)

    # 256x128 camera: same 2:1 geometry the lane calibration was made for,
    # twice the resolution of the stock 128x64
    cam = "      fieldOfView 1\n      width 128\n"
    w = w.replace(cam, "      fieldOfView 1\n      width 256\n      height 128\n", 1)
    assert "width 256" in w

    # the stock world already has 4 oil barrels standing in the lanes (Webots'
    # own avoidance demo): make them stop & go obstacles for the supervisor
    k = 0

    def _barrel(m):
        nonlocal k
        k += 1
        return f"\nDEF OBST_BARREL_{k} OilBarrel {{"
    w = re.sub(r"\nOilBarrel \{", _barrel, w)

    extra = []
    sup_args = ['"--log"', f'"{out_dir}/supervisor.csv"', '"--clear-after"', '"4"']
    if args.duration > 0:
        sup_args += ['"--duration"', f'"{args.duration:g}"']
    if args.movie:
        sup_args += ['"--movie"', f'"{out_dir}/{args.movie}"']
    if args.stage == "full":
        sup_args += ['"--actors"', '"../../worlds/actors.txt"']
    extra.append(f"""Robot {{
  name "scenario_supervisor"
  controller "scenario_supervisor"
  controllerArgs [ {" ".join(sup_args)} ]
  supervisor TRUE
}}""")

    if args.stage == "full":
        import json
        import scenarios
        plan = json.load(open(os.path.join(HERE, "scenario_plan.json"), encoding="utf-8"))
        text, used, actors = scenarios.build(os.path.join(ROOT, "worlds", "ref_path.csv"), plan)
        extra.append(text)
        with open(os.path.join(ROOT, "worlds", "actors.txt"), "w", encoding="utf-8") as f:
            f.write("# DEF trigger_x trigger_y radius speed_kmh x1 y1 x2 y2 ...\n")
            f.write("\n".join(actors) + "\n")
        first = w.index("\n", w.index("EXTERNPROTO")) + 1
        w = w[:first] + scenarios.proto_decls(used, w) + w[first:]

    w = w.rstrip() + "\n" + "\n".join(extra) + "\n"
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write(w)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
