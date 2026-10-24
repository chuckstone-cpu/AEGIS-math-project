"""Scenario + map dressing generator for apex_city.wbt.

Everything is placed on the reference path the ego car actually drives
(webots/worlds/ref_path.csv, recorded by the supervisor in a calibration run),
at an arc length s [m] and a lateral offset d [m] (d > 0 = left of the car).
That keeps every obstacle exactly in (or exactly beside) the driven lane.
"""
import csv
import math
import os

PROTO = "https://raw.githubusercontent.com/cyberbotics/webots/R2025a/projects/"
PROTOS = {
    "ToyotaPriusSimple": "vehicles/protos/toyota/ToyotaPriusSimple.proto",
    "TeslaModel3Simple": "vehicles/protos/tesla/TeslaModel3Simple.proto",
    "CitroenCZeroSimple": "vehicles/protos/citroen/CitroenCZeroSimple.proto",
    "BmwX5Simple": "vehicles/protos/bmw/BmwX5Simple.proto",
    "Pedestrian": "humans/pedestrian/protos/Pedestrian.proto",
    "TrafficCone": "objects/traffic/protos/TrafficCone.proto",
    "OilBarrel": "objects/obstacles/protos/OilBarrel.proto",
    "DivergentIndicator": "objects/traffic/protos/DivergentIndicator.proto",
    "StreetLight": "objects/traffic/protos/StreetLight.proto",
    "BusStop": "objects/street_furniture/protos/BusStop.proto",
    "Bench": "objects/street_furniture/protos/Bench.proto",
    "TrashBin": "objects/street_furniture/protos/TrashBin.proto",
    "Pine": "objects/trees/protos/Pine.proto",
    "Oak": "objects/trees/protos/Oak.proto",
    "Fountain": "objects/street_furniture/protos/Fountain.proto",
    "ElectricalCabinet": "objects/street_furniture/protos/ElectricalCabinet.proto",
    "CautionSign": "objects/traffic/protos/CautionSign.proto",
}


class RefPath:
    def __init__(self, path_csv, step=1.0):
        pts = []
        with open(path_csv, newline="") as f:
            for r in csv.DictReader(f):
                p = (float(r["x"]), float(r["y"]))
                if not pts or math.dist(p, pts[-1]) >= step:
                    pts.append(p)
        self.pts = pts
        self.s = [0.0]
        for a, b in zip(pts, pts[1:]):
            self.s.append(self.s[-1] + math.dist(a, b))
        self.length = self.s[-1]

    def pose(self, s, d=0.0):
        """(x, y, yaw) at arc length s, shifted d metres to the left."""
        s = max(0.0, min(self.length, s))
        k = 0
        while k < len(self.s) - 2 and self.s[k + 1] < s:
            k += 1
        (x0, y0), (x1, y1) = self.pts[k], self.pts[k + 1]
        seg = max(1e-9, self.s[k + 1] - self.s[k])
        u = (s - self.s[k]) / seg
        # smooth heading over +-4 m so GPS noise does not twist objects
        a = self.pts[max(0, k - 4)]
        b = self.pts[min(len(self.pts) - 1, k + 5)]
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        x, y = x0 + u * (x1 - x0), y0 + u * (y1 - y0)
        return x - d * math.sin(yaw), y + d * math.cos(yaw), yaw

    def polyline(self, s0, s1, d0=0.0, d1=None, step=2.0):
        d1 = d0 if d1 is None else d1
        out, s = [], s0
        while s <= s1 + 1e-6:
            u = (s - s0) / max(1e-9, s1 - s0)
            # smoothstep lateral blend for a natural lane change
            d = d0 + (d1 - d0) * (3 * u * u - 2 * u * u * u)
            out.append(self.pose(s, d)[:2])
            s += step
        return out


def node(proto, x, y, yaw, z=0.0, **fields):
    body = "".join(f"\n  {k} {v}" for k, v in fields.items())
    return (f"{proto} {{\n  translation {x:.3f} {y:.3f} {z:.3f}\n"
            f"  rotation 0 0 1 {yaw:.4f}{body}\n}}")


def build(path_csv, plan):
    """Return (world_text, used_protos, actor_lines)."""
    P = RefPath(path_csv)
    out, used, actors = [], set(), []

    def add(proto, text):
        used.add(proto)
        out.append(text)

    # ---------------- S3: pedestrian on the zebra crossing ahead of start ------
    cx, cy = plan["crosswalk"]
    add("Pedestrian", node(
        "Pedestrian", cx, cy - 7.0, math.pi / 2, z=1.27,
        name='"pedestrian"', controllerArgs=f'[ "--trajectory={cx:.2f} {cy - 7.0:.2f}, '
        f'{cx:.2f} {cy + 9.0:.2f}" "--speed=1.1" ]',
        shirtColor="0.8 0.1 0.1"))

    # ---------------- S1: stalled car in the ego lane -------------------------
    x, y, a = P.pose(plan["stalled_car_s"])
    add("ToyotaPriusSimple", "DEF OBST_STALLED_CAR " + node(
        "ToyotaPriusSimple", x, y, a, z=0.4, name='"stalled car"', color="0.75 0.75 0.78"))
    x, y, a = P.pose(plan["stalled_car_s"] - 12, -3.0)
    add("CautionSign", node("CautionSign", x, y, a + math.pi / 2, name='"warning sign"'))

    # ---------------- S2: slow lead car (kinematic actor) ---------------------
    s0, s1 = plan["lead_car_s"]
    x, y, a = P.pose(s0)
    add("TeslaModel3Simple", "DEF OBST_LEAD_CAR " + node(
        "TeslaModel3Simple", x, y, a, z=0.4, name='"lead car"', color="0.1 0.3 0.75"))
    tx, ty, _ = P.pose(s0 - 45)
    pts = P.polyline(s0, s1)
    actors.append(f"OBST_LEAD_CAR {tx:.2f} {ty:.2f} 8 {plan['lead_speed_kmh']} " +
                  " ".join(f"{px:.2f} {py:.2f}" for px, py in pts))

    # ---------------- S5: cut-in from the adjacent (left) lane ---------------
    s0, s1 = plan["cut_in_s"]
    lane = plan["lane_width"]
    lane = -lane  # same-direction lane is on the RIGHT of the ego car
    x, y, a = P.pose(s0, lane)
    add("CitroenCZeroSimple", "DEF OBST_CUT_IN " + node(
        "CitroenCZeroSimple", x, y, a, z=0.4, name='"cut-in car"', color="0.1 0.55 0.25"))
    tx, ty, _ = P.pose(s0 - 25)
    pts = (P.polyline(s0, s0 + 14, lane, lane) + P.polyline(s0 + 16, s0 + 36, lane, 0.0)
           + P.polyline(s0 + 38, s1, 0.0))
    actors.append(f"OBST_CUT_IN {tx:.2f} {ty:.2f} 8 {plan['cut_in_speed_kmh']} " +
                  " ".join(f"{px:.2f} {py:.2f}" for px, py in pts))

    # ---------------- S4: cone lane closure (taper) + barrels -----------------
    s_c = plan["cones_s"]
    cones = []
    for i in range(6):  # taper from the lane edge across the lane
        x, y, a = P.pose(s_c + 2.2 * i, -1.9 + 0.75 * i)
        cones.append(node("TrafficCone", x - 0, y, a, name=f'"cone {i}"'))
    x, y, a = P.pose(s_c + 13, 0.0)
    cones.append(node("OilBarrel", x, y, a, z=0.44, name='"barrel 0"'))
    x, y, a = P.pose(s_c + 13, 1.4)
    cones.append(node("OilBarrel", x, y, a, z=0.44, name='"barrel 1"'))
    used.update({"TrafficCone", "OilBarrel"})
    x, y, a = P.pose(s_c, 0.0)
    out.append("DEF OBST_ROADWORKS Pose {\n  translation %.3f %.3f 0\n  children [\n%s\n  ]\n}"
               % (x, y, "\n".join(_relative(c, x, y) for c in cones)))
    x, y, a = P.pose(s_c - 10, -3.2)
    add("DivergentIndicator", node("DivergentIndicator", x, y, a + math.pi / 2,
                                   name='"divert sign"'))

    # ---------------- parked cars in the adjacent right lane (must be ignored)
    for i, s in enumerate(plan["parked_s"]):
        x, y, a = P.pose(s, -plan["parked_offset"])
        proto = ["BmwX5Simple", "ToyotaPriusSimple", "TeslaModel3Simple"][i % 3]
        col = ["0.15 0.15 0.15", "0.6 0.05 0.05", "0.9 0.9 0.9"][i % 3]
        add(proto, node(proto, x, y, a, z=0.4, name=f'"parked {i}"', color=col))

    # ---------------- map dressing along the route ----------------------------
    s = 20.0
    i = 0
    while s < P.length - 10:
        x, y, a = P.pose(s, -plan["curb_offset"])
        add("StreetLight", node("StreetLight", x, y, a - math.pi / 2, name=f'"street light {i}"'))
        x, y, a = P.pose(s + 18, -plan["curb_offset"] - 3.0)
        tree = "Pine" if i % 2 else "Oak"
        add(tree, node(tree, x, y, a, name=f'"route tree {i}"'))
        s += 45.0
        i += 1
    for k, s in enumerate(plan["bus_stops_s"]):
        x, y, a = P.pose(s, -plan["curb_offset"] - 1.5)
        add("BusStop", node("BusStop", x, y, a + math.pi, name=f'"bus stop {k}"'))
        x, y, a = P.pose(s + 6, -plan["curb_offset"] - 1.5)
        add("Bench", node("Bench", x, y, a + math.pi, name=f'"bench {k}"'))
        x, y, a = P.pose(s + 9, -plan["curb_offset"] - 1.0)
        add("TrashBin", node("TrashBin", x, y, a, name=f'"bin {k}"'))
    if plan.get("fountain"):
        fx, fy = plan["fountain"]
        add("Fountain", node("Fountain", fx, fy, 0.0, name='"fountain"'))

    return "\n".join(out), sorted(used), actors


def _relative(text, ox, oy):
    """Rewrite a node's world translation as relative to (ox, oy)."""
    head, rest = text.split("translation ", 1)
    nums, tail = rest.split("\n", 1)
    x, y, z = (float(v) for v in nums.split())
    return f"{head}translation {x - ox:.3f} {y - oy:.3f} {z:.3f}\n{tail}"


def proto_decls(used, world_text):
    lines = []
    for p in used:
        url = PROTO + PROTOS[p]
        if url not in world_text:
            lines.append(f'EXTERNPROTO "{url}"\n')
    return "".join(lines)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    print(RefPath(os.path.join(here, "..", "worlds", "ref_path.csv")).length)
