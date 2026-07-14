"""GEB benchmark — program voice. sample: sym_c4_0005 (symmetry_lab / rotation_order)"""

import math

POINTS = [(0, 0), (2, 0), (2, -1), (4, 1), (2, 3), (2, 2), (0, 2)]
ANGLE_DEG = 90.0
EXPECTED_ORDER = 4


def rotate(points, angle_deg, center=(0.0, 0.0)):
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    cx, cy = center
    return [(cx + ca * (x - cx) - sa * (y - cy),
             cy + sa * (x - cx) + ca * (y - cy)) for x, y in points]


def close(ps, qs, tol=1e-9):
    return all(abs(px - qx) < tol and abs(py - qy) < tol
               for (px, py), (qx, qy) in zip(ps, qs))


def rotation_order(points, angle_deg, max_k=360):
    pts = points
    for k in range(1, max_k + 1):
        pts = rotate(pts, angle_deg)
        if close(pts, points):
            return k
    raise ValueError("no finite order found")


if __name__ == "__main__":
    result = rotation_order(POINTS, ANGLE_DEG)
    assert result == EXPECTED_ORDER, (result, EXPECTED_ORDER)
    print(f"rotation_order(angle={ANGLE_DEG}) = {result}  [matches schema]")
