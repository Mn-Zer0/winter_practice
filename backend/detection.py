import cv2
import numpy as np
from ultralytics import YOLO

# PATHS
MODEL_A = "model/main_model.pt"
MODEL_SEG = "yolo11n-seg.pt"
IMAGE_PATH = "image_for_test1.jpg"

SEG_THRESHOLD = 0.1

# LOAD MODELS
model_A = YOLO(MODEL_A)
model_seg = YOLO(MODEL_SEG)

# LOAD IMAGE
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]
overlay = img.copy()

# 1. DETECT LETTER A
res_A = model_A(img, conf=0.3)[0]

if res_A.boxes is None or len(res_A.boxes) == 0:
    print("Letter A not found")
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    exit()

ax1, ay1, ax2, ay2 = res_A.boxes.xyxy[0].cpu().numpy().astype(int)
a_cx = (ax1 + ax2) // 2
a_cy = (ay1 + ay2) // 2

cv2.rectangle(overlay, (ax1, ay1), (ax2, ay2), (255, 0, 0), 2)
cv2.circle(overlay, (a_cx, a_cy), 4, (255, 0, 0), -1)

# 2. ROI AROUND LETTER A
rx1 = max(a_cx - 320, 0)
rx2 = min(a_cx + 320, w)
ry1 = max(ay1 - 40, 0)
ry2 = min(ay2 + 200, h)

roi = img[ry1:ry2, rx1:rx2]

# 3. FIND WHITE LINE
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
_, white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
edges = cv2.Canny(white, 50, 150)

lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=50,
    minLineLength=50,
    maxLineGap=20
)

if lines is None:
    print("White line not found")
    cv2.imshow("Result", overlay)
    cv2.waitKey(0)
    exit()

# 4. FIND LINE NEAR A
best_line = None
best_dist = 1e9

for l in lines:
    x1, y1, x2, y2 = l[0]
    gx1, gy1 = x1 + rx1, y1 + ry1
    gx2, gy2 = x2 + rx1, y2 + ry1

    if max(gy1, gy2) < ay1 or min(gy1, gy2) > ay2:
        continue

    if gx2 < ax1:
        dist = ax1 - max(gx1, gx2)
    elif gx1 > ax2:
        dist = min(gx1, gx2) - ax2
    else:
        continue

    if dist < best_dist:
        best_dist = dist
        best_line = (gx1, gy1, gx2, gy2)

if best_line is None:
    print("Line near A not found")
    exit()

# 5. EXTEND LINE
lx1, ly1, lx2, ly2 = best_line

if lx1 == lx2:
    pt1 = (lx1, 0)
    pt2 = (lx1, h)
else:
    k = (ly2 - ly1) / (lx2 - lx1)
    b = ly1 - k * lx1
    pt1 = (0, int(b))
    pt2 = (w, int(k * w + b))

cv2.line(overlay, pt1, pt2, (0, 255, 0), 3)

# 6. SIDE FUNCTION
def side_of_line(px, py, x1, y1, x2, y2):
    return (px - x1) * (y2 - y1) - (py - y1) * (x2 - x1)

a_side = side_of_line(a_cx, a_cy, lx1, ly1, lx2, ly2)

# 7. DETECT MOTORCYCLES (SEG, AREA)
res = model_seg(img, conf=0.3)[0]
VIOLATION = False

if res.masks is not None:
    for i, (box, cls) in enumerate(zip(res.boxes.xyxy, res.boxes.cls)):
        if int(cls) != 3:
            continue

        mx1, my1, mx2, my2 = box.cpu().numpy().astype(int)
        poly = res.masks.xy[i]

        if poly.shape[0] < 3:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        poly_int = poly.astype(np.int32)
        cv2.fillPoly(mask, [poly_int], 1)

        ys, xs = np.where(mask == 1)
        total = len(xs)

        if total == 0:
            continue

        same_side = 0
        for x, y in zip(xs, ys):
            m_side = side_of_line(x, y, lx1, ly1, lx2, ly2)
            if m_side == 0 or (m_side > 0 and a_side > 0) or (m_side < 0 and a_side < 0):
                same_side += 1

        ratio = same_side / total
        is_violation = ratio >= SEG_THRESHOLD

        if is_violation:
            VIOLATION = True
            color = (0, 0, 255)
            label = f"VIOLATION {ratio:.2f}"
        else:
            color = (0, 255, 0)
            label = f"motorcycle {ratio:.2f}"

        cv2.rectangle(overlay, (mx1, my1), (mx2, my2), color, 2)
        cv2.putText(
            overlay,
            label,
            (mx1, my1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

# SHOW RESULT
print("VIOLATION:", VIOLATION)
cv2.imshow("Result", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()