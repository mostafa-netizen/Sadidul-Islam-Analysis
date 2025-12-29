import cv2
import numpy as np


def find_template_location(image, template):
    best = None

    for scale in np.linspace(0.6, 1.4, 20):
        resized = cv2.resize(template, None, fx=scale, fy=scale)
        th, tw = resized.shape

        if th > image.shape[0] or tw > image.shape[1]:
            continue

        res = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
        _, val, _, loc = cv2.minMaxLoc(res)

        if best is None or val > best[0]:
            best = (val, loc, (tw, th))

    val, (x, y), (w, h) = best
    bbox = (x, y, x + w, y + h)

    return val, bbox

def is_horizontal(l):
    return abs(l[1] - l[3]) < 10

def is_vertical(l):
    return abs(l[0] - l[2]) < 10

def distance_to_bbox(line, bbox):
    bx1, by1, bx2, by2 = bbox
    x1, y1, x2, y2 = line
    print("bbox", bbox)
    print(line)

    distances = []

    if is_horizontal(line):
        distances.append(("top-y1", abs(y1 - by1)))
        distances.append(("top-y2", abs(y2 - by1)))
        distances.append(("bottom-y1", abs(y1 - by2)))
        distances.append(("bottom-y2", abs(y2 - by2)))

    if is_vertical(line):
        distances.append(("left-x1", abs(x1 - bx1)))
        distances.append(("left-x2", abs(x2 - bx1)))
        distances.append(("right-x1", abs(x1 - bx2)))
        distances.append(("right-x2", abs(x2 - bx2)))

    return distances

def find_closest_line(lines, bbox):
    best = None
    min_dist = float("inf")

    for line in lines:
        for side, dist in distance_to_bbox(line, bbox):
            if dist < min_dist:
                min_dist = dist
                best = {
                    "line": line,
                    "side": side,
                    "distance": dist
                }

    return best

def tile_image(img, tile_size=500, overlap=100):
    h, w = img.shape[:2]
    step = tile_size - overlap

    tiles = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            tile = img[y:y + tile_size, x:x + tile_size]
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                continue
            tiles.append((tile, x, y))
    return tiles

def detect_lines(tile):
    gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=120,
        minLineLength=100,
        maxLineGap=20
    )

    results = []
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]

            # keep horizontal or vertical
            if abs(x1 - x2) < 10 or abs(y1 - y2) < 10:
                results.append((x1, y1, x2, y2))

    return results

def detect_lines_global(img):
    tiles = tile_image(img)
    global_lines = []

    for tile, offset_x, offset_y in tiles:
        lines = detect_lines(tile)
        for x1, y1, x2, y2 in lines:
            global_lines.append((
                x1 + offset_x,
                y1 + offset_y,
                x2 + offset_x,
                y2 + offset_y
            ))

    return global_lines

def merge_lines(lines, dist_thresh=15):
    merged = []

    for line in lines:
        added = False
        for i, m in enumerate(merged):
            # horizontal
            if is_horizontal(line) and is_horizontal(m):
                if abs(line[1] - m[1]) < dist_thresh:
                    merged[i] = (
                        min(line[0], m[0]),
                        int((line[1] + m[1]) / 2),
                        max(line[2], m[2]),
                        int((line[3] + m[3]) / 2)
                    )
                    added = True
                    break

            # vertical
            elif not is_horizontal(line) and not is_horizontal(m):
                if abs(line[0] - m[0]) < dist_thresh:
                    merged[i] = (
                        int((line[0] + m[0]) / 2),
                        min(line[1], m[1]),
                        int((line[2] + m[2]) / 2),
                        max(line[3], m[3])
                    )
                    added = True
                    break

        if not added:
            merged.append(line)

    return merged
