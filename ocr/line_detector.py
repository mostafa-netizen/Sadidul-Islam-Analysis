import uuid

import cv2
import numpy as np


def find_template_location(image, template):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
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

    try:
        val, (x, y), (w, h) = best
        bbox = (x, y, x + w, y + h)

        return val, bbox
    except TypeError:
        return None, None

def is_horizontal(l):
    return abs(l[1] - l[3]) < 10

def is_vertical(l):
    return abs(l[0] - l[2]) < 10

def distance_to_bbox(line, bbox):
    bx1, by1, bx2, by2 = bbox
    x1, y1, x2, y2 = line

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

def find_contours(image_cropped):
    gray = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
    inverted_image = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inverted_image, 100, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def match_contours(source_cnt, image_crop):
    target_cnt_s = find_contours(image_crop)
    image_copy = image_crop.copy()
    matched = False
    for c in target_cnt_s:
        r = cv2.matchShapes(source_cnt, c, cv2.CONTOURS_MATCH_I1, 0.0)
        if r < 50 and cv2.contourArea(c) > 200:
            cv2.drawContours(image_copy, [c], -1, (0,255,0), 3)
            matched = True

    return matched, image_copy

def crop_template_location(image, template):
    val, bbox = find_template_location(image, template)
    if val is None or bbox is None:
        return False, bbox

    x1, y1, x2, y2 = bbox
    img_crop = image[y1:y2, x1:x2]
    return val, bbox, img_crop

def find_template_and_match(source_image):
    image = source_image.copy()

    template = cv2.imread("/home/sadid/PycharmProjects/sgs-drawing-analysis/img_templates/left-top.png", cv2.IMREAD_COLOR)
    cnt_s = find_contours(template)
    val, bbox, img_crop = crop_template_location(image, template)
    matched, image_r = match_contours(cnt_s[2], img_crop)

    if not matched:
        template = cv2.imread("/home/sadid/PycharmProjects/sgs-drawing-analysis/img_templates/left-bottom.png", cv2.IMREAD_COLOR)
        cnt_s = find_contours(template)
        val, bbox, img_crop = crop_template_location(image, template)
        matched, image_r = match_contours(cnt_s[2], img_crop)

    if not matched:
        template = cv2.imread("/home/sadid/PycharmProjects/sgs-drawing-analysis/img_templates/bottom-left.png", cv2.IMREAD_COLOR)
        cnt_s = find_contours(template)
        val, bbox, img_crop = crop_template_location(image, template)
        matched, image_r = match_contours(cnt_s[2], img_crop)

    # cv2.drawContours(image, [cnt_s[2]], -1, (0, 255, 0), 3)
    cv2.imwrite(f"/home/sadid/PycharmProjects/sgs-drawing-analysis/data/output/{uuid.uuid4()}.png", image_r)
    return matched, bbox, val


def point_inside_bbox(x, y, bbox):
    bx1, by1, bx2, by2 = bbox
    return bx1 <= x <= bx2 and by1 <= y <= by2


def distance_point_to_bbox(x, y, bbox):
    bx1, by1, bx2, by2 = bbox
    dx = max(bx1 - x, 0, x - bx2)
    dy = max(by1 - y, 0, y - by2)
    return (dx**2 + dy**2) ** 0.5

def detect_line_ending_in_bbox(lines, bbox):
    best_line = None
    best_dist = float("inf")

    for x1, y1, x2, y2 in lines:
        p1_inside = point_inside_bbox(x1, y1, bbox)
        p2_inside = point_inside_bbox(x2, y2, bbox)

        # exactly ONE endpoint must be inside
        if p1_inside ^ p2_inside:
            # outside endpoint
            if p1_inside:
                xo, yo = x2, y2
            else:
                xo, yo = x1, y1

            dist = distance_point_to_bbox(xo, yo, bbox)

            if dist < best_dist:
                best_dist = dist
                best_line = (x1, y1, x2, y2)

    return best_line
