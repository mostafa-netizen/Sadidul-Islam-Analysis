import os
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
        j, val, k, loc = cv2.minMaxLoc(res)

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

def tile_image(img, tile_size=1500, overlap=300):
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

def detect_vertical_lines(tile):
    bw = cv2.adaptiveThreshold(tile, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)))
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_CLOSE, bridge_kernel)
    extract_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 60))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, extract_kernel)
    # kernel = np.ones((5, 5), np.uint8)
    # vertical = cv2.dilate(vertical, kernel)
    # vertical = cv2.erode(vertical, kernel, iterations=3)
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 100 and w < 20:
            final_lines.append((x + w // 2, y, x + w // 2, y + h))

    return final_lines

def detect_horizontal_lines(tile):
    bw = cv2.adaptiveThreshold(tile, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))

    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1)))
    bridge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, bridge_kernel)
    extract_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, extract_kernel)
    # kernel = np.ones((5, 5), np.uint8)
    # horizontal = cv2.dilate(horizontal, kernel)
    # horizontal = cv2.erode(horizontal, kernel, iterations=3)
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    final_lines = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 100 and h < 20:
            final_lines.append((x, y + h // 2, x + w, y + h // 2))

    return final_lines

def detect_lines(tile):
    return detect_horizontal_lines(tile) + detect_vertical_lines(tile)

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

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def contour_orientation(cnt):
    pts = cnt.reshape(-1, 2).astype(np.float32)
    mean = np.mean(pts, axis=0)
    pts -= mean

    _, eigenvectors = cv2.PCACompute(pts, mean=None)
    angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0])
    return np.degrees(angle)

def match_contours(source_cnt, image_crop, area):
    target_cnt_s = find_contours(image_crop)
    # image_copy = image_crop.copy()
    scores = []
    cnt_s = []
    for c in target_cnt_s:
        # if r < 50 and cv2.contourArea(c) > 200:
        if cv2.contourArea(c) > area:
            r = cv2.matchShapes(source_cnt, c, cv2.CONTOURS_MATCH_I1, 0.0)
            angle_src = contour_orientation(source_cnt)
            angle_c = contour_orientation(c)

            angle_diff = abs(angle_src - angle_c)
            angle_diff = min(angle_diff, 360 - angle_diff)

            if angle_diff < 15:
                scores.append(r)
                cnt_s.append(c)

    return scores, cnt_s

def crop_template_location(image, template):
    val, bbox = find_template_location(image, template)
    if val is None or bbox is None:
        return False, bbox, None

    x1, y1, x2, y2 = bbox
    img_crop = image[y1:y2, x1:x2]
    return val, bbox, img_crop

def find_matched(image, template, template_val):
    template = cv2.imread(template, cv2.IMREAD_COLOR)
    cnt_s = find_contours(template)
    val, bbox, img_crop = crop_template_location(image, template)
    if val is None or bbox is None:
        return None
    # scores, cnt_s = match_contours(cnt_s[template_val[0]], img_crop, template_val[1])
    scores, cnt_s = match_contours(cnt_s[0], img_crop, template_val[1])
    if len(scores) > 0:
        index = np.argmin(scores)

        return scores[index], bbox, val
    return None

def bbox_position(bbox, img_shape):
    h, w = img_shape[:2]
    cx_img, cy_img = w / 2, h / 2

    x1, y1, x2, y2 = bbox
    cx_box = (x1 + x2) / 2
    cy_box = (y1 + y2) / 2

    dx = cx_box - cx_img
    dy = cy_box - cy_img

    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "bottom" if dy > 0 else "top"

def select_one(image, scores, bboxes, template_vals, templates, vals, thresh=2, retry=0):
    _scores = np.array(scores)
    indexes = np.argsort(_scores)
    index = 0
    found = False
    for index in indexes:
        if bbox_position(bboxes[index], image.shape) == template_vals[templates[index]][1]:
            # print(index, scores[index])
            # print(scores)
            # print(templates)

            found = True
            break

    if not found:
        return False, None, None
    else:
        xt1, yt1, xt2, yt2 = bboxes[index]
        # img = image[yt1:yt2, xt1:xt2]
        img_copy = image.copy()
        cv2.rectangle(img_copy, (xt1, yt1), (xt2, yt2), (255, 0, 0), 3)
        cv2.imwrite(f"data/tests/{uuid.uuid4().hex}.jpg", img_copy)

    return True, bboxes[index], vals[index]

def template_matching(image, template_vals, thresh):
    bboxes = []
    scores = []
    vals = []
    templates = []
    # for template in os.listdir("img_templates"):
    for template in template_vals.keys():
        r = find_matched(image, f"img_templates/{template}", template_vals[template][0])
        if r is not None:
            score, bbox, val = r
            bboxes.append(bbox)
            scores.append(score)
            vals.append(val)
            templates.append(template)

    if len(scores) > 0:
        return select_one(image, scores, bboxes, template_vals, templates, vals, thresh)
    else:
        return False, None, None

def find_template_and_match(source_image, thresh=2):
    image = source_image.copy()
    template_vals = {
        # "1.png": ([3, 100], 'left'),
        # "2.png": ([6, 100], 'left'),
        # "3.png": ([2, 100], 'left'),
        # "4.png": ([0, 100], 'left'),
        # "5.png": ([0, 100], 'right'),
        # "6.png": ([3, 100], 'top'),
        # "7.png": ([2, 100], 'right'),
        # "8.png": ([2, 100], 'right'),
        # "9.png": ([2, 100], 'right'),
        # "10.png": ([3, 100], 'right'),
        # "11.png": ([1, 100], 'right'),
        "bottom-left.png": ([5, 200], 'right'),
        "bottom-left-0.png": ([5, 200], 'right'),
        "bottom-left-1.png": ([5, 200], 'right'),
        "bottom-right-0.png": ([5, 200], 'left'),
        "left-bottom.png": ([2, 200], 'right'),
        "left-bottom-0.png": ([1, 200], 'right'),
        "left-top.png": ([2, 200], 'right'),
        "left-top-0.png": ([1, 200], 'right'),
        "left-top-1.png": ([1, 200], 'right'),
        "left-top-2.png": ([1, 200], 'right'),
        "left-top-3.png": ([1, 200], 'right'),
        "left-top-bottom-0.png": ([0, 300], 'right'),
        # "2-lines-indicator.png": ([0, 300], 'right'),
    }

    return template_matching(image, template_vals, thresh)

def find_post_tenson_template_and_match(source_image, thresh=2):
    image = source_image.copy()
    template_vals = {
        "angled-bottom-left.png": ([0, 200], 'right'),
        "angled-bottom-right.png": ([0, 200], 'left'),
        # "angled-top-bottom.png": ([5, 200], 'right')
    }

    return template_matching(image, template_vals, thresh)

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
