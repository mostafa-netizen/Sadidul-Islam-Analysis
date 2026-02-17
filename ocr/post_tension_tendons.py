import random
import re

import cv2
import numpy as np
import pandas as pd

from ocr.extractor import TextExtractor
from ocr.line_detector import detect_lines_global, merge_lines, detect_line_ending_in_bbox, template_matching, \
    is_grayscale, find_template_location
from ocr.line_utils import count_text_lines, distance


def line_y(line):
    # horizontal line → y1 == y2
    return line[1]


def bbox_center_y(bbox):
    return (bbox[1] + bbox[3]) / 2


def horizontal_overlap(line, bbox):
    lx1, _, lx2, _ = line
    bx1, _, bx2, _ = bbox

    overlap = min(lx2, bx2) - max(lx1, bx1)
    return overlap > 0


def find_post_tenson_template_and_match(source_image, thresh=2.0):
    image = source_image.copy()
    template_vals = {
        "angled-bottom-left.png": ([0, 200], 'right'),
        "angled-bottom-right.png": ([0, 200], 'left'),
        "angled-top-right.png": ([0, 200], 'left'),
        "angled-top-bottom.png": ([5, 200], 'right')
    }

    return template_matching(image, template_vals, thresh, ksize=(2, 2), start=3, stop=5, num=20)

def remove_noise(img_gray):
    if not is_grayscale(img_gray):
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(thresh, kernel)

    blur = cv2.GaussianBlur(erode, (7, 7), 0)
    ret, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
    erode = cv2.erode(thresh, kernel)

    blur = cv2.GaussianBlur(erode, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
    erode = cv2.erode(thresh, kernel)

    return erode

def detect_line_features(source_img, start=0.8, stop=1.5, num=20):
    template_vals = {
        "post-tension-line-dots.png": ([0, 200], 'right')
    }
    template = list(template_vals.keys())[0]
    template_img = cv2.imread(f"img_templates/{template}", cv2.IMREAD_COLOR)

    source_img = remove_noise(source_img)
    features = []
    crop_length = 200
    for i in range(int(source_img.shape[1]/crop_length)):
        xe1, xe2 = (i*crop_length), (i+1)*crop_length
        score, (x1, y1, x2, y2) = find_template_location(source_img[0:100, xe1:xe2], template_img, start=start, stop=stop, num=num)
        if score > 0.8:
            features.append([x1+(i*crop_length), y1, x2+(i*crop_length), y2])

    return features

def detect_post_tension_template_and_line(
        image, final_lines, x1, y1, x2, y2, line_count, b_th, m_th=2.00, c_left=0.0, c_right=0.60, c_up=1.0, c_down=2.0,
        retry=0
):
    w, h = x2 - x1, y2 - y1
    xe1, ye1, xe2, ye2 = x1 - int(w * c_left), y1 - int((h / line_count) * c_up), x2 + int(w * c_right), y2 + int((h / line_count) * c_down)
    e_bbox = xe1, ye1, xe2, ye2
    img_crop = image[ye1:ye2, xe1:xe2]

    if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
        matched, bbox, val = find_post_tenson_template_and_match(img_crop, m_th)
        found = None

        if matched:
            xt1, yt1, xt2, yt2 = bbox
            xt1, yt1, xt2, yt2 = xt1 + xe1, yt1 + ye1, xt2 + xe1, yt2 + ye1
            bbox = xt1, yt1, xt2, yt2
            found = detect_line_ending_in_bbox(final_lines, (xt1 - b_th, yt1 - b_th, xt2 + b_th, yt2 + b_th))
            if found is not None:
                return found, matched, bbox, val, e_bbox

        if not matched or found is None:
            vertical = 0.45
            horizontal = 0.2
            if retry < 2:
                c_up, c_down, c_right = c_up + vertical, c_down + vertical, c_right + horizontal
            elif retry == 2:
                c_up, c_down, c_left, c_right = 1.0, 2.0, 0.6, 0
            elif retry < 5:
                c_up, c_down, c_left = c_up + vertical, c_down + vertical, c_left + horizontal
            elif retry == 5:
                c_up, c_down, c_left, c_right = 1.0, 2.0, 0.6, 0.6
            elif retry < 8:
                c_up, c_down, c_left, c_right = c_up + vertical, c_down + vertical, c_left + horizontal, c_right + horizontal
            else:
                return None, matched, bbox, val, e_bbox

            return detect_post_tension_template_and_line(
                image, final_lines, x1, y1, x2, y2, line_count, b_th,
                c_left=c_left, c_right=c_right,
                c_up=c_up, c_down=c_down, m_th=m_th, retry=retry + 1
            )

    return None

def closest_bbox_per_line(lines, bboxes):
    result = []

    for line in lines:
        ly = line_y(line)

        best_bbox = None
        best_dist = float("inf")
        best_idx = -1

        for i, bbox in enumerate(bboxes):
            if not horizontal_overlap(line, bbox):
                continue  # skip unrelated objects

            by = bbox_center_y(bbox)
            dist = abs(ly - by)

            if dist < best_dist:
                best_dist = dist
                best_bbox = bbox
                best_idx = i

        result.append((line, best_bbox, best_dist, best_idx))

    return result

def remove_similar_lines(lines, y_thresh=10):
    """
    Remove similar horizontal lines.
    Keeps the longest line among those close vertically.
    """

    # sort by y coordinate
    lines = sorted(lines, key=lambda x: x[1])

    filtered = []
    used = [False] * len(lines)

    for i, l1 in enumerate(lines):
        if used[i]:
            continue

        group = [i]
        y1 = l1[1]

        # group close lines
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue

            y2 = lines[j][1]

            if abs(y1 - y2) <= y_thresh:
                group.append(j)
                used[j] = True
            else:
                break

        # pick the longest line in this group
        best_idx = max(
            group,
            key=lambda idx: abs(lines[idx][2] - lines[idx][0])
        )

        filtered.append(lines[best_idx])

    return filtered


def extract_post_tension_tendons(words, image):
    text_extractor = TextExtractor(words, debug=True)
    dist_per_inch = text_extractor.get_post_tenson_scale(debug=True)
    value = text_extractor.get_post_tenson_tendons()
    height, width = image.shape[:2]
    pixel_per_inch = width * dist_per_inch

    erode = remove_noise(image)

    raw_lines = detect_lines_global(erode)
    final_lines = merge_lines(raw_lines, 2)

    vis = image.copy()
    tendon_lines = []
    for x1, y1, x2, y2 in final_lines:
        img_crop = image[y1 - 50:y2 + 50, x1 - 100:x2 + 100]
        features = detect_line_features(source_img=img_crop)

        if len(features) > 2:
            tendon_lines.append([x1, y1, x2, y2])

    tendon_lines = remove_similar_lines(tendon_lines, 20)
    # for x1, y1, x2, y2 in final_lines:
    #     cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    i = 0
    tendon_boxes = []
    tendon_value = []
    for tendon in value:
        i = i + 1
        x1, y1, x2, y2 = tendon.x1.min(), tendon.y1.min(), tendon.x2.max(), tendon.y2.max()
        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
        tendon_boxes.append([x1, y1, x2, y2])
        tendon_value.append(tendon.iloc[0].value)

    matches = closest_bbox_per_line(tendon_lines, tendon_boxes)

    excel = []
    for line, bbox, dist, idx in matches:
        if line is None or bbox is None or dist > 150:
            continue
        color = (0, 0, 255)
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        xl1, yl1, xl2, yl2 = line
        cv2.line(vis, (xl1, yl1), (xl2, yl2), color, 3)
        x1, y1, x2, y2 = bbox
        length_in_inch = distance(xl1, yl1, xl2, yl2) / pixel_per_inch
        length_in_feet = length_in_inch / 12
        measurement = f"{int(length_in_feet)}'{int(length_in_inch % 12)}\""
        cv2.putText(vis, measurement, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 3)

        match = re.search(r"(\d+\.?\d*)", tendon_value[idx])

        force_per_feet = 0
        if match:
            force_per_feet = float(match.group(1))

        excel.append([tendon_value[idx], measurement, force_per_feet * length_in_feet])

    excel = pd.DataFrame(excel, columns=["Callouts", "Length", "Total Force"])
    return vis, excel
