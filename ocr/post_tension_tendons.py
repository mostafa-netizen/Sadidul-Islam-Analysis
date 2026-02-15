import random

import cv2
import numpy as np
import pandas as pd

from ocr.extractor import TextExtractor
from ocr.line_detector import detect_lines_global, merge_lines, detect_line_ending_in_bbox, template_matching
from ocr.line_utils import count_text_lines


def find_post_tenson_template_and_match(source_image, thresh=2.0):
    image = source_image.copy()
    template_vals = {
        "angled-bottom-left.png": ([0, 200], 'right'),
        "angled-bottom-right.png": ([0, 200], 'left'),
        "angled-top-right.png": ([0, 200], 'left'),
        "angled-top-bottom.png": ([5, 200], 'right')
    }

    return template_matching(image, template_vals, thresh, ksize=(2, 2))

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

def extract_post_tension_tendons(words, image):
    text_extractor = TextExtractor(words, debug=True)
    value = text_extractor.get_post_tenson_tendons()
    height, width = image.shape[:2]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(thresh, kernel)

    blur = cv2.GaussianBlur(erode, (7, 7), 0)
    ret, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
    erode = cv2.erode(thresh, kernel)

    blur = cv2.GaussianBlur(erode, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY_INV)
    erode = cv2.erode(thresh, kernel)

    raw_lines = detect_lines_global(erode)
    final_lines = merge_lines(raw_lines, 5)

    vis = image.copy()
    for x1, y1, x2, y2 in final_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    b_th = 10
    i = 0
    excel = []
    for tendon in value:
        color = (0, 0, 255)
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        i = i + 1
        x1, y1, x2, y2 = tendon.x1.min(), tendon.y1.min(), tendon.x2.max(), tendon.y2.max()
        line_count = count_text_lines(tendon)
        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)  # indicator bbox
        data = detect_post_tension_template_and_line(image, final_lines, x1, y1, x2, y2, line_count, b_th)

        if data is not None:
            found, matched, bbox, val, e_bbox = data
            # print(bbox, matched)

            xe1, ye1, xe2, ye2 = e_bbox
            cv2.putText(vis, f"{i}", (xe1+200, ye1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if matched:
                xt1, yt1, xt2, yt2 = bbox
                if found is not None:
                    cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 3)
                    cv2.rectangle(vis, (xt1, yt1), (xt2, yt2), color, 2)
                    xl1, yl1, xl2, yl2 = found
                    cv2.line(vis, (xl1, yl1), (xl2, yl2), color, 4)
                    # try:
                    #     measurement = "~{:.2f}{}".format((distance(xl1, yl1, xl2, yl2)/pixel_dist)*measure, unit)
                    #     cv2.putText(vis, f"{measurement}", (xe1, ye1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    #     excel.append([" ".join(tendon.value.tolist()), measurement])
                    # except Exception as e:
                    #     pass
                    #     # print("Measurement error:", e)
                else:
                    # color = (255, 0, 0)
                    cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 2)
                    cv2.rectangle(vis, (xt1, yt1), (xt2, yt2), color, 2)
                    cv2.putText(vis, f"{matched}", (xt1, yt1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # color = (0, 0, 255)
                cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 1)
                cv2.putText(vis, f"{matched}", (xe1, ye1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    excel = pd.DataFrame(excel, columns=["Callouts", "Measurements"])
    return vis, excel
