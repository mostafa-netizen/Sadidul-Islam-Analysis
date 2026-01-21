import math
import cv2
import numpy as np
import pandas as pd
from ocr.extractor import TextExtractor
from ocr.line_detector import detect_lines_global, merge_lines, find_template_and_match, detect_line_ending_in_bbox


def distance(xl1, yl1, xl2, yl2):
    return math.hypot(xl2 - xl1, yl2 - yl1)


def draw_boxes(image, df, color=(0, 255, 0), thickness=2):
    """
    image: original image (H, W, 3)
    df: dataframe with columns x1,y1,x2,y2 in [0,1]
    """
    img = image.copy()
    h, w = img.shape[:2]

    for _, row in df.iterrows():
        x1 = int(row["x1"] * w)
        y1 = int(row["y1"] * h)
        x2 = int(row["x2"] * w)
        y2 = int(row["y2"] * h)

        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            color,
            thickness
        )
        label = row["value"]
        cv2.putText(
            img,
            label,
            (x1, max(y1 - 5, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA
        )

    return img

def count_text_lines(df, height_ratio=0.7):
    df = df.copy()

    df["cy"] = (df["y1"] + df["y2"]) / 2

    heights = df["y2"] - df["y1"]
    avg_height = np.median(heights)

    y_thresh = avg_height * height_ratio

    df = df.sort_values("cy")

    lines = []
    for cy in df["cy"]:
        if not lines or abs(cy - lines[-1]) > y_thresh:
            lines.append(cy)

    return len(lines)

def detect_template_and_line(
        image, final_lines, x1, y1, x2, y2, line_count, b_th, m_th=2.0, c_left=0.0, c_right=0.60, c_up=1.0, c_down=2.0,
        retry=0
):
    w, h = x2 - x1, y2 - y1
    xe1, ye1, xe2, ye2 = x1 - int(w * c_left), y1 - int((h / line_count) * c_up), x2 + int(w * c_right), y2 + int((h / line_count) * c_down)
    e_bbox = xe1, ye1, xe2, ye2
    img_crop = image[ye1:ye2, xe1:xe2]

    if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
        matched, bbox, val = find_template_and_match(img_crop, m_th)
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

            return detect_template_and_line(
                image, final_lines, x1, y1, x2, y2, line_count, b_th,
                c_left=c_left, c_right=c_right,
                c_up=c_up, c_down=c_down, m_th=m_th, retry=retry + 1
            )

    return None

def extract_tendons(words, image):
    text_extractor = TextExtractor(words, debug=True)
    measure, unit, pixel_dist = text_extractor.get_scale()
    value = text_extractor.get_tendons()
    height, width = image.shape[:2]

    if pixel_dist is not None:
        pixel_dist *= width

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(thresh, kernel)

    raw_lines = detect_lines_global(erode)
    final_lines = merge_lines(raw_lines)

    vis = image.copy()
    # for x1, y1, x2, y2 in final_lines:
    #     cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    b_th = 10
    i = 0
    excel = []
    for tendon in value:
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        i = i + 1
        print("tendon", i)
        is_banded = not tendon.loc[tendon.value.str.contains("BANDED")].empty
        if not is_banded:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        x1, y1, x2, y2 = tendon.x1.min(), tendon.y1.min(), tendon.x2.max(), tendon.y2.max()
        line_count = count_text_lines(tendon)
        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)  # indicator bbox

        if i != 10:
            continue

        data = detect_template_and_line(image, final_lines, x1, y1, x2, y2, line_count, b_th)
        if data is not None:
            found, matched, bbox, val, e_bbox = data
            xe1, ye1, xe2, ye2 = e_bbox

            cv2.putText(vis, f"{i}", (xe1+200, ye1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            if matched:
                xt1, yt1, xt2, yt2 = bbox
                if found is not None:
                    cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 3)
                    cv2.rectangle(vis, (xt1, yt1), (xt2, yt2), color, 2)
                    xl1, yl1, xl2, yl2 = found
                    cv2.line(vis, (xl1, yl1), (xl2, yl2), color, 4)
                    try:
                        measurement = "~{:.2f}{}".format((distance(xl1, yl1, xl2, yl2)/pixel_dist)*measure, unit)
                        cv2.putText(vis, f"{measurement}", (xe1, ye1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        excel.append([" ".join(tendon.value.tolist()), measurement])
                    except Exception as e:
                        pass
                        # print("Measurement error:", e)
                else:
                    color = (255, 0, 0)
                    cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 2)
                    cv2.rectangle(vis, (xt1, yt1), (xt2, yt2), color, 2)
                    cv2.putText(vis, f"{matched}", (xt1, yt1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                color = (0, 0, 255)
                cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 1)
                cv2.putText(vis, f"{matched}", (xe1, ye1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    excel = pd.DataFrame(excel, columns=["Callouts", "Measurements"])
    return vis, excel
