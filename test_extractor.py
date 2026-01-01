import random

import cv2
import numpy as np
import pandas as pd

from ocr.extractor import TextExtractor
from ocr.line_detector import detect_lines_global, merge_lines, find_template_and_match, detect_line_ending_in_bbox


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


if __name__ == '__main__':
    words = pd.read_csv('/home/sadid/PycharmProjects/sgs-drawing-analysis/data/final.csv')
    text_extractor = TextExtractor(words, debug=True)
    value = text_extractor.get_tendons()

    img = cv2.imread("data/original.png")
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((2, 2), np.uint8)
    erode = cv2.erode(thresh, kernel)

    raw_lines = detect_lines_global(erode)
    final_lines = merge_lines(raw_lines)

    vis = img.copy()
    for x1, y1, x2, y2 in final_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    i = 0
    for tendon in value:
        x1, y1, x2, y2 = tendon.x1.min(), tendon.y1.min(), tendon.x2.max(), tendon.y2.max()
        x1, y1, x2, y2 = int(x1*width), int(y1*height), int(x2*width), int(y2*height)  # indicator bbox
        # vis = draw_boxes(vis, tendon)

        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        w, h = x2 - x1, y2 - y1
        xe1, ye1, xe2, ye2 = x1 - w, y1 - h, x2 + w, y2 + int(h*1.5)

        cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 3)
        img_crop = img[ye1:ye2, xe1:xe2]

        i = i + 1

        if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
            cv2.imwrite(f"data/examples-output/tendon_image_{i}.png", img_crop)
            matched, bbox, val = find_template_and_match(img_crop)
            xt1, yt1, xt2, yt2 = bbox
            xt1, yt1, xt2, yt2 = xt1 + xe1, yt1  + ye1, xt2 + xe1, yt2 + ye1
            bbox = (xt1, yt1, xt2, yt2)
            cv2.rectangle(vis, (xt1, yt1), (xt2, yt2), color, 2)
            cv2.putText(vis, f"{matched}", (xt1, yt1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            if matched:
                found = detect_line_ending_in_bbox(final_lines, (xt1, yt1, xt2, yt2))
                print(bbox, found)
                if found is not None:
                    xl1, yl1, xl2, yl2 = found
                    cv2.line(vis, (xl1, yl1), (xl2, yl2), color, 4)

        # result = find_closest_line(final_lines, (x1, y1, x2, y2))
        # x1, y1, x2, y2 = result["line"]
        # cv2.line(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)

    cv2.imwrite("data/ocr_boxes_tendon.png", vis)
