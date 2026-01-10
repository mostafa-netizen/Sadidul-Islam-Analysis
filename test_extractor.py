import os
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

def extract_tendons(words, image):
    text_extractor = TextExtractor(words, debug=True)
    value = text_extractor.get_tendons()
    height, width = image.shape[:2]
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
    for tendon in value:
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        is_banded = not tendon.loc[tendon.value.str.contains("BANDED")].empty
        if not is_banded:
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        x1, y1, x2, y2 = tendon.x1.min(), tendon.y1.min(), tendon.x2.max(), tendon.y2.max()
        x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)  # indicator bbox
        # vis = draw_boxes(vis, tendon)
        w, h = x2 - x1, y2 - y1
        xe1, ye1, xe2, ye2 = x1 - w, y1 - h, x2 + w, y2 + int(h * 2.5)
        img_crop = image[ye1:ye2, xe1:xe2]
        i = i + 1
        if img_crop.shape[0] > 0 and img_crop.shape[1] > 0:
            # cv2.imwrite(f"data/examples-output/tendon_image_{i}.png", img_crop)
            matched, bbox, val = find_template_and_match(img_crop)

            if matched:
                xt1, yt1, xt2, yt2 = bbox
                xt1, yt1, xt2, yt2 = xt1 + xe1, yt1 + ye1, xt2 + xe1, yt2 + ye1
                found = detect_line_ending_in_bbox(final_lines, (xt1 - b_th, yt1 - b_th, xt2 + b_th, yt2 + b_th))
                if found is not None:
                    cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 3)
                    cv2.rectangle(vis, (xt1, yt1), (xt2, yt2), color, 2)
                    cv2.putText(vis, f"{matched}", (xt1, yt1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    xl1, yl1, xl2, yl2 = found
                    cv2.line(vis, (xl1, yl1), (xl2, yl2), color, 4)
            #     else:
            #         color = (255, 0, 0)
            #         cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 2)
            #         cv2.rectangle(vis, (xt1, yt1), (xt2, yt2), color, 2)
            #         cv2.putText(vis, f"{matched}", (xt1, yt1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # else:
            #     color = (0, 0, 255)
            #     cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 1)
            #     cv2.putText(vis, f"{matched}", (xe1, ye1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

    return vis

def main():
    word_df = pd.read_csv('/home/sadid/PycharmProjects/sgs-drawing-analysis/data/final.csv')
    os.makedirs("data", exist_ok=True)
    image = cv2.imread("data/original.png")
    vis = extract_tendons(word_df, image)
    cv2.imwrite(f"data/ocr_boxes_tendon-{0}.png", vis)
    print("Finished")

if __name__ == '__main__':
    main()
