import cv2
import numpy as np
import pandas as pd
from ocr.extractor import TextExtractor
from ocr.line_detector import detect_lines_global, merge_lines, find_template_location


def connect_dashed_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    dashed = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, connect_kernel)

    kernel = np.ones((2, 2), np.uint8)
    blur = cv2.GaussianBlur(dashed, (7, 7), 0)
    ret, thresh = cv2.threshold(blur, 80, 255, cv2.THRESH_BINARY)
    erode = cv2.erode(thresh, kernel, iterations=2)
    dilate = cv2.dilate(erode, kernel, iterations=2)

    return dilate

def detect_line_features_dash(source_img, start=0.8, stop=1.5, num=20):
    template_vals = {
        "post-tension-line-dots.png": ([0, 200], 'right')
    }
    template = list(template_vals.keys())[0]
    template_img = cv2.imread(f"img_templates/{template}", cv2.IMREAD_COLOR)

    features = []
    crop_length = 50
    for i in range(int(source_img.shape[1]/crop_length)):
        xe1, xe2 = (i*crop_length), (i+1)*crop_length
        score, (x1, y1, x2, y2) = find_template_location(source_img[0:100, xe1:xe2], template_img, start=start, stop=stop, num=num)
        if score > 0.8:
            features.append([x1+(i*crop_length), y1, x2+(i*crop_length), y2])

    return features

def extract_beam_based_tendons(words, image):
    text_extractor = TextExtractor(words, debug=True)
    value = text_extractor.get_beam_based_tendons(debug=False)
    height, width = image.shape[:2]

    # dist_per_inch = text_extractor.get_post_tenson_scale(debug=False)
    # pixel_per_inch = width * dist_per_inch

    dashed = connect_dashed_lines(image)
    raw_lines = detect_lines_global(dashed)
    final_lines = merge_lines(raw_lines, 2)

    vis = image.copy()
    for _, row in value.iterrows():
        x1 = int(row["x1"] * width)
        y1 = int(row["y1"] * height)
        x2 = int(row["x2"] * width)
        y2 = int(row["y2"] * height)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, str(row["value"]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    tendon_lines = []
    for x1, y1, x2, y2 in final_lines:
        img_crop = image[y1 - 50:y2 + 50, x1 - 100:x2 + 100]
        features = detect_line_features_dash(source_img=img_crop)

        if len(features) > 2:
            # tendon_lines.append([x1, y1, x2, y2])
            cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    excel = []
    excel = pd.DataFrame(excel, columns=["Callouts", "Length", "Total Force"])
    return vis, excel
