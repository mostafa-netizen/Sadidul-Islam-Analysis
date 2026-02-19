import cv2
import pandas as pd
from ocr.extractor import TextExtractor
from ocr.line_detector import detect_lines_global, merge_lines


def connect_dashed_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)
    connect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 3))
    dashed = cv2.morphologyEx(horizontal, cv2.MORPH_CLOSE, connect_kernel)

    return dashed

def extract_beam_based_tendons(words, image):
    text_extractor = TextExtractor(words, debug=True)
    value = text_extractor.get_beam_based_tendons(debug=False)
    height, width = image.shape[:2]

    # dist_per_inch = text_extractor.get_post_tenson_scale(debug=False)
    # pixel_per_inch = width * dist_per_inch

    dashed = connect_dashed_lines(image)
    raw_lines = detect_lines_global(dashed)
    final_lines = merge_lines(raw_lines, 2)

    print(len(final_lines))
    vis = image.copy()
    for _, row in value.iterrows():
        x1 = int(row["x1"] * width)
        y1 = int(row["y1"] * height)
        x2 = int(row["x2"] * width)
        y2 = int(row["y2"] * height)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, str(row["value"]), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for x1, y1, x2, y2 in final_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    excel = []
    excel = pd.DataFrame(excel, columns=["Callouts", "Length", "Total Force"])
    return vis, excel
