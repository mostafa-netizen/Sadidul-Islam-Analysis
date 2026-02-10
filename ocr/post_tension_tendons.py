import cv2
import numpy as np
import pandas as pd

from ocr.extractor import TextExtractor
from ocr.line_detector import detect_lines_global, merge_lines


def extract_post_tension_tendons(words, image):
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
    for x1, y1, x2, y2 in final_lines:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # b_th = 10
    # i = 0
    # excel = []
    # for tendon in value:
    #     # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    #     i = i + 1
    #     print("tendon", i)
    #     is_banded = not tendon.loc[tendon.value.str.contains("BANDED")].empty
    #     if not is_banded:
    #         color = (255, 0, 0)
    #     else:
    #         color = (0, 0, 255)
    #
    #     x1, y1, x2, y2 = tendon.x1.min(), tendon.y1.min(), tendon.x2.max(), tendon.y2.max()
    #     line_count = count_text_lines(tendon)
    #     x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)  # indicator bbox
    #     data = detect_template_and_line(image, final_lines, x1, y1, x2, y2, line_count, b_th)
    #     if data is not None:
    #         found, matched, bbox, val, e_bbox = data
    #         xe1, ye1, xe2, ye2 = e_bbox
    #
    #         cv2.putText(vis, f"{i}", (xe1+200, ye1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    #
    #         if matched:
    #             xt1, yt1, xt2, yt2 = bbox
    #             if found is not None:
    #                 cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 3)
    #                 cv2.rectangle(vis, (xt1, yt1), (xt2, yt2), color, 2)
    #                 xl1, yl1, xl2, yl2 = found
    #                 cv2.line(vis, (xl1, yl1), (xl2, yl2), color, 4)
    #                 try:
    #                     measurement = "~{:.2f}{}".format((distance(xl1, yl1, xl2, yl2)/pixel_dist)*measure, unit)
    #                     cv2.putText(vis, f"{measurement}", (xe1, ye1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    #                     excel.append([" ".join(tendon.value.tolist()), measurement])
    #                 except Exception as e:
    #                     pass
    #                     # print("Measurement error:", e)
    #             else:
    #                 color = (255, 0, 0)
    #                 cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 2)
    #                 cv2.rectangle(vis, (xt1, yt1), (xt2, yt2), color, 2)
    #                 cv2.putText(vis, f"{matched}", (xt1, yt1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #         else:
    #             color = (0, 0, 255)
    #             cv2.rectangle(vis, (xe1, ye1), (xe2, ye2), color, 1)
    #             cv2.putText(vis, f"{matched}", (xe1, ye1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
    #
    # excel = pd.DataFrame(excel, columns=["Callouts", "Measurements"])
    excel = None
    return vis, excel
