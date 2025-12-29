import cv2

from ocr.line_detector import detect_lines_global, merge_lines

img = cv2.imread("data/original.png")

raw_lines = detect_lines_global(img)
final_lines = merge_lines(raw_lines)

# visualize
for x1, y1, x2, y2 in final_lines:
    cv2.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

cv2.imwrite("lines_detected.png", img)
