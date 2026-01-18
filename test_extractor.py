import os
import cv2
from ocr.line_utils import extract_tendons


def main():
    import pandas as pd
    i = 0
    word_df = pd.read_csv(f'/home/sadid/PycharmProjects/sgs-drawing-analysis/data/final{i}.csv')
    os.makedirs("data", exist_ok=True)
    image = cv2.imread(f"data/original{i}.png")
    vis, excel = extract_tendons(word_df, image)

    if vis is not None:
        excel.to_excel("data/tendons.xlsx", index=False)
        cv2.imwrite(f"data/ocr_boxes_tendon-{0}.png", vis)
    print("Finished")

if __name__ == '__main__':
    main()
