import cv2
import pandas as pd

from main import draw_boxes
from ocr.extractor import TextExtractor


if __name__ == '__main__':
    words = pd.read_csv('data/final.csv')
    text_extractor = TextExtractor(words, debug=True)
    value = text_extractor.get_tendons()

    vis = cv2.imread("data/original.png")
    for tendon in value:
        vis = draw_boxes(vis, tendon)
    cv2.imwrite("data/ocr_boxes_tendon.png", vis)
    # print(value)
    #
    # print(words.head())
    # df = words.loc[words.value.str.contains("TENDONS").fillna(False)]
    # print(df.head())
    # print(df.shape)

