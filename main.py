import os
import numpy as np
import tqdm
from pdf2image import convert_from_path
import pandas as pd
import cv2

from ocr.line_utils import extract_tendons
from ocr.ocr_utils import tile_ocr



def main():
    input_path = '/home/sadid/PycharmProjects/sgs-drawing-analysis/data/1029 Market (Struct 4.19.2017) compress.pdf'
    # input_path = '/home/sadid/PycharmProjects/sgs-drawing-analysis/data/plan.pdf'
    gpu = True

    images = convert_from_path(input_path)
    images = [
        images[8],
        images[10],
        images[12]
    ]
    print("Total images: ", len(images))
    os.makedirs("data/final_output", exist_ok=True)
    progress = tqdm.tqdm(total=len(images))
    excels = []
    for i, drawing in enumerate(images):
        print("page: ", i + 1)
        drawing = np.asarray(drawing)
        df_final = tile_ocr(drawing, batch_size=24, gpu=gpu)
        # cv2.imwrite(f"data/original{i}.png", drawing)
        # df_final.to_csv(f"data/final{i}.csv", index=False)
        vis, excel = extract_tendons(df_final, drawing)
        excel["page"] = i + 1
        excels.append(excel)
        cv2.imwrite(f"data/final_output/tendons-{i}.png", vis)
        progress.update(1)

    excels = pd.concat(excels)
    print("Total tendons: ", len(excels))
    print(excels.head())
    excels.to_excel("data/final_output/tendons.xlsx", index=False)


if __name__ == '__main__':
    main()
