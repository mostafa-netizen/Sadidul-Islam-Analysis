import os
import shutil

import numpy as np
import tqdm
from pdf2image import convert_from_path
import pandas as pd
import cv2

from ocr.line_utils import extract_tendons
from ocr.post_tension_tendons import extract_post_tension_tendons
from ocr.ocr_utils import tile_ocr



def main():
    input_path = 'data/pdfs/Miliennium Garage Original Structural Drawings.pdf'
    # input_path = 'data/pdfs/1029 Market (Struct 4.19.2017) compress.pdf'
    # input_path = 'data/pdfs/plan.pdf'
    cache_ocr = 'data/miliennium_garage'
    gpu = True

    images = convert_from_path(input_path)
    # images = [
    #     images[8],
    #     images[10],
    #     images[12]
    # ]
    images = [
        images[4],
    ]
    print("Total images: ", len(images))

    directory_path = "data/final_output"
    debug_path = "data/debug"
    print("removing old files...")
    if os.path.exists(debug_path) and os.path.isdir(debug_path):
        shutil.rmtree(debug_path)
    os.makedirs(debug_path, exist_ok=True)
    if os.path.exists(directory_path) and os.path.isdir(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path, exist_ok=True)

    progress = tqdm.tqdm(total=len(images))
    excels = []
    for i, drawing in enumerate(images):
        print("page: ", i + 1)
        drawing = np.asarray(drawing)
        df_final = tile_ocr(drawing, batch_size=24, gpu=gpu)
        # cv2.imwrite(f"{cache_ocr}/original{i}.png", drawing)
        # df_final.to_csv(f"{cache_ocr}/original{i}.csv", index=False)
        vis, excel = extract_post_tension_tendons(df_final, drawing)
        # vis, excel = extract_tendons(df_final, drawing)
        if excel is not None:
            excel["page"] = i + 1
            excels.append(excel)
        cv2.imwrite(f"{directory_path}/tendons-{i}.png", vis)
        progress.update(1)

    if len(excels) > 0:
        excels = pd.concat(excels)
        print("Total tendons: ", len(excels))
        print(excels.head())
        excels.to_excel(f"{directory_path}/tendons.xlsx", index=False)
    else:
        print("No tendons found")


if __name__ == '__main__':
    main()
