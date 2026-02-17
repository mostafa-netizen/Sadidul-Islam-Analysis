import glob
import os
import shutil

import numpy as np
import tqdm
from PIL import Image
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
    cache = True

    if cache and os.path.exists(cache_ocr):
        print("Reading cached images")
        images = []
        names = list(glob.glob(f"{cache_ocr}/*.png"))

        for i in range(len(names)):
            images.append(Image.open(f"{cache_ocr}/original{i}.png"))
    else:
        os.makedirs(cache_ocr, exist_ok=True)
        images = convert_from_path(input_path)
        if cache:

            for i, image in enumerate(images):
                image = np.asarray(image)
                cv2.imwrite(f"{cache_ocr}/original{i}.png", image)

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
        acceptable = [4]
        # acceptable = [8, 10, 12]
        if i not in acceptable:
            continue
        print("page: ", i + 1)
        drawing = np.asarray(drawing)

        cache_dir = f"{cache_ocr}/original{i}.csv"
        if cache and os.path.exists(cache_dir):
            print("Reading cached words")
            df_final = pd.read_csv(cache_dir)
        else:
            df_final = tile_ocr(drawing, batch_size=24, gpu=gpu)

            if cache:
                df_final.to_csv(cache_dir, index=False)

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
