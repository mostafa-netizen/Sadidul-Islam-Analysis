import os
from itertools import batched

import numpy as np
import tqdm
from pdf2image import convert_from_path
from ocr.doctr import OCR
import pandas as pd
import cv2

from test_extractor import extract_tendons


def crop_tiles(image, tile_size=1000, overlap=250):
    h, w = image.shape[:2]
    stride = tile_size - overlap

    tiles = []
    tile_id = 0

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            tile = image[y:y + tile_size, x:x + tile_size]
            if tile.size == 0:
                continue

            tiles.append({
                "tile_id": tile_id,
                "x_offset": x,
                "y_offset": y,
                "image": tile
            })
            tile_id += 1
    return tiles


def project_tile_df_to_global(df_tile, x_offset, y_offset, tile_w, tile_h, full_w, full_h, tile_id ):
    df = df_tile.copy()

    # tile → pixel
    df["x1_px"] = df["x1"] * tile_w + x_offset
    df["y1_px"] = df["y1"] * tile_h + y_offset
    df["x2_px"] = df["x2"] * tile_w + x_offset
    df["y2_px"] = df["y2"] * tile_h + y_offset

    # pixel → global normalized
    df["x1"] = df["x1_px"] / full_w
    df["y1"] = df["y1_px"] / full_h
    df["x2"] = df["x2_px"] / full_w
    df["y2"] = df["y2_px"] / full_h

    df["tile_id"] = tile_id

    return df[["value", "confidence", "x1", "y1", "x2", "y2", "tile_id"]]


def box_iou(a, b):
    xa1, ya1, xa2, ya2 = a
    xb1, yb1, xb2, yb2 = b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)

    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0


def deduplicate_ocr(df, iou_thresh=0.6):
    df = df.sort_values("confidence", ascending=False).reset_index(drop=True)

    keep = []
    suppressed = set()

    for i in range(len(df)):
        if i in suppressed:
            continue

        keep.append(i)
        box_i = df.loc[i, ["x1", "y1", "x2", "y2"]].values
        text_i = df.loc[i, "value"].strip().lower()

        for j in range(i + 1, len(df)):
            if j in suppressed:
                continue

            text_j = df.loc[j, "value"].strip().lower()
            if text_i != text_j:
                continue

            box_j = df.loc[j, ["x1", "y1", "x2", "y2"]].values
            if box_iou(box_i, box_j) >= iou_thresh:
                suppressed.add(j)

    return df.loc[keep].reset_index(drop=True)


def tile_ocr(drawing, gpu, batch_size=2) -> pd.DataFrame:
    full_h, full_w = drawing.shape[:2]
    tiles = crop_tiles(drawing)
    docs = [tile["image"] for tile in tiles]
    ocr = OCR(gpu=gpu)
    results = []

    for batch in list(batched(docs, batch_size)):
        if len(batch) == 0:
            break

        results.extend(ocr.from_image(list(batch)))

    all_dfs = []
    for i in range(len(tiles)):
        df_tile = results[i]
        tile = tiles[i]

        if df_tile is None or df_tile.empty:
            continue

        df_global = project_tile_df_to_global(
            df_tile,
            tile["x_offset"],
            tile["y_offset"],
            tile["image"].shape[1],
            tile["image"].shape[0],
            full_w=full_w,
            full_h=full_h,
            tile_id=tile["tile_id"]
        )

        all_dfs.append(df_global)

    df_final = pd.concat(all_dfs, ignore_index=True)
    df_final = deduplicate_ocr(df_final, iou_thresh=0.6)
    df_final["word_idx"] = range(len(df_final))

    return df_final


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


def main():
    input_path = '/home/sadid/PycharmProjects/sgs-drawing-analysis/data/1029 Market (Struct 4.19.2017) compress.pdf'
    gpu = True

    images = convert_from_path(input_path)
    # images = images[10:11]
    print("Total images: ", len(images))
    os.makedirs("data/final_output", exist_ok=True)
    progress = tqdm.tqdm(total=len(images))
    for i, drawing in enumerate(images):
        drawing = np.asarray(drawing)
        df_final = tile_ocr(drawing, batch_size=24, gpu=gpu)
        # cv2.imwrite(f"data/original.png", drawing)
        # df_final.to_csv("data/final.csv", index=False)
        vis = extract_tendons(df_final, drawing)
        cv2.imwrite(f"data/final_output/tendons-{i}.png", vis)
        progress.update(1)


if __name__ == '__main__':
    main()
