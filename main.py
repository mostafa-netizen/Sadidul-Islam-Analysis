import numpy as np
from pdf2image import convert_from_path
from ocr.doctr import OCR
import pandas as pd


def crop_tiles(image, tile_size=1000):
    tiles = []
    h, w = image.shape[:2]

    for r in range(0, h, tile_size):
        for c in range(0, w, tile_size):
            tile = image[r:r+tile_size, c:c+tile_size]
            tiles.append({
                "row": r // tile_size,
                "col": c // tile_size,
                "y_offset": r,
                "x_offset": c,
                "image": tile
            })
    return tiles


def project_tile_df_to_global(
    df_tile,
    x_offset,
    y_offset,
    tile_w,
    tile_h,
    full_w,
    full_h
):
    df = df_tile.copy()

    # Tile-normalized → tile-pixel
    df["x1_px"] = df["x1"] * tile_w
    df["y1_px"] = df["y1"] * tile_h
    df["x2_px"] = df["x2"] * tile_w
    df["y2_px"] = df["y2"] * tile_h

    # Tile-pixel → global-pixel
    df["x1_px"] += x_offset
    df["y1_px"] += y_offset
    df["x2_px"] += x_offset
    df["y2_px"] += y_offset

    # Global-pixel → global-normalized
    df["x1"] = df["x1_px"] / full_w
    df["y1"] = df["y1_px"] / full_h
    df["x2"] = df["x2_px"] / full_w
    df["y2"] = df["y2_px"] / full_h

    return df[["word_idx", "value", "confidence", "x1", "y1", "x2", "y2"]]


if __name__ == '__main__':
    images = convert_from_path('/home/sadid/PycharmProjects/sgs-drawing-analysis/data/1029 Market (Struct 4.19.2017) compress.pdf')
    print(len(images))

    drawing = np.asarray(images[8])
    all_dfs = []
    tiles = crop_tiles(drawing)
    docs = [tile["image"] for tile in tiles]
    print(len(docs))
    ocr = OCR(gpu=True)
    results = ocr.from_image(docs[:5])
    print(len(results))
    results = ocr.from_image(docs[5:10])
    print(len(results))
    results = ocr.from_image(docs[10:15])
    print(len(results))
    # for tile in docs:
    #     tile_img = tile["image"]
    #     print(tile_img.shape)
    #     df_tile = run_ocr(tile_img)
    #
    #     if len(df_tile) == 0:
    #         continue
    #
    #     df_global = project_tile_df_to_global(
    #         df_tile=df_tile,
    #         x_offset=tile["x_offset"],
    #         y_offset=tile["y_offset"],
    #         tile_w=tile_img.shape[1],
    #         tile_h=tile_img.shape[0],
    #         full_w=W,
    #         full_h=H
    #     )
    #
    #     all_dfs.append(df_global)
    #
    # df_final = pd.concat(all_dfs, ignore_index=True)

    # print(np.asarray(images[8]).shape)
    # ocr = OCR(gpu=True)
    # results = ocr.from_image([doc1])
    # # print(results[0][['word_idx', 'value', 'confidence', 'x1', 'y1', 'x2', 'y2']])
    # print(" ".join(results[0]["value"]))
