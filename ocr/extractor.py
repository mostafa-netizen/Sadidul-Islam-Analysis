import re
from operator import index

import numpy as np
import pandas as pd

from ocr.base_extractor import BaseExtractor


class TextExtractor(BaseExtractor):
    def __init__(
            self,
            words,
            debug=False
    ):
        super().__init__(
            words,
            debug=debug
        )
        self.words.fillna("", inplace=True)
        self.string = " ".join(self.words.value.tolist()) if self.words is not None else ""
        self.word_separators = [":", "-", ".", ",", "?"]
        self.columns = ["word_idx", "value", "confidence", "x1", "y1", "x2", "y2"]

    def get_beam_based_tendons(self, debug=False):
        keyword = r"^B[--]\d+$"
        all_keywords = self.find_keyword(keyword, debug)
        # tendons = []
        #
        # for i, _ in all_keywords.reset_index().iterrows():
        #     ref_df = self.find_keywords([{"keyword": keyword, "index": i}], debug)
        #     position = self.parse_position([1, -1, -3, 3])
        #     top, left, bottom, right = self.calculate_dimension(ref_df, position)
        #     value = self.filter_all(self.words, position, top, bottom, left, right, debug)
        #
        #     tendon = []
        #     try:
        #         tendon.append(value.loc[value.value.str.contains(keyword, na=False)].iloc[0:1])
        #     except IndexError:
        #         pass
        #
        #     tendon = pd.concat(tendon)
        #     if tendon.shape[0] > 0:
        #         tendons.append(tendon)

        return all_keywords

    def get_post_tenson_tendons(self, debug=False, keyword="F="):
        all_keywords = self.find_keyword(keyword, debug)
        tendons = []

        for i, _ in all_keywords.reset_index().iterrows():
            ref_df = self.find_keywords([{"keyword": keyword, "index": i}], debug)
            position = self.parse_position([1, -1, -3, 3])
            top, left, bottom, right = self.calculate_dimension(ref_df, position)
            value = self.filter_all(self.words, position, top, bottom, left, right, debug)

            tendon = []
            try:
                tendon.append(value.loc[value.value.str.contains(keyword, na=False)].iloc[0:1])
            except IndexError:
                pass

            tendon = pd.concat(tendon)
            if tendon.shape[0] > 0:
                tendons.append(tendon)

        return tendons

    def get_post_tenson_scale(self, debug=False):
        try:
            numbers_df = self.words[self.words["value"].str.isnumeric()]
            numbers_df.loc[:, "value"] = pd.to_numeric(numbers_df["value"], errors="coerce").astype("int")
            numbers_df = numbers_df[(numbers_df["value"] >= 1) & (numbers_df["value"] <= 15)]
            numbers_df.sort_values(by=["y1"], inplace=True)

            y = np.sort(numbers_df["y1"].values)
            bin_width = 0.002

            max_count = 0
            best_start = None

            left = 0
            for right in range(len(y)):
                while y[right] - y[left] > bin_width:
                    left += 1

                count = right - left + 1
                if count > max_count:
                    max_count = count
                    best_start = y[left]

            numbers_df = numbers_df[(numbers_df["y1"] >= best_start) & (numbers_df["y1"] <= best_start + bin_width)]
            numbers_df.sort_values(by=["value"], inplace=True)
            numbers_df.loc[:, "diff"] = numbers_df["value"].shift(-1) - numbers_df["value"]
            dist = 0
            measure = 0
            for i in range(numbers_df.shape[0]):
                if numbers_df.iloc[i]["diff"] == 1:
                    try:
                        ref_df = numbers_df.iloc[i:i+2].reset_index(drop=True)

                        ref_df["value"] = ref_df.value.astype(str)
                        position = self.parse_position([4, 2, -1, 1])
                        top, left, bottom, right = self.calculate_dimension(ref_df, position, end_to_end=True)
                        value = self.filter_all(self.words, position, top, bottom, left, right, False)
                        match = re.match(r"(\d+)'\s*-\s*(\d+)\"", value.iloc[0].value)

                        if match:
                            feet = int(match.group(1))
                            inches = int(match.group(2))
                            p1 = (ref_df.iloc[0].x1 + ref_df.iloc[0].x2) / 2
                            p2 = (ref_df.iloc[1].x1 + ref_df.iloc[1].x2) / 2

                            dist = abs(p1 - p2)
                            measure = (feet * 12) + inches
                    except IndexError:
                        pass

            return dist/measure
        except (ZeroDivisionError, TypeError):
            return 0

    def get_tendons(self, debug=False, keyword="TENDON"):
        all_keywords = self.find_keyword(keyword, debug)
        tendons = []

        for i, _ in all_keywords.reset_index().iterrows():
            ref_df = self.find_keywords([{"keyword": keyword, "index": i}], debug)
            position = self.parse_position([1, -4, -4, 4])
            top, left, bottom, right = self.calculate_dimension(ref_df, position)
            value = self.filter_all(self.words, position, top, bottom, left, right, debug)

            tendon = []
            try:
                tendon.append(value.loc[value.value.str.contains(keyword, na=False)].iloc[0:1])
            except IndexError:
                pass
            try:
                tendon.append(value.loc[value.value.str.contains("BANDED", na=False)].iloc[0:1])
            except IndexError:
                pass
            try:
                tendon.append(value.loc[value.value.str.match(r"\(\s*\d\s*\)", na=False)].iloc[0:1])
            except IndexError:
                pass

            tendon = pd.concat(tendon)
            if tendon.shape[0] > 1:
                tendons.append(tendon)

        return tendons

    def get_scale(self, debug=False):
        keyword = "SCALE"
        all_keywords = self.find_keyword(keyword, debug)
        try:
            for i, _ in all_keywords.reset_index().iterrows():
                pattern = r"^\d+'$"
                ref_df = self.find_keywords([{"keyword": keyword, "index": i}], debug)
                position = self.parse_position([5, -4, 50, 150])
                top, left, bottom, right = self.calculate_dimension(ref_df, position)
                value = self.filter_all(self.words, position, top, bottom, left, right, debug)
                mask = value['value'].astype(str).str.match(pattern)
                dimension_df = value[mask]
                dimension_df = dimension_df.sort_values(by=["x1"])
                dimension_df = dimension_df.reset_index(drop=True)
                dimension_df['x_mid'] = (dimension_df['x1'] + dimension_df['x2']) / 2

                dist_0_1 = abs(dimension_df.loc[0, 'x_mid'] - dimension_df.loc[1, 'x_mid'])
                d1 = int(re.search(r"\d+", dimension_df.loc[0, 'value']).group())
                d2 = int(re.search(r"\d+", dimension_df.loc[1, 'value']).group())

                return abs(d2 - d1), dimension_df.loc[0, 'value'][-1], dist_0_1
        except Exception as e:
            pass
            # print("Measure detection error:", e)

        return None, None, None

