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

    def get_tendons(self, debug=False):
        keyword = "TENDON"
        all_keywords = self.find_keyword(keyword, debug)
        tendons = []

        for i, row in all_keywords.reset_index().iterrows():
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
                tendon.append(value.loc[value.value.str.match("\(\s*\d\s*\)", na=False)].iloc[0:1])
            except IndexError:
                pass

            tendon = pd.concat(tendon)
            if tendon.shape[0] > 1:
                tendons.append(tendon)

        return tendons
        # # try:
        # keywords, position, debug = find["keywords"], find["position_of_value"], find["debug"]
        # position = self.parse_position(position)
        # # print(position)
        # # print(ref_df)
        # top, left, bottom, right = self.calculate_dimension(ref_df, position)
        # value = self.filter_all(self.words, position, top, bottom, left, right, debug)
        #
        # if debug:
        #     print("value")
        #     print(value[self.columns].to_string())
        #
        # if "words" in find.keys():
        #     value = value.iloc[0:find["words"]]
        #
        # result = self.merge_values(value)
        #
        # if "regex_parse" in find.keys():
        #     result = re.search(find["regex_parse"], result).group(0)
        #
        # return result
        # # except Exception as e:
        # #
        # #     return None
