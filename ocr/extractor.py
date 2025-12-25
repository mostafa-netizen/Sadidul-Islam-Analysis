import re

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

        self.string = " ".join(self.words.value.tolist()) if self.words is not None else ""
        self.word_separators = [":", "-", ".", ",", "?"]
        self.columns = ["word_idx", "value", "confidence", "x1", "y1", "x2", "y2"]

    def get_value_from_position(self, find):
        try:
            keywords, position, debug = find["keywords"], find["position_of_value"], find["debug"]
            position = self.parse_position(position)
            ref_df = self.find_keywords(keywords, debug)

            top, left, bottom, right = self.calculate_dimension(ref_df, position)
            value = self.filter_all(self.words, position, top, bottom, left, right, debug)

            if debug:
                print("value")
                print(value[self.columns].to_string())

            if "words" in find.keys():
                value = value.iloc[0:find["words"]]

            result = self.merge_values(value)

            if "regex_parse" in find.keys():
                result = re.search(find["regex_parse"], result).group(0)

            return result
        except Exception as e:
            return None
