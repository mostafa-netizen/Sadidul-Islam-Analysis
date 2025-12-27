import pandas as pd
from rapidfuzz.fuzz_py import ratio


class BaseExtractor:
    def __init__(
            self,
            words,
            debug=False
    ):
        self.debug = debug
        self.words = words
        self.columns = []
        self.position_names = ['bottom', 'top', 'left', 'right']

    def parse_position(self, position):
        return dict(zip(self.position_names, position))

    @staticmethod
    def calculate_height_width(ref_df):
        if ref_df is None or ref_df.empty:
            raise Exception('Reference DataFrame is empty')

        height = ref_df.y2.mean() - ref_df.y1.mean()
        width = (ref_df.iloc[0].x2 - ref_df.iloc[0].x1) / len(ref_df.iloc[0].value) if len(
            ref_df.iloc[0].value) > 0 else 0
        return height, width

    @staticmethod
    def calculate_top_left(ref_df, line_h, char_w, down, left):
        if ref_df is None or ref_df.empty:
            raise Exception('Reference DataFrame is empty')

        return ref_df.y1.min() + (line_h * down), ref_df.iloc[0].x1 + (char_w * left)

    @staticmethod
    def calculate_bottom_right(ref_df, line_h, char_w, up, right):
        if ref_df is None or ref_df.empty:
            raise Exception('Reference DataFrame is empty')

        return ref_df.y2.max() + (line_h * up), ref_df.iloc[0].x2 + (char_w * right)

    def calculate_dimension(self, ref_df, position):
        if ref_df is None or ref_df.empty:
            raise Exception('Reference DataFrame is empty')

        line_h, char_w = self.calculate_height_width(ref_df)
        top, left = self.calculate_top_left(ref_df, line_h, char_w, position["top"], position["left"])
        bottom, right = self.calculate_bottom_right(ref_df, line_h, char_w, position["bottom"], position["right"])

        return top, left, bottom, right

    def find_keyword(self, keyword, debug=False):
        df = self.words.loc[self.words.value.str.contains(keyword)]

        if debug:
            print("keyword", keyword)
            print(df[self.columns].to_string(index=False))

        return df

    @staticmethod
    def match_words(keyword, value, threshold=0.1):
        rat = ratio(keyword, value)
        if rat < threshold:
            return False
        return True

    def find_keywords(self, keywords, debug=False):
        df_prev = None
        dfs = []

        for i in range(len(keywords)):
            keyword = keywords[i]
            if i == 0:
                index = keyword["index"]
                curr_df = self.find_keyword(keyword["keyword"], debug).iloc[index:index + 1]
                try:
                    if not self.match_words(keyword["keyword"], curr_df.iloc[0].value):
                        raise Exception("Keyword not found")
                except IndexError:
                    raise Exception("Keyword not found")
                dfs.append(curr_df)
            else:
                curr_df = df_prev

            try:
                position = self.parse_position(keyword["next_keyword_position"])
                top, left, bottom, right = self.calculate_dimension(curr_df, position)

                df_next = self.filter_all(self.words, position, top, bottom, left, right).iloc[0:1]

                if self.match_words(keywords[i+1]["keyword"], df_next.iloc[0].value):
                    dfs.append(df_next)
                    df_prev = df_next
            except KeyError:
                pass

        df = pd.concat(dfs)
        df["x1"] = df["x1"].min()
        df["y1"] = df["y1"].min()
        df["x2"] = df["x2"].max()
        df["y2"] = df["y2"].max()
        keyword_str = " ".join(k["keyword"] for k in keywords if "keyword" in k)
        if not self.match_words(keyword_str, self.merge_values(df), 0.4):
            raise Exception("Keyword not found")

        if debug:
            print("keywords", keywords)
            print(df[self.columns].to_string(index=False))

        return df

    def filter_top(self, value, position, top, debug=False):
        if position["top"] != 0:
            if debug:
                print("filtering top")
            value = value.loc[
                (
                        value.y1 >= top
                )
            ]
            if debug:
                print(value[self.columns].to_string(index=False))

        return value

    def filter_bottom(self, value, position, bottom, debug=False):
        if position["bottom"] != 0:
            if debug:
                print("filtering bottom")
            value = value.loc[
                (
                        value.y2 <= bottom
                )
            ]
            if debug:
                print(value[self.columns].to_string(index=False))

        return value

    def filter_left(self, value, position, left, debug=False):
        if position["left"] != 0:
            if debug:
                print("filtering left", left)
            value = value.loc[
                (
                        value.x1 >= left
                )
            ]
            if debug:
                print(value[self.columns].to_string(index=False))

        return value

    def filter_right(self, value, position, right, debug=False):
        if position["right"] != 0:
            if debug:
                print("filtering right")
            value = value.loc[(
                    value.x2 <= right
            )
            ]
            if debug:
                print(value[self.columns].to_string(index=False))

        return value

    def filter_all(self, value, position, top, bottom, left, right, debug=False):
        value = value.copy()

        value = self.filter_top(value, position, top, debug)
        value = self.filter_bottom(value, position, bottom, debug)
        value = self.filter_left(value, position, left, debug)
        value = self.filter_right(value, position, right, debug)

        return value

    @staticmethod
    def merge_values(line_df):
        return " ".join(line_df["value"].tolist()).strip()

