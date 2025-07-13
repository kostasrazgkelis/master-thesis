import jellyfish
import pandas as pd


class MyTransformation:

    def __init__(self):
        pass

    def combine_and_trim_chunks(self, row, trim):
        chunks = [str(x) for x in row[1:]]
        trimmed_chunks = [c[:trim] if trim > 0 else c for c in chunks]
        combined = "".join(trimmed_chunks)
        return (row[0], combined)

    def apply_soundex(self, df: pd.DataFrame, match_column: str):
        for col_name in df.columns:
            if col_name != match_column:
                df[col_name] = df[col_name].apply(lambda x: jellyfish.soundex(str(x)))
