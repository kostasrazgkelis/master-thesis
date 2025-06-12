import hashlib
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import StringType
import pandas as pd
import jellyfish

@pandas_udf(StringType())
def trim_one_char(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: x[:-1] if pd.notnull(x) and len(x) > 0 else "")

@pandas_udf(StringType())
def trim_two_chars(s: pd.Series) -> pd.Series:
    return s.apply(lambda x: x[:-2] if pd.notnull(x) and len(x) > 1 else "")

def get_trimmer_udf(trim=0):
    if trim == 1:
        return trim_one_char
    elif trim == 2:
        return trim_two_chars
    else:
        return pandas_udf(lambda s: s, StringType())
    
@pandas_udf(StringType())
def soundex_udf(col: pd.Series) -> pd.Series:
    return col.apply(lambda x: jellyfish.soundex(str(x)) if pd.notnull(x) else "")

@pandas_udf(StringType())
def hash_udf(col: pd.Series) -> pd.Series:
    return col.apply(lambda x: hashlib.sha256(str(x).encode('utf-8')).hexdigest() if pd.notnull(x) else "")