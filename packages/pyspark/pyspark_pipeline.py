from pyspark.sql import functions as F
from pyspark.sql import DataFrame, SparkSession

class PySparkPipeline:
    def __init__(
        self,
        df1: DataFrame,
        df2: DataFrame,
        spark: SparkSession,
        expected=None,
        threshold=3,
        match_column="0",  # default to "0" for ID
        trim=0,
    ):
        if expected is None:
            expected = {}
        self.expected = expected
        self.df1 = df1
        self.df2 = df2
        self.spark = spark
        self.threshold = threshold
        self.match_column = match_column
        self.trim = trim

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.precision = 0.0
        self.recall = 0.0

    def evaluate(self):
        cols = ['1', '2', '3', '4', '5']

        # Ground truth: all pairs with same ID
        ground_truth = self.df1.join(self.df2, on=[self.match_column], how="inner")

        # Join with aliases for column-wise comparison
        joined = self.df1.alias("a").join(
            self.df2.alias("b"),
            F.col(f"a.{self.match_column}") == F.col(f"b.{self.match_column}"),
            how="inner"
        )

        # Count how many columns match
        match_exprs = [
            (F.col(f"a.{c}") == F.col(f"b.{c}")).cast("int") for c in cols
        ]
        joined = joined.withColumn("match_count", sum(match_exprs))

        # Filter rows where at least threshold columns match
        result = joined.filter(F.col("match_count") >= self.threshold)

        # True Positives: IDs present in both ground_truth and result
        tp_df = ground_truth.select(self.match_column).distinct().join(
            result.select(F.col(f"a.{self.match_column}").alias(self.match_column)).distinct(),
            on=self.match_column,
            how="inner"
        )
        tp = tp_df.count()

        # False Negatives: ground truth IDs not in result
        fn = ground_truth.select(self.match_column).distinct().count() - tp

        # False Positives: IDs in result but not in ground_truth
        fp_df = result.select(F.col(f"a.{self.match_column}").alias(self.match_column)).distinct().join(
            ground_truth.select(self.match_column).distinct(),
            on=self.match_column,
            how="left_anti"
        )
        fp = fp_df.count()

        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        self.recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Optionally, store result DataFrame for further inspection
        self.result = result

    def printResults(self):
        print(f"Expected: {self.expected}")
        print(f"True Positives (tp): {self.tp}")
        print(f"False Positives (fp): {self.fp}")
        print(f"False Negatives (fn): {self.fn}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")