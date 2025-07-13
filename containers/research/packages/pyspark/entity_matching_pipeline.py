"""
Multi-Party Entity Matching Pipeline for PySpark

This module provides a pipeline for entity matching (record linkage)
that can process multiple datasets using phonetic and similarity-based matching.
"""

from dataclasses import dataclass
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import List, Dict, Tuple, Any


@dataclass
class MatchingConfig:
    """Configuration for entity matching pipeline"""

    id_column: str = "0"
    match_columns: List[str] = None
    origin_column: str = "origin"
    similarity_threshold: float = 0.6
    min_matching_columns: int = 3
    use_soundex: bool = True
    bucket_assignment_strategy: str = "max_similarity"

    def __post_init__(self):
        if self.match_columns is None:
            self.match_columns = ["1", "2", "3", "4", "5"]


class MultiPartyRecordLinkage:
    """
    Pipeline for multi-party entity matching using PySpark.
    Matches records from multiple datasets based on similarity scores.
    """

    def __init__(self, spark: SparkSession, config: MatchingConfig = None):
        self.spark = spark
        self.config = config or MatchingConfig()
        self.ground_truth_cache = None

    def preprocess_dataframe(self, df: DataFrame, origin_id: int) -> DataFrame:
        """Add origin column and apply transformations to dataframe."""
        df = df.withColumn(self.config.origin_column, F.lit(origin_id))

        if self.config.use_soundex:
            df = self._apply_soundex_transformation(df)

        return df

    def _apply_soundex_transformation(self, df: DataFrame) -> DataFrame:
        """Apply soundex transformation to matching columns."""
        from packages.utils.spark_udfs import soundex_udf

        for col in self.config.match_columns:
            if col in df.columns:
                df = df.withColumn(col, soundex_udf(F.col(col)))
        return df

    def create_entity_keys(self, df: DataFrame, alias_prefix: str = "") -> DataFrame:
        """Create entity keys by concatenating matching columns."""
        entity_key_expr = F.concat_ws(
            "", *[F.col(c) for c in self.config.match_columns]
        )
        id_alias = f"{alias_prefix}id" if alias_prefix else "id"

        select_cols = [
            F.col(self.config.id_column).alias(id_alias),
            entity_key_expr.alias("entity_key"),
        ] + [F.col(c) for c in self.config.match_columns]

        return df.select(*select_cols)

    def calculate_similarity_matrix(
        self, left_df: DataFrame, right_df: DataFrame
    ) -> DataFrame:
        """Calculate similarity scores between all pairs from left and right dataframes."""
        joined = left_df.alias("left").crossJoin(right_df.alias("right"))

        match_expressions = []
        for col in self.config.match_columns:
            match_expr = (F.col(f"left.{col}") == F.col(f"right.{col}")).cast("int")
            match_expressions.append(match_expr)

        total_matches = sum(match_expressions)
        total_columns = len(self.config.match_columns)
        similarity_score = total_matches / total_columns

        return (
            joined.withColumn("match_count", total_matches)
            .withColumn("similarity_score", similarity_score)
            .withColumn("total_columns", F.lit(total_columns))
        )

    def apply_similarity_threshold(self, similarity_df: DataFrame) -> DataFrame:
        """Filter pairs based on similarity threshold and minimum matching columns."""
        return similarity_df.filter(
            (F.col("similarity_score") >= self.config.similarity_threshold)
            & (F.col("match_count") >= self.config.min_matching_columns)
        )

    def assign_to_buckets(self, filtered_pairs: DataFrame) -> DataFrame:
        """Assign entities to buckets based on the configured strategy."""
        if self.config.bucket_assignment_strategy == "max_similarity":
            return self._assign_by_max_similarity(filtered_pairs)
        else:
            return self._assign_by_threshold(filtered_pairs)

    def _assign_by_max_similarity(self, filtered_pairs: DataFrame) -> DataFrame:
        """Assign entities to buckets by selecting the highest similarity match."""
        max_similarity = (
            filtered_pairs.groupBy("left.left_id")
            .agg(F.max("similarity_score").alias("max_similarity"))
            .withColumnRenamed("left_id", "max_left_id")
        )

        best_matches = filtered_pairs.join(
            max_similarity,
            (filtered_pairs["left.left_id"] == max_similarity["max_left_id"])
            & (filtered_pairs["similarity_score"] == max_similarity["max_similarity"]),
        ).select(
            F.col("left.left_id").alias("entity_id"),
            F.col("right.right_id").alias("bucket_id"),
            F.col("similarity_score"),
        )

        buckets = best_matches.groupBy("bucket_id").agg(
            F.collect_list("entity_id").alias("assigned_entities"),
            F.avg("similarity_score").alias("avg_similarity"),
            F.count("entity_id").alias("bucket_size"),
        )

        return buckets

    def _assign_by_threshold(self, filtered_pairs: DataFrame) -> DataFrame:
        """Assign entities to buckets by accepting all pairs above threshold."""
        buckets = (
            filtered_pairs.groupBy("right.right_id")
            .agg(
                F.collect_list("left.left_id").alias("assigned_entities"),
                F.avg("similarity_score").alias("avg_similarity"),
                F.count("left.left_id").alias("bucket_size"),
            )
            .withColumnRenamed("right_id", "bucket_id")
        )

        return buckets

    def evaluate_buckets(
        self, buckets: DataFrame, ground_truth_df: DataFrame
    ) -> Dict[str, Any]:
        """Evaluate bucket quality against ground truth."""
        if self.ground_truth_cache is None:
            self.ground_truth_cache = ground_truth_df.cache()

        gt_count = self.ground_truth_cache.count()

        tp_buckets = buckets.filter(
            F.array_contains(F.col("assigned_entities"), F.col("bucket_id"))
        ).join(
            self.ground_truth_cache,
            buckets.bucket_id == self.ground_truth_cache[self.config.id_column],
            how="inner",
        )

        tp = tp_buckets.count()
        fp = buckets.filter(
            ~F.array_contains(F.col("assigned_entities"), F.col("bucket_id"))
        ).count()
        fn = gt_count - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        total_buckets = buckets.count()
        avg_bucket_size = buckets.agg(F.avg("bucket_size")).collect()[0][0] or 0.0

        return {
            "ground_truth": gt_count,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "total_buckets": total_buckets,
            "average_bucket_size": avg_bucket_size,
        }

    def run_pipeline(
        self,
        left_df: DataFrame,
        right_df: DataFrame,
        left_origin: int = 1,
        right_origin: int = 2,
    ) -> Tuple[DataFrame, Dict[str, Any]]:
        """Run the complete entity matching pipeline."""
        print("Starting Multi-Party Entity Matching Pipeline...")

        print("Step 1: Preprocessing dataframes...")
        left_processed = self.preprocess_dataframe(left_df, left_origin)
        right_processed = self.preprocess_dataframe(right_df, right_origin)

        print("Step 2: Creating entity keys...")
        left_entities = self.create_entity_keys(left_processed, "left_")
        right_entities = self.create_entity_keys(right_processed, "right_")

        print("Step 3: Calculating similarity matrix...")
        similarity_matrix = self.calculate_similarity_matrix(
            left_entities, right_entities
        )

        print("Step 4: Applying similarity threshold...")
        filtered_pairs = self.apply_similarity_threshold(similarity_matrix)

        print("Step 5: Assigning entities to buckets...")
        buckets = self.assign_to_buckets(filtered_pairs)
        buckets.cache()

        print("Step 6: Calculating ground truth...")
        ground_truth = (
            left_processed.join(
                right_processed, on=[self.config.id_column], how="inner"
            )
            .select(F.col(self.config.id_column))
            .distinct()
        )

        print("Step 7: Evaluating results...")
        metrics = self.evaluate_buckets(buckets, ground_truth)

        print("Pipeline completed successfully!")
        return buckets, metrics

    def print_metrics(self, metrics: Dict[str, Any]):
        """Print evaluation metrics."""
        print("\n" + "=" * 50)
        print("ENTITY MATCHING EVALUATION RESULTS")
        print("=" * 50)
        print(f"Ground Truth Matches: {metrics['ground_truth']}")
        print(f"True Positives: {metrics['true_positives']}")
        print(f"False Positives: {metrics['false_positives']}")
        print(f"False Negatives: {metrics['false_negatives']}")
        print("-" * 30)
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print("-" * 30)
        print(f"Total Buckets: {metrics['total_buckets']}")
        print(f"Average Bucket Size: {metrics['average_bucket_size']:.2f}")
        print("=" * 50)

    def preprocess_multiple_dataframes(
        self, dataframes_with_origins: List[Tuple[DataFrame, int]]
    ) -> DataFrame:
        """Preprocess multiple dataframes by adding origins and union them together."""
        if not dataframes_with_origins:
            raise ValueError("At least one dataframe must be provided")

        processed_dfs = []
        for df, origin_id in dataframes_with_origins:
            processed_df = self.preprocess_dataframe(df, origin_id)
            processed_dfs.append(processed_df)

        unified_df = processed_dfs[0]
        for df in processed_dfs[1:]:
            unified_df = unified_df.union(df)

        return unified_df

    def run_multi_dataset_pipeline(
        self,
        left_df: DataFrame,
        right_dataframes_with_origins: List[Tuple[DataFrame, int]],
        left_origin: int = 1,
    ) -> Tuple[DataFrame, Dict[str, Any]]:
        """Run the entity matching pipeline with one left dataset and multiple right datasets."""
        print("Starting Multi-Dataset Entity Matching Pipeline...")
        print(
            f"Processing 1 left dataset vs {len(right_dataframes_with_origins)} right datasets"
        )

        print("Step 1: Preprocessing left dataframe...")
        left_processed = self.preprocess_dataframe(left_df, left_origin)

        print("Step 1b: Preprocessing and union right dataframes...")
        right_unified = self.preprocess_multiple_dataframes(
            right_dataframes_with_origins
        )
        print(f"Unified right dataset has {right_unified.count()} total records")

        print("Step 2: Creating entity keys...")
        left_entities = self.create_entity_keys(left_processed, "left_")
        right_entities = self.create_entity_keys(right_unified, "right_")

        print("Step 3: Calculating similarity matrix...")
        similarity_matrix = self.calculate_similarity_matrix(
            left_entities, right_entities
        )

        print("Step 4: Applying similarity threshold...")
        filtered_pairs = self.apply_similarity_threshold(similarity_matrix)

        print("Step 5: Assigning entities to buckets...")
        buckets = self.assign_to_buckets(filtered_pairs)
        buckets.cache()

        print("Step 6: Calculating ground truth...")
        ground_truth = (
            left_processed.join(right_unified, on=[self.config.id_column], how="inner")
            .select(F.col(self.config.id_column))
            .distinct()
        )

        print("Step 7: Evaluating results...")
        metrics = self.evaluate_buckets(buckets, ground_truth)

        print("Multi-dataset pipeline completed successfully!")
        return buckets, metrics

    def run_full_multi_dataset_pipeline(
        self,
        left_dataframes_with_origins: List[Tuple[DataFrame, int]],
        right_dataframes_with_origins: List[Tuple[DataFrame, int]],
    ) -> Tuple[DataFrame, Dict[str, Any]]:
        """Run the entity matching pipeline with multiple datasets on both sides."""
        print("Starting Full Multi-Dataset Entity Matching Pipeline...")
        print(
            f"Processing {len(left_dataframes_with_origins)} left datasets vs {len(right_dataframes_with_origins)} right datasets"
        )

        print("Step 1a: Preprocessing and union left dataframes...")
        left_unified = self.preprocess_multiple_dataframes(left_dataframes_with_origins)
        print(f"Unified left dataset has {left_unified.count()} total records")

        print("Step 1b: Preprocessing and union right dataframes...")
        right_unified = self.preprocess_multiple_dataframes(
            right_dataframes_with_origins
        )
        print(f"Unified right dataset has {right_unified.count()} total records")

        print("Step 2: Creating entity keys...")
        left_entities = self.create_entity_keys(left_unified, "left_")
        right_entities = self.create_entity_keys(right_unified, "right_")

        print("Step 3: Calculating similarity matrix...")
        similarity_matrix = self.calculate_similarity_matrix(
            left_entities, right_entities
        )

        print("Step 4: Applying similarity threshold...")
        filtered_pairs = self.apply_similarity_threshold(similarity_matrix)

        print("Step 5: Assigning entities to buckets...")
        buckets = self.assign_to_buckets(filtered_pairs)
        buckets.cache()

        print("Step 6: Calculating ground truth...")
        ground_truth = (
            left_unified.join(right_unified, on=[self.config.id_column], how="inner")
            .select(F.col(self.config.id_column))
            .distinct()
        )

        print("Step 7: Evaluating results...")
        metrics = self.evaluate_buckets(buckets, ground_truth)

        print("Full multi-dataset pipeline completed successfully!")
        return buckets, metrics


def create_sample_pipeline(
    spark: SparkSession,
    similarity_threshold: float = 0.6,
    min_matching_columns: int = 3,
) -> MultiPartyRecordLinkage:
    """Create a pre-configured pipeline for typical use cases."""
    config = MatchingConfig(
        similarity_threshold=similarity_threshold,
        min_matching_columns=min_matching_columns,
        use_soundex=True,
        bucket_assignment_strategy="max_similarity",
    )
    return MultiPartyRecordLinkage(spark, config)


def create_dataset_list_with_origins(
    dataframes: List[DataFrame], origin_ids: List[int] = None
) -> List[Tuple[DataFrame, int]]:
    """Helper function to create a list of (dataframe, origin_id) tuples."""
    if origin_ids is None:
        origin_ids = list(range(1, len(dataframes) + 1))

    if len(dataframes) != len(origin_ids):
        raise ValueError("Number of dataframes must match number of origin IDs")

    return list(zip(dataframes, origin_ids))


def create_multi_dataset_pipeline(
    spark: SparkSession,
    left_dataframes: List[DataFrame],
    right_dataframes: List[DataFrame],
    left_origins: List[int] = None,
    right_origins: List[int] = None,
    similarity_threshold: float = 0.6,
    min_matching_columns: int = 3,
) -> Tuple[DataFrame, Dict[str, Any]]:
    """Convenience function to run multi-dataset pipeline with simple inputs."""
    config = MatchingConfig(
        similarity_threshold=similarity_threshold,
        min_matching_columns=min_matching_columns,
        use_soundex=True,
        bucket_assignment_strategy="max_similarity",
    )

    pipeline = MultiPartyRecordLinkage(spark, config)
    left_datasets_with_origins = create_dataset_list_with_origins(
        left_dataframes, left_origins
    )
    right_datasets_with_origins = create_dataset_list_with_origins(
        right_dataframes, right_origins
    )

    return pipeline.run_full_multi_dataset_pipeline(
        left_datasets_with_origins, right_datasets_with_origins
    )


def run_entity_matching(
    spark: SparkSession,
    left_df: DataFrame,
    right_dataframes: List[DataFrame],
    similarity_threshold: float = 0.6,
    min_matching_columns: int = 3,
) -> Tuple[DataFrame, Dict[str, Any]]:
    """
    Simple function to run entity matching pipeline.

    Args:
        spark: Spark session
        left_df: Left dataframe (e.g., df1)
        right_dataframes: List of right dataframes (e.g., [df2, df3, df4])
        similarity_threshold: Minimum similarity for matching
        min_matching_columns: Minimum columns that must match

    Returns:
        Tuple of (buckets_dataframe, evaluation_metrics)
    """
    config = MatchingConfig(
        similarity_threshold=similarity_threshold,
        min_matching_columns=min_matching_columns,
        use_soundex=True,
        bucket_assignment_strategy="max_similarity",
    )

    pipeline = MultiPartyRecordLinkage(spark, config)

    # Create list of right dataframes with origins starting from 2
    right_with_origins = create_dataset_list_with_origins(
        right_dataframes, list(range(2, len(right_dataframes) + 2))
    )

    return pipeline.run_multi_dataset_pipeline(
        left_df, right_with_origins, left_origin=1
    )
