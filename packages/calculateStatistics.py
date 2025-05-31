import numpy as np
from collections import defaultdict
from functools import lru_cache
from packages.timeDecorator import timeit


class DatasetEvaluator:
    def __init__(self, df1, df2, expected={}, threshold=3, match_column=0, trim=0):
        """
        df1, df2: pandas DataFrames with columns [id, col1, ..., col5]
        expected: dictionary with keys 'tp', 'fp', 'fn'
        threshold: number of matching columns to count as a match (default=3)
        """
        self.expected = expected

        self.df1 = df1
        self.df2 = df2
        self.threshold = threshold
        self.match_column = match_column
        self.trim = trim

        self.ground_truth_ids = np.intersect1d(
            df1[self.match_column], df2[self.match_column]
        )

        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.precision = 0.0
        self.recall = 0.0
        self.elapsed_time = 0.0

    @lru_cache(maxsize=None)
    def _is_similar(self, row1, row2):
        return np.sum(np.array(row1) == np.array(row2)) >= self.threshold

    @lru_cache(maxsize=None)
    def fast_chunk(self, s: str):
        step = len(s) // (5)
        return np.array([s[i : i + step] for i in range(0, len(s), step)])

    @timeit
    def preproccess(self):

        def combine_and_trim_chunks(row):
            chunks = [str(x) for x in row[1:]]  # skip ID
            trimmed_chunks = [c[: -self.trim] if self.trim > 0 else c for c in chunks]
            combined = "".join(trimmed_chunks)
            return (row[self.match_column], combined)

        self.df1_proc = self.df1.apply(combine_and_trim_chunks, axis=1).to_numpy()
        self.df2_proc = self.df2.apply(combine_and_trim_chunks, axis=1).to_numpy()

        self.df2_keys = np.array([row[0] for row in self.df2_proc])
        self.df2_vals = np.array([row[1] for row in self.df2_proc])

    @timeit
    def evaluate(self):
        # Build df2 buckets: combined string -> [ids]
        self.hashed_bucket = defaultdict(list)
        for k, v in zip(self.df2_vals, self.df2_keys):
            self.hashed_bucket[k].append(v)

        # Precompute chunked df2 strings once
        self.chunked_df2 = {k: self.fast_chunk(k) for k in self.hashed_bucket}

        # Precompute chunked df1 strings once
        self.chunked_df1 = [
            (match_id, self.fast_chunk(combined))
            for match_id, combined in self.df1_proc
        ]

        for match_id, row1_chunks in self.chunked_df1:
            for data_key, row2_chunks in self.chunked_df2.items():
                match_count = np.count_nonzero(row1_chunks == row2_chunks)
                if match_count >= self.threshold:
                    self.hashed_bucket[data_key].append(match_id)
                    break

    @timeit
    def calculateStatistics(self):
        tp = 0
        fp = 0
        ground_truth_set = set(self.ground_truth_ids)

        for bucket in self.hashed_bucket.values():
            if len(bucket) > 1:
                if any(item in ground_truth_set for item in bucket[1:]):
                    tp += 1
                else:
                    fp += 1

        self.tp = tp
        self.fp = fp
        self.fn = len(ground_truth_set) - tp
        self.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        self.recall = tp / (tp + self.fn) if (tp + self.fn) > 0 else 0

    def printResults(self):
        # Print and assert
        print(f"Expected: {self.expected}")
        print(f"Ground Truth Size: {len(self.ground_truth_ids)}")
        print(f"True Positives: {self.tp}")
        print(f"False Positives: {self.fp}")
        print(f"False Negatives: {self.fn}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")
