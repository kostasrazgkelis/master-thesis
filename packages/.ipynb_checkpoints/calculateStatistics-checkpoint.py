# import numpy as np
# import time
# from collections import defaultdict
# import pandas as pd
# import multiprocessing as mp

# class DatasetEvaluator:
#     def __init__(self, df1, df2, expected={}, threshold=3):
#         self.df1 = df1
#         self.df2 = df2
#         self.expected = expected
#         self.threshold = threshold
#         self.ground_truth_ids = np.intersect1d(df1[0], df2[0])
        
#         self.tp = 0
#         self.fp = 0
#         self.fn = 0
#         self.precision = 0.0
#         self.recall = 0.0
#         self.elapsed_time = 0.0

#     def _is_similar(self, row1, row2):
#         return np.sum(np.array(row1) == np.array(row2)) >= self.threshold

#     def _chunkify(self, combined):
#         return tuple(combined[i:i + 4] for i in range(0, len(combined), 4))

#     def _match_worker(self, args):
#         value, combined, df2_buckets_keys = args
#         row1 = self._chunkify(combined)
#         for data in df2_buckets_keys:
#             row2 = self._chunkify(data)
#             if self._is_similar(row1, row2):
#                 return (data, value)
#         return None

#     def evaluate(self):
#         df1_proc = self.df1.apply(lambda x: (x[0], ''.join(map(str, x[1:]))), axis=1).to_numpy()
#         df2_proc = self.df2.apply(lambda x: (x[0], ''.join(map(str, x[1:]))), axis=1).to_numpy()

#         df2_buckets = defaultdict(list)
#         for row in df2_proc:
#             df2_buckets[row[1]].append(row[0])

#         df2_keys = list(df2_buckets.keys())

#         start_time = time.time()

#         # Parallel processing
#         with mp.Pool(mp.cpu_count()) as pool:
#             args = [(value, combined, df2_keys) for value, combined in df1_proc]
#             results = pool.map(self._match_worker, args)

#         matched = set()
#         fp = 0

#         for result in results:
#             if result:
#                 key, value = result
#                 df2_buckets[key].append(value)

#         for row in df2_buckets:
#             bucket = df2_buckets[row]
#             if len(bucket) > 1:
#                 for id_ in set(bucket[1:]):
#                     if id_ in self.ground_truth_ids:
#                         matched.add(id_)
#                     else:
#                         fp += 1

#         tp = sum(1 for x in matched if x in self.ground_truth_ids)
#         fn = len(self.ground_truth_ids) - tp

#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0

#         self.elapsed_time = time.time() - start_time
#         self.tp = tp
#         self.fp = fp
#         self.fn = fn
#         self.precision = precision
#         self.recall = recall

#     def printResults(self):
#         print(f"Expected: {self.expected}")
#         print(f"Ground Truth Size: {len(self.ground_truth_ids)}")
#         print(f"True Positives: {self.tp}")
#         print(f"False Positives: {self.fp}")
#         print(f"False Negatives: {self.fn}")
#         print(f"Precision: {self.precision:.4f}")
#         print(f"Recall: {self.recall:.4f}")
#         print(f"Elapsed Time: {self.elapsed_time:.2f} seconds")

import numpy as np
import time
from collections import defaultdict

class DatasetEvaluator:
    def __init__(self, df1, df2, expected={}, threshold=3, match_column=0):
        """
        df1, df2: pandas DataFrames with columns [id, col1, ..., col5]
        expected: dictionary with keys 'tp', 'fp', 'fn'
        threshold: number of matching columns to count as a match (default=3)
        """
        self.df1 = df1
        self.df2 = df2
        self.expected = expected
        self.threshold = threshold
        self.match_column = match_column
        self.ground_truth_ids = np.intersect1d(df1[self.match_column], df2[self.match_column])
        
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.precision = 0.0
        self.recall = 0.0
        self.elapsed_time = 0.0

    def _is_similar(self, row1, row2):
        return np.sum(np.array(row1) == np.array(row2)) >= self.threshold

    def evaluate(self):
        df1_proc = self.df1.apply(lambda x: (x[self.match_column], ''.join(map(str, x[1:]))), axis=1).to_numpy()
        df2_proc = self.df2.apply(lambda x: (x[self.match_column], ''.join(map(str, x[1:]))), axis=1).to_numpy()

        id_df1 = np.array([row[0] for row in df1_proc])
        id_df2 = np.array([row[0] for row in df2_proc])

        # Build bucket of df2 rows grouped by values
        df2_buckets = defaultdict(list)
        for row in df2_proc:
            df2_buckets[row[1]].append(row[0])

        matched = set()
        fp = 0

        start_time = time.time()

        for value, combined in df1_proc:
            row1 = tuple(combined[i:i + 4] for i in range(0, len(combined), 4))

            for data in df2_buckets:
                row2 = tuple(data[i:i + 4] for i in range(0, len(data), 4))
                if self._is_similar(row1, row2):
                    df2_buckets[data].append(value)
                    break

        self.elapsed_time = time.time() - start_time

        for row in df2_buckets:
            bucket = df2_buckets[row]
            if len(bucket) > 1:
                for id_ in set(bucket[1:]):
                    if id_ in self.ground_truth_ids:                                                                                 
                        matched.add(id_)
                    else:
                        fp += 1

        tp = sum(1 for x in matched if x in self.ground_truth_ids)
        fn = len(self.ground_truth_ids) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # Save results
        self.tp = tp
        self.fp = fp
        self.fn = fn
        self.precision = precision
        self.recall = recall

    def printResults(self):
        # Print and assert
        print(f"Expected: {self.expected}")
        print(f"Ground Truth Size: {len(self.ground_truth_ids)}")
        print(f"True Positives: {self.tp}")
        print(f"False Positives: {self.fp}")
        print(f"False Negatives: {self.fn}")
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")
        print(f"Elapsed Time: {self.elapsed_time:.2f} seconds")
