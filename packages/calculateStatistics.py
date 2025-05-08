import numpy as np
import time
from collections import defaultdict
from functools import lru_cache

class DatasetEvaluator:
    def __init__(self, df1, df2, expected={}, threshold=3, match_column=0):
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
        
        self.ground_truth_ids = np.intersect1d(df1[self.match_column], df2[self.match_column])
        
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.precision = 0.0
        self.recall = 0.0
        self.elapsed_time = 0.0

    @lru_cache(maxsize=None)
    def _is_similar(self, row1, row2):
        return np.sum(np.array(row1) == np.array(row2)) >= self.threshold


    def evaluate(self):
        # Preprocess df1 and df2: (ID, combined string)
        df1_proc = self.df1.apply(lambda x: (x[self.match_column], ''.join(map(str, x[1:]))), axis=1).to_numpy()
        df2_proc = self.df2.apply(lambda x: (x[self.match_column], ''.join(map(str, x[1:]))), axis=1).to_numpy()
        
        # Separate IDs and combined strings from df2
        df2_keys = np.array([row[0] for row in df2_proc if row[0] in self.ground_truth_ids] )
        df2_vals = np.array([row[1] for row in df2_proc])
        
        # Build df2 buckets: combined string -> [ids]
        df2_buckets = defaultdict(list)
        for k, v in zip(df2_vals, df2_keys):
            df2_buckets[k].append(v)
        
        # Precompute chunks for df2 values
        chunked_df2 = {
            k: np.array([k[i:i+4] for i in range(0, len(k), 4)])
            for k in df2_buckets
        }
        
        # Precompute chunks for df1 values
        chunked_df1 = [
            (match_id, np.array([combined[i:i+4] for i in range(0, len(combined), 4)]))
            for match_id, combined in df1_proc
        ]
        
        # Matching process
        start_time = time.time()
        for match_id, row1_chunks in chunked_df1:
            for data, row2_chunks in chunked_df2.items():
                match_count = np.sum(row1_chunks == row2_chunks)
                if match_count >= self.threshold:
                    df2_buckets[data].append(match_id)
                    break
                    
        self.elapsed_time = time.time() - start_time
    
        # Evaluate precision/recall
        matched = set()
        fp = 0
        ground_truth_ids_np = np.array(list(self.ground_truth_ids))
        
        for bucket in df2_buckets.values():
            if len(bucket) > 1:
                ids_to_check = np.array(bucket[1:])  # exclude the original
                mask = np.isin(ids_to_check, ground_truth_ids_np)
                matched |= set(ids_to_check[mask])
                fp += mask.size - np.count_nonzero(mask)
    
        tp = len(matched)
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

# import numpy as np
# import time
# from collections import defaultdict
# import json

# class DatasetEvaluator:
#     def __init__(self, df1, df2, expected={}, threshold=3, match_column=0):
#         """
#         df1, df2: pandas DataFrames with columns [id, col1, ..., col5]
#         expected: dictionary with keys 'tp', 'fp', 'fn'
#         threshold: number of matching columns to count as a match (default=3)
#         """
#         self.df1 = df1
#         self.df2 = df2
#         self.expected = expected
#         self.threshold = threshold
#         self.match_column = match_column
#         self.ground_truth_ids = np.intersect1d(df1[self.match_column], df2[self.match_column])
        
#         self.tp = 0
#         self.fp = 0
#         self.fn = 0
#         self.precision = 0.0
#         self.recall = 0.0
#         self.elapsed_time = 0.0
#         self.hash_buckets = defaultdict(list)
        

#     def _is_similar(self, row1, row2):
#         return np.sum(np.array(row1) == np.array(row2)) >= 1

#     def _slice_row_string(self, combined_str, chunk_size=4):
#         return tuple(combined_str[i:i + chunk_size] for i in range(0, len(combined_str), chunk_size))
    
#     def evaluate(self):    
#         self.df1_proc = self.df1.apply(lambda x: (x[self.match_column], ''.join(map(str, x[1:]))), axis=1).to_numpy()
#         self.df2_proc = self.df2.apply(lambda x: (x[self.match_column], ''.join(map(str, x[1:]))), axis=1).to_numpy()

#         start_time = time.time()
        
#         # Nested defaultdict: {prefix_key: {suffix: [match_ids]}}
#         self.hash_buckets = defaultdict(lambda: defaultdict(list))
    
#         for match_id, combined in self.df1_proc:
#             prefix_key = combined[:8]
#             suffix = combined[8:]
#             row1 = self._slice_row_string(suffix)
    
#             # Try to find a similar existing suffix in this prefix group
#             found_match = False
#             for existing_suffix in self.hash_buckets[prefix_key]:
#                 row2 = self._slice_row_string(existing_suffix)
#                 if self._is_similar(row1, row2):
#                     self.hash_buckets[prefix_key][existing_suffix].append(match_id)
#                     found_match = True
#                     break
    
#             if not found_match:
#                 self.hash_buckets[prefix_key][suffix].append(match_id)
    

#         for match_id, combined in self.df2_proc:
#             prefix_key = combined[:8]
#             suffix = combined[8:]
#             row1 = self._slice_row_string(suffix)
    
#             # Try to find a similar existing suffix in this prefix group
#             found_match = False
#             for existing_suffix in self.hash_buckets[prefix_key]:
#                 row2 = self._slice_row_string(existing_suffix)
#                 if self._is_similar(row1, row2):
#                     self.hash_buckets[prefix_key][existing_suffix].append(match_id)
#                     found_match = True
#                     break
    
#             if not found_match:
#                 self.hash_buckets[prefix_key][suffix].append(match_id)
                
#         self.elapsed_time = time.time() - start_time

#         print(json.dumps(self.hash_buckets, indent=4))


    
        
#     def calculateStatistics(self):
        
#         fp, tp, fn, precision, recall = 0 , 0 , 0, 0, 0        
#         matched = set()


#         tp = sum(1 for x in matched if x[0] in self.ground_truth_ids)
#         fn = len(self.ground_truth_ids) - tp

#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0

#         # Save results
#         self.tp = tp
#         self.fp = fp
#         self.fn = fn
#         self.precision = precision
#         self.recall = recall

#     def printResults(self):
#         # Print and assert
#         print(f"Expected: {self.expected}")
#         print(f"Ground Truth Size: {len(self.ground_truth_ids)}")
#         print(f"True Positives: {self.tp}")
#         print(f"False Positives: {self.fp}")
#         print(f"False Negatives: {self.fn}")
#         print(f"Precision: {self.precision:.4f}")
#         print(f"Recall: {self.recall:.4f}")
#         print(f"Elapsed Time: {self.elapsed_time:.2f} seconds")
