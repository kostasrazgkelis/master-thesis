import pandas as pd
import random
import string


class SyntheticMatcherDataset:
    def __init__(
        self,
        size,
        datasets_ratio=(1, 1),
        ground_truth_ratio=0.25,
        true_positive_ratio=0.66,
        expected=None,
        threshold=3,
    ):
        self.size = size
        self.datasets_ratio = datasets_ratio
        self.ground_truth_ratio = ground_truth_ratio
        self.true_positive_ratio = true_positive_ratio

        self.threshold = threshold
        self.ground_truth_ids = set()
        self.false_negatives = set()

        self.df1 = None
        self.df2 = None
        self.expected = expected
        self.ground_truth_matches = set()  # IDs that should match (True Positives)
        self.false_negatives = set()  # IDs that should match but were made hard
        self._generate()

    @staticmethod
    def _generate_soundex_code():
        """Generate a fake soundex-like 4-character code."""
        return random.choice(string.ascii_uppercase) + "".join(
            random.choices(string.digits, k=3)
        )

    def _generate_dataframe(self, n_rows):
        """Generate a base DataFrame."""
        data = []
        for i in range(n_rows):
            id_ = f"ID{i:05d}"
            soundex_codes = [self._generate_soundex_code() for _ in range(5)]
            data.append([id_] + soundex_codes)
        columns = ["id", "col1", "col2", "col3", "col4", "col5"]
        return pd.DataFrame(data, columns=columns)

    def _modify_columns(self, row, n_to_modify):
        cols = ["col1", "col2", "col3", "col4", "col5"]
        cols_to_modify = random.sample(cols, n_to_modify)
        for col in cols_to_modify:
            row[col] = self._generate_soundex_code()
        return row

    def _generate(self):
        df1 = self._generate_dataframe(self.size)
        df2_rows = []

        n_ground_truth = int(
            self.size * self.ground_truth_ratio * self.datasets_ratio[0]
        )
        n_true_positives = int(n_ground_truth * self.true_positive_ratio)
        n_false_positives = int(n_true_positives * self.true_positive_ratio)

        all_indices = list(df1.index)

        ground_truth_indices = random.sample(all_indices, n_ground_truth)
        remaining_indices = list(set(all_indices) - set(ground_truth_indices))

        # -- TRUE POSITIVES: same ID, >=3 columns match
        tp_indices = random.sample(ground_truth_indices, n_true_positives)
        for idx in tp_indices:
            row = df1.loc[idx].copy()
            self.ground_truth_ids.add(row["id"])
            modified_row = row.copy()

            n_to_modify = random.randint(0, 2)  # At most 2 columns different
            modified_row = self._modify_columns(modified_row, n_to_modify)
            df2_rows.append(modified_row.tolist())

        # -- FALSE POSITIVES: different ID, >=3 columns match
        fp_indices = random.sample(remaining_indices, n_false_positives)
        for idx in fp_indices:
            row = df1.loc[idx].copy()
            modified_row = row.copy()
            modified_row["id"] = f"NEW{random.randint(10000, 99999)}"  # Different ID
            n_to_modify = random.randint(0, 2)  # Still at least 3 match
            modified_row = self._modify_columns(modified_row, n_to_modify)
            df2_rows.append(modified_row.tolist())

        # -- FALSE NEGATIVES: same ID, <3 columns match
        fn_indices = list(set(ground_truth_indices) - set(tp_indices))
        for idx in fn_indices:
            row = df1.loc[idx].copy()
            self.ground_truth_ids.add(row["id"])
            modified_row = row.copy()
            n_to_modify = random.randint(3, 5)  # Less than 3 columns match
            modified_row = self._modify_columns(modified_row, n_to_modify)
            self.false_negatives.add(row["id"])
            df2_rows.append(modified_row.tolist())

        # -- RANDOM NOISE: completely new entries
        n_used = n_true_positives + n_false_positives + len(fn_indices)
        n_random = self.size * self.datasets_ratio[1] - n_used
        for _ in range(n_random):
            new_id = f"NEW{random.randint(10000, 99999)}"
            new_row = [new_id] + [self._generate_soundex_code() for _ in range(5)]
            df2_rows.append(new_row)

        # Final shuffle and assignment
        random.shuffle(df2_rows)
        df2 = pd.DataFrame(df2_rows, columns=df1.columns)

        self.expected["gt"] = len(ground_truth_indices)
        self.expected["tp"] = len(tp_indices)
        self.expected["fp"] = len(fp_indices)
        self.expected["fn"] = len(fn_indices)

        self.df1 = df1
        self.df2 = df2
