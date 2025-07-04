"""Simple dataset loader for entity matching experiments."""
import os
import pandas as pd
import random
import string
import jellyfish
import hashlib

# Set up data path
DATA_PATH = os.path.join(os.path.dirname(__file__), "../../data")


class DataFrameCollection:
    """Collection of DataFrames with method chaining."""

    def __init__(self, dataframes,  excluded_columns=None):
        self.dataframes = dataframes
        self.excluded_columns = excluded_columns or []
    
    def _generate_random_soundex(self):
        """Generate a random fake Soundex code."""
        first_letter = random.choice(string.ascii_uppercase)
        digits = ''.join(random.choices("0123456789", k=3))
        return first_letter + digits

    def _generate_fake_records(self, n_records, n_columns=5):
        """Generate a list of fake records with Soundex codes."""
        return [
            {
                str(0): "fake",
                **{str(i): self._generate_random_soundex() for i in range(1, n_columns + 1)}
            }
            for _ in range(n_records)
        ]
    
    def _add_noise(self, frac):
        """Add synthetic fake rows."""
        noisy_dfs = []
        for df in self.dataframes:
            noise_df = pd.DataFrame(self._generate_fake_records(int(len(df) * frac)))
            combined = pd.concat([df, noise_df], ignore_index=True)
            noisy_dfs.append(combined)
        return DataFrameCollection(noisy_dfs, excluded_columns=self.excluded_columns)

    def exclude(self, columns):
        """Set columns to exclude from transformations like soundex."""
        return DataFrameCollection(self.dataframes, excluded_columns=columns)
    
    def soundex(self, except_columns=None):
        """Apply Soundex encoding to all string columns, except listed ones."""

        encoded_dfs = []
        for df in self.dataframes:
            new_df = df.copy()
            for col in new_df.columns:
                if col in self.excluded_columns:
                    continue
                if new_df[col].dtype == object:
                    new_df[col] = new_df[col].apply(lambda x: jellyfish.soundex(str(x)) if pd.notnull(x) else x)
            encoded_dfs.append(new_df)

        return DataFrameCollection(encoded_dfs, excluded_columns=self.excluded_columns)
    
    def hash_sha256(self):
        """Apply SHA-256 hash to all object columns, except excluded ones."""
        hashed_dfs = []
        for df in self.dataframes:
            new_df = df.copy()
            for col in new_df.columns:
                if col in self.excluded_columns:
                    continue
                if new_df[col].dtype == object:
                    new_df[col] = new_df[col].apply(
                        lambda x: hashlib.sha256(str(x).encode()).hexdigest() if pd.notnull(x) else x
                    )
            hashed_dfs.append(new_df)

        return DataFrameCollection(hashed_dfs, excluded_columns=self.excluded_columns)

    def noise_50(self):
        return self._add_noise(0.5)

    def noise_100(self):
        return self._add_noise(1.0)

    def __getitem__(self, idx):
        return self.dataframes[idx]

    def __iter__(self):
        return iter(self.dataframes)

    def __len__(self):
        return len(self.dataframes)


class MyDatasets:
    """Simple dataset loader."""
    
    def _load_datasets(self, size, matches):
        """Load A, B, C, D, E datasets."""
        files = [f"{size}_{letter}_{matches}.csv" for letter in "ABCDE"]
        datasets = []
        
        for file in files:
            path = os.path.join(DATA_PATH, file)
            df = pd.read_csv(path)[['0', '1', '2', '3', '4', '5']]
            datasets.append(df)
        
        return DataFrameCollection(datasets)
    
    def size_1000(self):
        return self._load_datasets("1000", "250")
    
    def size_10000(self):
        return self._load_datasets("10000", "2500")
    
    def size_50000(self):
        return self._load_datasets("50000", "12500")
    
    def size_75000(self):
        return self._load_datasets("75000", "18750")
    
    def size_100000(self):
        return self._load_datasets("100000", "25000")


# Create global instance for easy access
createDataFrames = MyDatasets()
