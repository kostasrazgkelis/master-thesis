"""
Dataset Generation and Loading Utilities

This module provides utilities for loading and managing multiple datasets
of different sizes for entity matching experiments.
"""
from typing import List
import os
import pandas as pd

FILE_PATHS = "/data/thesis/packages/data"


class MyDatasets:
    """
    A utility class for loading pre-generated datasets of various sizes.
    
    The datasets are organized by size and contain matching records for
    entity matching experiments. Each size category includes 5 datasets
    (A, B, C, D, E) with a specified number of total records and matches.
    """
    
    def __init__(self):
        """
        Initialize the dataset loader.
        
        Args:
            data_path (str): Path to the directory containing the datasets
        """
        self.file_paths = FILE_PATHS

    def _load_datasets(self, prefix: str, matches: str) -> List[pd.DataFrame]:
        """
        Load a set of datasets with the given prefix and match count.
        
        Args:
            prefix (str): Size prefix for the dataset files (e.g., "1000", "10000")
            matches (str): Number of matches in the datasets
            
        Returns:
            List[pd.DataFrame]: List of 5 pandas DataFrames (A, B, C, D, E)
            
        Raises:
            FileNotFoundError: If any of the dataset files cannot be found
        """
        files = [
            f"{prefix}_A_{matches}.csv",
            f"{prefix}_B_{matches}.csv", 
            f"{prefix}_C_{matches}.csv",
            f"{prefix}_D_{matches}.csv",
            f"{prefix}_E_{matches}.csv",
        ]

        datasets = []
        for file in files:
            file_path = os.path.join(self.file_paths, file)
            try:
                df = pd.read_csv(file_path, header=0)
                datasets.append(df.copy())
            except FileNotFoundError as exc:
                raise FileNotFoundError(f"Dataset file not found: {file_path}") from exc
             
        return datasets

    @property
    def size_1000(self) -> List[pd.DataFrame]:
        """
        Load small datasets with 1,000 records each and 250 matches.
        
        Returns:
            List[pd.DataFrame]: 5 datasets with 1K records each
        """
        return self._load_datasets("1000", "250")

    @property
    def size_10000(self) -> List[pd.DataFrame]:
        """
        Load medium datasets with 10,000 records each and 2,500 matches.
        
        Returns:
            List[pd.DataFrame]: 5 datasets with 10K records each
        """
        return self._load_datasets("10000", "2500")

    @property
    def size_50000(self) -> List[pd.DataFrame]:
        """
        Load large datasets with 50,000 records each and 12,500 matches.
        
        Returns:
            List[pd.DataFrame]: 5 datasets with 50K records each
        """
        return self._load_datasets("50000", "12500")

    @property
    def size_75000(self) -> List[pd.DataFrame]:
        """
        Load extra large datasets with 75,000 records each and 18,750 matches.
        
        Returns:
            List[pd.DataFrame]: 5 datasets with 75K records each
        """
        return self._load_datasets("75000", "18750")

    @property
    def size_100000(self) -> List[pd.DataFrame]:
        """
        Load very large datasets with 100,000 records each and 25,000 matches.
        
        Returns:
            List[pd.DataFrame]: 5 datasets with 100K records each
        """
        return self._load_datasets("100000", "25000")
