"""
PySpark Entity Matching Package

This package provides robust, reusable PySpark-based solutions for entity matching
(record linkage) using phonetic and similarity-based matching techniques.
"""

from .entity_matching_pipeline import (
    MultiPartyRecordLinkage,
    MatchingConfig,
    create_sample_pipeline,
    create_dataset_list_with_origins,
    create_multi_dataset_pipeline,
)

__all__ = [
    "MultiPartyRecordLinkage",
    "MatchingConfig",
    "create_sample_pipeline",
    "create_dataset_list_with_origins",
    "create_multi_dataset_pipeline",
]

__version__ = "1.0.0"
