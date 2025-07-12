# Entity Matching Pipeline Documentation

## Overview

The **Multi-Party Entity Matching Pipeline** is a clean, refactored PySpark-based solution for entity matching (record linkage) that can process multiple datasets using phonetic and similarity-based matching techniques.

## Architecture

```
Input DataFrames → Preprocessing → Entity Key Generation → Similarity Matrix → 
Threshold Filtering → Bucket Assignment → Evaluation → Output Buckets
```

### Key Components

1. **MultiPartyRecordLinkage**: Main pipeline class 
2. **MatchingConfig**: Configuration dataclass for pipeline parameters
3. **Entity Key Generation**: Creates composite keys from matching columns
4. **Similarity Matrix**: Cross-join with similarity scoring
5. **Bucket Assignment**: Groups similar entities together
6. **Evaluation**: Computes precision, recall, and F1-score

## Features

- ✅ **Simple API**: Easy-to-use `run_entity_matching()` function
- ✅ **Configurable Matching**: Customize similarity thresholds, matching columns, and strategies
- ✅ **Phonetic Matching**: Built-in Soundex transformation support
- ✅ **Multiple Assignment Strategies**: Max similarity or threshold-based bucket assignment
- ✅ **Comprehensive Evaluation**: Precision, recall, F1-score, and bucket statistics
- ✅ **Multi-Dataset Support**: Match one dataset against multiple datasets
- ✅ **Scalable**: Built on PySpark for distributed processing


## Quick Start

### Simple Usage (Recommended)

```python
from packages.pyspark.entity_matching_pipeline import run_entity_matching

# Simple way to match df1 against [df2, df3, df4]
buckets, metrics = run_entity_matching(
    spark=spark,
    left_df=df1,
    right_dataframes=[df2, df3, df4],
    similarity_threshold=0.6,
    min_matching_columns=3
)

# Print results
print("Buckets:")
buckets.show()

print("Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")
```

### Advanced Usage

```python
from packages.pyspark.entity_matching_pipeline import MultiPartyRecordLinkage, MatchingConfig

# Create custom configuration
config = MatchingConfig(
    id_column="0",
    match_columns=['1', '2', '3', '4', '5'],
    similarity_threshold=0.6,
    min_matching_columns=3,
    use_soundex=True
)

# Initialize pipeline
pipeline = MultiPartyRecordLinkage(spark, config)

# Run pipeline
buckets, metrics = pipeline.run_pipeline(df1, df2)

# View results
pipeline.print_metrics(metrics)
```

## Main Use Case: df1 vs [df2, df3, df4, ...]

The pipeline is optimized for the common use case where you have one primary dataset (df1) that you want to match against multiple other datasets:

```python
# Load your datasets
df1 = spark.read.csv("data/df1.csv", header=False).toDF("0", "1", "2", "3", "4", "5")
df2 = spark.read.csv("data/df2.csv", header=False).toDF("0", "1", "2", "3", "4", "5")
df3 = spark.read.csv("data/df3.csv", header=False).toDF("0", "1", "2", "3", "4", "5")
df4 = spark.read.csv("data/df4.csv", header=False).toDF("0", "1", "2", "3", "4", "5")

# Run matching - simple one-liner
buckets, metrics = run_entity_matching(
    spark=spark,
    left_df=df1,
    right_dataframes=[df2, df3, df4],
    similarity_threshold=0.6,
    min_matching_columns=3
)

# Access results
print(f"Total buckets: {buckets.count()}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

## Configuration Options

### MatchingConfig Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `id_column` | str | "0" | Column name for unique identifiers |
| `match_columns` | List[str] | ['1', '2', '3', '4', '5'] | Columns to use for matching |
| `origin_column` | str | "origin" | Column to identify data source |
| `similarity_threshold` | float | 0.6 | Minimum similarity score for matching |
| `min_matching_columns` | int | 3 | Minimum number of columns that must match |
| `use_soundex` | bool | True | Whether to apply soundex transformation |
| `bucket_assignment_strategy` | str | "max_similarity" | "max_similarity" or "threshold" |

### Bucket Assignment Strategies

1. **max_similarity**: For each entity, assign to the bucket with highest similarity score
2. **threshold**: Assign entities to all buckets above the similarity threshold

## Pipeline Steps

The pipeline follows these clean, well-defined steps:

### 1. Preprocessing
- Add origin column to identify data source
- Apply phonetic transformations (Soundex) if configured
- Prepare dataframes for matching

### 2. Entity Key Generation
- Create composite keys by concatenating matching columns
- Generate alias prefixes for join operations


### 3. Similarity Matrix Calculation
- Perform cross-join between left and right entities
- Calculate column-wise matches
- Compute similarity scores as ratio of matching columns


### 4. Threshold Filtering
- Filter pairs based on similarity threshold
- Apply minimum matching columns constraint


### 5. Bucket Assignment
- Apply configured assignment strategy
- Group entities into buckets
- Calculate bucket statistics


### 6. Ground Truth Calculation
- Calculate ground truth for evaluation


### 7. Evaluation
- Compare results against ground truth
- Calculate precision, recall, and F1-score
- Generate bucket quality metrics

## API Reference

### Main Functions

#### `run_entity_matching(spark, left_df, right_dataframes, similarity_threshold=0.6, min_matching_columns=3)`

**The recommended way to use the pipeline.** Simple function that handles the most common use case.

**Parameters:**
- `spark`: SparkSession
- `left_df`: Left dataframe (e.g., df1)
- `right_dataframes`: List of right dataframes (e.g., [df2, df3, df4])
- `similarity_threshold`: Minimum similarity for matching (default: 0.6)
- `min_matching_columns`: Minimum columns that must match (default: 3)

**Returns:**
- Tuple of `(buckets_dataframe, evaluation_metrics)`

**Example:**
```python
buckets, metrics = run_entity_matching(
    spark=spark,
    left_df=df1,
    right_dataframes=[df2, df3, df4]
)
```

### Advanced Functions

#### `MultiPartyRecordLinkage` Class

For advanced users who need more control over the pipeline configuration.

**Methods:**
- `run_pipeline(left_df, right_df, left_origin=1, right_origin=2)`: Basic two-dataset matching
- `run_multi_dataset_pipeline(left_df, right_dataframes_with_origins, left_origin=1)`: One vs many matching
- `run_full_multi_dataset_pipeline(left_dataframes_with_origins, right_dataframes_with_origins)`: Many vs many matching

#### Helper Functions

##### `create_sample_pipeline(spark, similarity_threshold=0.6, min_matching_columns=3)`
Creates a pre-configured pipeline with typical settings.

##### `create_dataset_list_with_origins(dataframes, origin_ids=None)`
Helper to create list of (dataframe, origin_id) tuples.

##### `create_multi_dataset_pipeline(spark, left_dataframes, right_dataframes, ...)`
Convenience function for complex multi-dataset scenarios.

## Usage Examples

### Example 1: Basic Usage (Most Common)

```python
from packages.pyspark.entity_matching_pipeline import run_entity_matching

# This is what most users will use
buckets, metrics = run_entity_matching(
    spark=spark,
    left_df=df1,
    right_dataframes=[df2, df3, df4, df5]
)
```

### Example 2: Custom Thresholds

```python
# Strict matching
buckets, metrics = run_entity_matching(
    spark=spark,
    left_df=df1,
    right_dataframes=[df2, df3],
    similarity_threshold=0.8,
    min_matching_columns=4
)
```

### Example 3: Advanced Configuration

```python
from packages.pyspark.entity_matching_pipeline import MultiPartyRecordLinkage, MatchingConfig

# Custom configuration
config = MatchingConfig(
    similarity_threshold=0.7,
    min_matching_columns=4,
    use_soundex=False,  # Disable soundex
    bucket_assignment_strategy="threshold"
)

pipeline = MultiPartyRecordLinkage(spark, config)
buckets, metrics = pipeline.run_pipeline(df1, df2)
```

## Experiment Notebooks

The `docs/experiments/` directory contains comprehensive examples:

### 1. `00_small_sample_experiment.ipynb`
- **Purpose**: Basic functionality test with synthetic data
- **Data**: Small synthetic datasets (6 records each)
- **Focus**: Verify pipeline works correctly with known ground truth

### 2. `00_df1_df2_sample_seed_42_experiment.ipynb`
- **Purpose**: Reproducible test with 10% sample using fixed seed
- **Data**: 10% sample from df1 and df2 (seed=42)
- **Focus**: Compare results with previous analysis

### 3. `00_df1_df2_experiment.ipynb`
- **Purpose**: Full-scale test with complete df1 and df2 datasets
- **Data**: Complete df1 and df2 datasets
- **Focus**: Performance on large datasets

### 4. `01_df1_df2345_experiment.ipynb`
- **Purpose**: Multi-dataset matching demonstration
- **Data**: df1 (left) vs df2, df3, df4, df5 (right datasets)
- **Focus**: Showcase the main use case

## Refactoring Improvements

The pipeline has been significantly refactored for better maintainability:


## Multi-Dataset Support

The pipeline now supports matching against multiple datasets on the right side (and optionally left side as well). This is useful when you want to match entities against multiple data sources, each with their own origin tracking.

### Key Features

- **Primary Use Case**: One dataset (df1) vs multiple datasets ([df2, df3, df4, ...])
- **Origin Tracking**: Each dataset gets its own origin identifier for traceability
- **Automatic Unioning**: Right datasets are automatically preprocessed and unioned together
- **Full Multi-Dataset**: Support for multiple datasets on both left and right sides (advanced usage)

### Usage Examples

#### Example 1: Recommended Usage Pattern

```python
# This is the main use case the pipeline was designed for
buckets, metrics = run_entity_matching(
    spark=spark,
    left_df=df1,                    # One left dataset
    right_dataframes=[df2, df3, df4]  # Multiple right datasets
)
```

#### Example 2: Advanced Multi-Dataset with Origins

```python
from packages.pyspark.entity_matching_pipeline import MultiPartyRecordLinkage, MatchingConfig

# Create pipeline
config = MatchingConfig(similarity_threshold=0.6, min_matching_columns=3)
pipeline = MultiPartyRecordLinkage(spark, config)

# Define right datasets with explicit origins
right_datasets_with_origins = [
    (customer_df, 101),    # Customer database
    (employee_df, 102),    # Employee database  
    (vendor_df, 103),      # Vendor database
]

# Run multi-dataset pipeline
buckets, metrics = pipeline.run_multi_dataset_pipeline(
    left_df=target_df,
    right_dataframes_with_origins=right_datasets_with_origins,
    left_origin=1
)
```

#### Example 3: Full Multi-Dataset (Advanced)

```python
# Multiple datasets on both sides (for advanced users)
left_datasets_with_origins = [(dataset_a, 1), (dataset_b, 2)]
right_datasets_with_origins = [(dataset_c, 101), (dataset_d, 102), (dataset_e, 103)]

buckets, metrics = pipeline.run_full_multi_dataset_pipeline(
    left_datasets_with_origins,
    right_datasets_with_origins
)
```

### Benefits of the Refactored Pipeline

1. **Simplified API**: Most users only need `run_entity_matching()`
2. **Data Source Tracking**: Each dataset maintains its identity through origin tracking
3. **Scalability**: Handle datasets from multiple sources in a single pipeline run
4. **Flexibility**: Mix and match different numbers of datasets on each side
5. **Efficiency**: Single cross-join operation instead of multiple separate runs
6. **Comprehensive Results**: Unified bucket assignments across all data sources
7. **Clean Output**: Professional logging without emoji clutter
8. **Better Maintainability**: Cleaner, more readable codebase

### Use Cases

- **Customer Deduplication**: Match customer records against multiple databases
- **Entity Resolution**: Resolve entities across different data silos
- **Data Integration**: Merge records from multiple acquisition sources
- **Compliance**: Match entities against multiple watchlists or reference data

## Performance Considerations

1. **Data Sampling**: Use `.sample()` for large datasets during development
2. **Caching**: Pipeline automatically caches intermediate results
3. **Partitioning**: Consider repartitioning based on entity keys for better performance
4. **Memory**: Monitor memory usage during cross-join operations
5. **Spark Configuration**: Use adaptive query execution for better performance:
   ```python
   spark = SparkSession.builder \
       .config("spark.sql.adaptive.enabled", "true") \
       .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
       .getOrCreate()
   ```

## Evaluation Metrics

The pipeline provides comprehensive evaluation metrics:

- **Ground Truth**: Number of actual matching pairs
- **True Positives**: Correctly identified matches
- **False Positives**: Incorrectly identified matches  
- **False Negatives**: Missed matches
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **Total Buckets**: Number of generated buckets
- **Average Bucket Size**: Mean number of entities per bucket

## Output Format

The pipeline returns:
1. **Buckets DataFrame**: Contains bucket assignments with columns:
   - `bucket_id`: Identifier of the bucket
   - `assigned_entities`: List of entity IDs in the bucket
   - `avg_similarity`: Average similarity score in the bucket
   - `bucket_size`: Number of entities in the bucket

2. **Metrics Dictionary**: Evaluation results as described above

## Integration with Existing Code

The pipeline is designed to integrate seamlessly with existing PySpark workflows:

```python
# Read your data
df1 = spark.read.csv("data/df1.csv", header=True)
df2 = spark.read.csv("data/df2.csv", header=True)

# Apply existing transformations
df1 = apply_custom_preprocessing(df1)
df2 = apply_custom_preprocessing(df2)

# Run entity matching
pipeline = MultiPartyRecordLinkage(spark, config)
buckets, metrics = pipeline.run_pipeline(df1, df2)

# Continue with downstream processing
final_results = buckets.join(other_data, on="bucket_id")
```

## Extensibility

The pipeline is designed for extensibility:

1. **Custom Similarity Functions**: Override `calculate_similarity_matrix()`
2. **Custom Assignment Strategies**: Add new strategies to `assign_to_buckets()`
3. **Custom Evaluation**: Extend `evaluate_buckets()` with domain-specific metrics
4. **Custom Preprocessing**: Override `preprocess_dataframe()` for domain-specific transforms

## Best Practices

### Getting Started
1. **Start Simple**: Use `run_entity_matching()` for most use cases
2. **Test with Samples**: Use small data samples first to validate configuration
3. **Iterative Tuning**: Adjust thresholds based on evaluation metrics

### Configuration Guidelines
```python
# Conservative (high precision, lower recall)
buckets, metrics = run_entity_matching(
    spark=spark, left_df=df1, right_dataframes=[df2, df3],
    similarity_threshold=0.8, min_matching_columns=4
)

# Balanced (good precision/recall trade-off) 
buckets, metrics = run_entity_matching(
    spark=spark, left_df=df1, right_dataframes=[df2, df3],
    similarity_threshold=0.6, min_matching_columns=3
)

# Liberal (higher recall, lower precision)
buckets, metrics = run_entity_matching(
    spark=spark, left_df=df1, right_dataframes=[df2, df3],
    similarity_threshold=0.4, min_matching_columns=2
)
```

### Performance Optimization
1. **Monitor Performance**: Use Spark UI to monitor job execution
2. **Validate Results**: Always compare against known ground truth when available
3. **Document Configurations**: Keep track of configuration parameters for reproducibility

## Integration with Existing Code

The refactored pipeline integrates seamlessly with existing PySpark workflows:

```python
# Read your data
df1 = spark.read.csv("data/df1.csv", header=False).toDF("0", "1", "2", "3", "4", "5")
df2 = spark.read.csv("data/df2.csv", header=False).toDF("0", "1", "2", "3", "4", "5")
df3 = spark.read.csv("data/df3.csv", header=False).toDF("0", "1", "2", "3", "4", "5")

# Apply existing transformations if needed
df1 = apply_custom_preprocessing(df1)
df2 = apply_custom_preprocessing(df2)
df3 = apply_custom_preprocessing(df3)

# Run entity matching - one simple line
buckets, metrics = run_entity_matching(
    spark=spark,
    left_df=df1,
    right_dataframes=[df2, df3]
)

# Continue with downstream processing
final_results = buckets.join(other_data, on="bucket_id")
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce similarity threshold or use data sampling
2. **Poor Performance**: Check for data skew, consider repartitioning
3. **Low Precision**: Increase similarity threshold or min matching columns
4. **Low Recall**: Decrease similarity threshold or min matching columns

### Debug Mode

Enable verbose logging to understand pipeline execution:

```python
import logging
logging.getLogger("py4j").setLevel(logging.WARNING)
spark.sparkContext.setLogLevel("INFO")
```

## File Structure

```
packages/pyspark/
├── entity_matching_pipeline.py     # Refactored pipeline (470 lines, clean)
└── __init__.py

docs/experiments/                    # New experiment notebooks
├── 00_small_sample_experiment.ipynb           # Basic functionality test
├── 00_df1_df2_sample_seed_42_experiment.ipynb # Reproducible sample test  
├── 00_df1_df2_experiment.ipynb                # Full dataset test
└── 01_df1_df2345_experiment.ipynb             # Multi-dataset demonstration

examples/
├── simple_usage_example.py         # Basic usage example
└── entity_matching_pipeline_example.ipynb
```

## Summary of Refactoring

The entity matching pipeline has been significantly improved while maintaining all core functionality:

### What Changed
- ✅ **Reduced complexity**: 601 lines → 470 lines (22% reduction)
- ✅ **Removed emoji icons**: Clean, professional step messages
- ✅ **Simplified documentation**: Concise, focused comments
- ✅ **Added simple API**: `run_entity_matching()` function for common use cases
- ✅ **Better organization**: Cleaner method structure and grouping

### What Stayed the Same
- ✅ **All algorithms**: Matching logic, similarity calculation, bucket assignment
- ✅ **All configuration options**: MatchingConfig parameters unchanged
- ✅ **All evaluation metrics**: Precision, recall, F1-score calculations
- ✅ **Multi-dataset support**: Complex scenarios still fully supported
- ✅ **Performance characteristics**: Same scalability and efficiency

### Main Benefits
1. **Easier to use**: Simple one-line API for common cases
2. **Easier to read**: Clean code without clutter
3. **Easier to maintain**: Better organized, well-documented
4. **Easier to extend**: Clear structure for future enhancements

This refactored pipeline provides a production-ready, maintainable solution for entity matching that scales from simple two-dataset comparisons to complex multi-dataset scenarios.
