# Entity Matching Pipeline

> A PySpark-based pipeline for record linkage and entity resolution across multiple datasets

## Project Overview

This thesis project implements a scalable entity matching pipeline designed to identify and link records representing the same real-world entities across multiple datasets. The system addresses the common problem of entity resolution where identical entities appear with variations due to spelling differences, data entry errors, or formatting inconsistencies.

**Research Problem**: How can we automatically identify and group records that represent the same entities across multiple datasets when exact matches are not possible due to data quality issues?

**Solution**: A configurable PySpark pipeline that uses phonetic matching and similarity scoring to create entity clusters with comprehensive evaluation metrics.

## Technical Implementation

The pipeline processes multiple CSV datasets and performs entity matching using the following approach:

```python
from packages.pyspark.entity_matching_pipeline import run_entity_matching
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("EntityMatching").getOrCreate()

# Load datasets
df1 = spark.read.csv("data/df1.csv", header=False).toDF("0", "1", "2", "3", "4", "5")
df2 = spark.read.csv("data/df2.csv", header=False).toDF("0", "1", "2", "3", "4", "5") 
df3 = spark.read.csv("data/df3.csv", header=False).toDF("0", "1", "2", "3", "4", "5")

# Execute entity matching
buckets, metrics = run_entity_matching(
    spark=spark,
    left_df=df1,
    right_dataframes=[df2, df3],
    similarity_threshold=0.6,
    min_matching_columns=3
)

# Analyze results
print(f"Total entity groups: {buckets.count()}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

## Problem Domain and Motivation

Entity resolution is a fundamental challenge in data integration and master data management. Common scenarios include:

- **Customer Data Integration**: Merging customer records from different business systems
- **Academic Research**: Linking publications and authors across bibliographic databases  
- **Healthcare**: Patient record matching across hospital systems
- **E-commerce**: Product catalog deduplication and matching
- **Compliance**: Entity matching against watchlists and regulatory databases

### Challenges Addressed

1. **Spelling Variations**: "Smith" vs "Smyth", "McDonald" vs "MacDonald"
2. **Data Entry Errors**: Typos and transcription mistakes
3. **Formatting Differences**: "123 Main St" vs "123 Main Street"
4. **Incomplete Information**: Missing or partial field values
5. **Scalability**: Processing large datasets with millions of records

## System Architecture

### 1. Data Preprocessing
- **Origin Tracking**: Each dataset receives a unique identifier to maintain data lineage
- **Phonetic Normalization**: Soundex transformation converts text to phonetic codes (e.g., "Smith" and "Smyth" both become "S530")
- **Schema Standardization**: Ensures consistent column naming and data types

### 2. Entity Key Generation
- **Composite Key Creation**: Concatenates matching columns to create entity signatures
- **Cross-Dataset Preparation**: Structures data for efficient comparison operations

### 3. Similarity Calculation
- **Cross-Join Processing**: Compares all records from left dataset against right datasets
- **Column-wise Matching**: Calculates similarity based on matching attribute columns
- **Scoring Algorithm**: Generates similarity scores as ratio of matching columns to total columns

### 4. Threshold Filtering and Bucket Assignment
- **Configurable Thresholds**: Filters record pairs based on similarity scores and minimum matching columns
- **Max Similarity Strategy**: Assigns entities to buckets with highest similarity scores
- **Alternative Threshold Strategy**: Accepts all pairs above specified similarity threshold

### 5. Evaluation and Metrics
- **Ground Truth Calculation**: Determines actual matching pairs for evaluation
- **Performance Metrics**: Calculates precision, recall, F1-score, and bucket statistics
- **Comprehensive Reporting**: Provides detailed analysis of matching quality

## Implementation Details

### Core Components

**MultiPartyRecordLinkage Class**: Main pipeline implementation with configurable matching parameters

**MatchingConfig**: Configuration dataclass supporting:
- Similarity thresholds (0.0 - 1.0)
- Minimum matching column requirements
- Phonetic transformation options
- Bucket assignment strategies

**Helper Functions**: Simplified API for common use cases and multi-dataset scenarios

### Algorithm Characteristics

- **Time Complexity**: O(n×m) where n is left dataset size and m is combined right dataset size
- **Space Complexity**: Optimized through Spark's distributed processing and caching strategies
- **Scalability**: Tested on datasets ranging from hundreds to millions of records

## Project Structure and Organization

```
Thesis/
├── packages/
│   ├── pyspark/                    # Core pipeline implementation
│   │   └── entity_matching_pipeline.py
│   └── utils/                      # Supporting utilities
│       ├── transformations.py
│       └── spark_udfs.py
├── docs/
│   ├── experiments/                # Experimental notebooks
│   │   ├── 00_small_sample_experiment.ipynb
│   │   ├── 00_df1_df2_sample_seed_42_experiment.ipynb
│   │   ├── 00_df1_df2_experiment.ipynb
│   │   └── 01_df1_df2345_experiment.ipynb
│   ├── ENTITY_MATCHING_PIPELINE.md # Complete documentation
│   └── pipelines/                  # Architecture diagrams
├── data/                           # Sample datasets
│   ├── df1.csv through df5.csv
│   └── output/                     # Results storage
├── examples/                       # Usage examples
└── requirements.txt               # Dependencies
```


# Entity Matching Pipeline

> A PySpark-based pipeline for record linkage and entity resolution across multiple datasets

## Project Overview

This thesis project implements a scalable entity matching pipeline designed to identify and link records representing the same real-world entities across multiple datasets. The system addresses the common problem of entity resolution where identical entities appear with variations due to spelling differences, data entry errors, or formatting inconsistencies.

**Research Problem**: How can we automatically identify and group records that represent the same entities across multiple datasets when exact matches are not possible due to data quality issues?

**Solution**: A configurable PySpark pipeline that uses phonetic matching and similarity scoring to create entity clusters with comprehensive evaluation metrics.

## Technical Implementation

The pipeline processes multiple CSV datasets and performs entity matching using the following approach:

```python
from packages.pyspark.entity_matching_pipeline import run_entity_matching
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.appName("EntityMatching").getOrCreate()

# Load datasets
df1 = spark.read.csv("data/df1.csv", header=False).toDF("0", "1", "2", "3", "4", "5")
df2 = spark.read.csv("data/df2.csv", header=False).toDF("0", "1", "2", "3", "4", "5") 
df3 = spark.read.csv("data/df3.csv", header=False).toDF("0", "1", "2", "3", "4", "5")

# Execute entity matching
buckets, metrics = run_entity_matching(
    spark=spark,
    left_df=df1,
    right_dataframes=[df2, df3],
    similarity_threshold=0.6,
    min_matching_columns=3
)

# Analyze results
print(f"Total entity groups: {buckets.count()}")
print(f"F1-Score: {metrics['f1_score']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
```

## Problem Domain and Motivation

Entity resolution is a fundamental challenge in data integration and master data management. Common scenarios include:

- **Customer Data Integration**: Merging customer records from different business systems
- **Academic Research**: Linking publications and authors across bibliographic databases  
- **Healthcare**: Patient record matching across hospital systems
- **E-commerce**: Product catalog deduplication and matching
- **Compliance**: Entity matching against watchlists and regulatory databases

### Challenges Addressed

1. **Spelling Variations**: "Smith" vs "Smyth", "McDonald" vs "MacDonald"
2. **Data Entry Errors**: Typos and transcription mistakes
3. **Formatting Differences**: "123 Main St" vs "123 Main Street"
4. **Incomplete Information**: Missing or partial field values
5. **Scalability**: Processing large datasets with millions of records

## System Architecture

### 1. Data Preprocessing
- **Origin Tracking**: Each dataset receives a unique identifier to maintain data lineage
- **Phonetic Normalization**: Soundex transformation converts text to phonetic codes (e.g., "Smith" and "Smyth" both become "S530")
- **Schema Standardization**: Ensures consistent column naming and data types

### 2. Entity Key Generation
- **Composite Key Creation**: Concatenates matching columns to create entity signatures
- **Cross-Dataset Preparation**: Structures data for efficient comparison operations

### 3. Similarity Calculation
- **Cross-Join Processing**: Compares all records from left dataset against right datasets
- **Column-wise Matching**: Calculates similarity based on matching attribute columns
- **Scoring Algorithm**: Generates similarity scores as ratio of matching columns to total columns

### 4. Threshold Filtering and Bucket Assignment
- **Configurable Thresholds**: Filters record pairs based on similarity scores and minimum matching columns
- **Max Similarity Strategy**: Assigns entities to buckets with highest similarity scores
- **Alternative Threshold Strategy**: Accepts all pairs above specified similarity threshold

### 5. Evaluation and Metrics
- **Ground Truth Calculation**: Determines actual matching pairs for evaluation
- **Performance Metrics**: Calculates precision, recall, F1-score, and bucket statistics
- **Comprehensive Reporting**: Provides detailed analysis of matching quality

## Implementation Details

### Core Components

**MultiPartyRecordLinkage Class**: Main pipeline implementation with configurable matching parameters

**MatchingConfig**: Configuration dataclass supporting:
- Similarity thresholds (0.0 - 1.0)
- Minimum matching column requirements
- Phonetic transformation options
- Bucket assignment strategies

**Helper Functions**: Simplified API for common use cases and multi-dataset scenarios

### Algorithm Characteristics

- **Time Complexity**: O(n×m) where n is left dataset size and m is combined right dataset size
- **Space Complexity**: Optimized through Spark's distributed processing and caching strategies
- **Scalability**: Tested on datasets ranging from hundreds to millions of records

## Project Structure and Organization

```
Thesis/
├── packages/
│   ├── pyspark/                    # Core pipeline implementation
│   │   └── entity_matching_pipeline.py
│   └── utils/                      # Supporting utilities
│       ├── transformations.py
│       └── spark_udfs.py
├── docs/
│   ├── experiments/                # Experimental notebooks
│   │   ├── 00_small_sample_experiment.ipynb
│   │   ├── 00_df1_df2_sample_seed_42_experiment.ipynb
│   │   ├── 00_df1_df2_experiment.ipynb
│   │   └── 01_df1_df2345_experiment.ipynb
│   ├── ENTITY_MATCHING_PIPELINE.md # Complete documentation
│   └── pipelines/                  # Architecture diagrams
├── data/                           # Sample datasets
│   ├── df1.csv through df5.csv
│   └── output/                     # Results storage
├── examples/                       # Usage examples
└── requirements.txt               # Dependencies
```



-----------------------------------------------

7. **Vortex PySpark Pipeline Proposal**
    ![Data Architecture](docs/pipelines/hashed_vortex_proposal_architecture.png)
    There is another variation of the multiparty record linkage (MRL) solution where we build a vortex that will store the total data in an indexed structure. This should 
    reduce complexity of the system, caching the MRL hashed data and be used later when requested by any party.



## Getting Started

### Prerequisites

- **Python 3.7+**: Required for modern PySpark compatibility
- **Java 8 or 11**: Required by Apache Spark
- **Apache Spark 3.x**: Automatically installed via requirements.txt
- **Minimum 8GB RAM**: Recommended for processing medium-sized datasets

### Installation Steps

1. **Clone the repository**:
### Prerequisites

- **Python 3.7+**: Required for modern PySpark compatibility
- **Java 8 or 11**: Required by Apache Spark
- **Apache Spark 3.x**: Automatically installed via requirements.txt
- **Minimum 8GB RAM**: Recommended for processing medium-sized datasets

### Installation Steps

1. **Clone the repository**:
```bash
git clone https://github.com/kostasrazgkelis/Thesis.git
cd Thesis
```

2. **Set up Python environment** (recommended):
```bash
# Create virtual environment
python -m venv thesis_env

# Activate environment
# On Windows:
thesis_env\Scripts\activate
# On macOS/Linux:
source thesis_env/bin/activate
```

3. **Install dependencies**:
2. **Set up Python environment** (recommended):
```bash
# Create virtual environment
python -m venv thesis_env

# Activate environment
# On Windows:
thesis_env\Scripts\activate
# On macOS/Linux:
source thesis_env/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "from pyspark.sql import SparkSession; print('Spark installed successfully')"
```

### Running the Pipeline

1. **Basic execution**:
```bash
# Navigate to project directory
cd Thesis

# Run simple example
python examples/simple_usage_example.py
```

2. **Interactive exploration**:
```bash
# Start Jupyter notebook
jupyter notebook

# Open any experiment notebook in docs/experiments/
# Recommended starting point: 00_small_sample_experiment.ipynb
```

3. **Custom dataset processing**:
```python
# Place your CSV files in the data/ directory
# Ensure columns are named: "0", "1", "2", "3", "4", "5"
# Column "0" should contain unique identifiers
# Columns "1"-"5" contain matching attributes

from packages.pyspark.entity_matching_pipeline import run_entity_matching
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("YourExperiment").getOrCreate()

# Load your datasets
df1 = spark.read.csv("data/your_file1.csv", header=False).toDF("0", "1", "2", "3", "4", "5")
df2 = spark.read.csv("data/your_file2.csv", header=False).toDF("0", "1", "2", "3", "4", "5")

# Run matching
buckets, metrics = run_entity_matching(spark, df1, [df2])
```

## Experimental Framework

### Designed Experiments

1. **Small Sample Experiment** (`00_small_sample_experiment.ipynb`)
   - **Purpose**: Validate core functionality with synthetic data
   - **Dataset Size**: 6 records per dataset
   - **Expected Results**: Perfect precision and recall
   - **Usage**: Initial testing and algorithm verification

2. **Reproducible Sample Test** (`00_df1_df2_sample_seed_42_experiment.ipynb`)
   - **Purpose**: Consistent results with fixed random seed
   - **Dataset Size**: 10% sample from full datasets
   - **Focus**: Reproducibility and baseline establishment
   - **Seed**: 42 (for consistent sampling across runs)

3. **Full Dataset Evaluation** (`00_df1_df2_experiment.ipynb`)
   - **Purpose**: Production-scale performance assessment
   - **Dataset Size**: Complete df1 and df2 datasets
   - **Focus**: Scalability and real-world performance
   - **Metrics**: Comprehensive evaluation including execution time

4. **Multi-Dataset Integration** (`01_df1_df2345_experiment.ipynb`)
   - **Purpose**: Demonstrate multi-dataset matching capabilities
   - **Dataset Configuration**: df1 vs [df2, df3, df4, df5]
   - **Focus**: Complex scenario handling and origin tracking
   - **Analysis**: Cross-dataset matching patterns and statistics

### Configuration Examples

**Conservative Matching** (High Precision):
```python
buckets, metrics = run_entity_matching(
    spark=spark, left_df=df1, right_dataframes=[df2],
    similarity_threshold=0.8,  # 80% similarity required
    min_matching_columns=4     # 4 out of 5 columns must match
)
```

**Balanced Matching** (Precision-Recall Balance):
```python
buckets, metrics = run_entity_matching(
    spark=spark, left_df=df1, right_dataframes=[df2],
    similarity_threshold=0.6,  # 60% similarity required
    min_matching_columns=3     # 3 out of 5 columns must match
)
```

**Liberal Matching** (High Recall):
```python
buckets, metrics = run_entity_matching(
    spark=spark, left_df=df1, right_dataframes=[df2],
    similarity_threshold=0.4,  # 40% similarity required
    min_matching_columns=2     # 2 out of 5 columns must match
)
```

## Results Interpretation

### Output Analysis

**Buckets DataFrame Structure**:
- `bucket_id`: Unique identifier for each entity cluster
- `assigned_entities`: List of record IDs grouped in this cluster
- `bucket_size`: Number of records in the cluster
- `avg_similarity`: Average similarity score within the cluster

**Evaluation Metrics**:
- **Precision**: TP / (TP + FP) - Proportion of identified matches that are correct
- **Recall**: TP / (TP + FN) - Proportion of actual matches that were identified  
- **F1-Score**: Harmonic mean of precision and recall - Overall matching quality
- **Ground Truth**: Number of actual matching pairs in the dataset

### Performance Benchmarking

Expected performance characteristics based on experimental analysis:
- **Small datasets** (< 1K records): Near real-time processing
- **Medium datasets** (1K - 100K records): Processing time in minutes
- **Large datasets** (> 100K records): May require cluster computing resources

## Research Contributions

1. **Scalable Architecture**: Distributed processing capability for large-scale entity resolution
2. **Configurable Algorithms**: Flexible similarity thresholds and matching strategies
3. **Comprehensive Evaluation**: Automated quality assessment with standard metrics
4. **Multi-Dataset Support**: Simultaneous matching across multiple data sources
5. **Clean Implementation**: Well-documented, maintainable codebase suitable for academic review

## Code Quality and Documentation

- **Modular Design**: Clear separation of concerns with dedicated classes and functions
- **Type Hints**: Full type annotations for better code clarity and IDE support
- **Comprehensive Testing**: Multiple experiment notebooks validating different scenarios
- **Academic Standards**: Professional documentation suitable for thesis submission
- **Reproducible Results**: Fixed random seeds and deterministic algorithms where applicable

## Future Extensions

Potential areas for further research and development:
- **Advanced Similarity Metrics**: Integration of machine learning-based similarity functions
- **Real-time Processing**: Stream processing capabilities for continuous entity matching
- **Interactive Visualization**: Graphical tools for exploring entity clusters and relationships
- **Domain-Specific Adaptations**: Specialized matching rules for different data types
- **Performance Optimization**: Advanced caching and indexing strategies

## References and Further Reading

- **[Complete Technical Documentation](docs/ENTITY_MATCHING_PIPELINE.md)**: Detailed API reference and advanced usage
- **[System Architecture](docs/pipelines/data_architecture.png)**: Visual overview of system components
- **Experimental Notebooks**: Comprehensive examples demonstrating various use cases and configurations

## Academic Context

This implementation addresses fundamental challenges in data integration and master data management, particularly relevant for:
- Database systems research
- Information retrieval and extraction
- Data quality and cleaning methodologies
- Large-scale data processing systems

The pipeline serves as both a practical tool for entity resolution tasks and a foundation for further research in automated data integration techniques.
4. **Verify installation**:
```bash
python -c "from pyspark.sql import SparkSession; print('Spark installed successfully')"
```

### Running the Pipeline

1. **Basic execution**:
```bash
# Navigate to project directory
cd Thesis

# Run simple example
python examples/simple_usage_example.py
```

2. **Interactive exploration**:
```bash
# Start Jupyter notebook
jupyter notebook

# Open any experiment notebook in docs/experiments/
# Recommended starting point: 00_small_sample_experiment.ipynb
```

3. **Custom dataset processing**:
```python
# Place your CSV files in the data/ directory
# Ensure columns are named: "0", "1", "2", "3", "4", "5"
# Column "0" should contain unique identifiers
# Columns "1"-"5" contain matching attributes

from packages.pyspark.entity_matching_pipeline import run_entity_matching
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("YourExperiment").getOrCreate()

# Load your datasets
df1 = spark.read.csv("data/your_file1.csv", header=False).toDF("0", "1", "2", "3", "4", "5")
df2 = spark.read.csv("data/your_file2.csv", header=False).toDF("0", "1", "2", "3", "4", "5")

# Run matching
buckets, metrics = run_entity_matching(spark, df1, [df2])
```

## Experimental Framework

### Designed Experiments

1. **Small Sample Experiment** (`00_small_sample_experiment.ipynb`)
   - **Purpose**: Validate core functionality with synthetic data
   - **Dataset Size**: 6 records per dataset
   - **Expected Results**: Perfect precision and recall
   - **Usage**: Initial testing and algorithm verification

2. **Reproducible Sample Test** (`00_df1_df2_sample_seed_42_experiment.ipynb`)
   - **Purpose**: Consistent results with fixed random seed
   - **Dataset Size**: 10% sample from full datasets
   - **Focus**: Reproducibility and baseline establishment
   - **Seed**: 42 (for consistent sampling across runs)

3. **Full Dataset Evaluation** (`00_df1_df2_experiment.ipynb`)
   - **Purpose**: Production-scale performance assessment
   - **Dataset Size**: Complete df1 and df2 datasets
   - **Focus**: Scalability and real-world performance
   - **Metrics**: Comprehensive evaluation including execution time

4. **Multi-Dataset Integration** (`01_df1_df2345_experiment.ipynb`)
   - **Purpose**: Demonstrate multi-dataset matching capabilities
   - **Dataset Configuration**: df1 vs [df2, df3, df4, df5]
   - **Focus**: Complex scenario handling and origin tracking
   - **Analysis**: Cross-dataset matching patterns and statistics

### Configuration Examples

**Conservative Matching** (High Precision):
```python
buckets, metrics = run_entity_matching(
    spark=spark, left_df=df1, right_dataframes=[df2],
    similarity_threshold=0.8,  # 80% similarity required
    min_matching_columns=4     # 4 out of 5 columns must match
)
```

**Balanced Matching** (Precision-Recall Balance):
```python
buckets, metrics = run_entity_matching(
    spark=spark, left_df=df1, right_dataframes=[df2],
    similarity_threshold=0.6,  # 60% similarity required
    min_matching_columns=3     # 3 out of 5 columns must match
)
```

**Liberal Matching** (High Recall):
```python
buckets, metrics = run_entity_matching(
    spark=spark, left_df=df1, right_dataframes=[df2],
    similarity_threshold=0.4,  # 40% similarity required
    min_matching_columns=2     # 2 out of 5 columns must match
)
```

## Results Interpretation

### Output Analysis

**Buckets DataFrame Structure**:
- `bucket_id`: Unique identifier for each entity cluster
- `assigned_entities`: List of record IDs grouped in this cluster
- `bucket_size`: Number of records in the cluster
- `avg_similarity`: Average similarity score within the cluster

**Evaluation Metrics**:
- **Precision**: TP / (TP + FP) - Proportion of identified matches that are correct
- **Recall**: TP / (TP + FN) - Proportion of actual matches that were identified  
- **F1-Score**: Harmonic mean of precision and recall - Overall matching quality
- **Ground Truth**: Number of actual matching pairs in the dataset

### Performance Benchmarking

Expected performance characteristics based on experimental analysis:
- **Small datasets** (< 1K records): Near real-time processing
- **Medium datasets** (1K - 100K records): Processing time in minutes
- **Large datasets** (> 100K records): May require cluster computing resources

## Research Contributions

1. **Scalable Architecture**: Distributed processing capability for large-scale entity resolution
2. **Configurable Algorithms**: Flexible similarity thresholds and matching strategies
3. **Comprehensive Evaluation**: Automated quality assessment with standard metrics
4. **Multi-Dataset Support**: Simultaneous matching across multiple data sources
5. **Clean Implementation**: Well-documented, maintainable codebase suitable for academic review

## Code Quality and Documentation

- **Modular Design**: Clear separation of concerns with dedicated classes and functions
- **Type Hints**: Full type annotations for better code clarity and IDE support
- **Comprehensive Testing**: Multiple experiment notebooks validating different scenarios
- **Academic Standards**: Professional documentation suitable for thesis submission
- **Reproducible Results**: Fixed random seeds and deterministic algorithms where applicable

## Future Extensions

Potential areas for further research and development:
- **Advanced Similarity Metrics**: Integration of machine learning-based similarity functions
- **Real-time Processing**: Stream processing capabilities for continuous entity matching
- **Interactive Visualization**: Graphical tools for exploring entity clusters and relationships
- **Domain-Specific Adaptations**: Specialized matching rules for different data types
- **Performance Optimization**: Advanced caching and indexing strategies

## References and Further Reading

- **[Complete Technical Documentation](docs/ENTITY_MATCHING_PIPELINE.md)**: Detailed API reference and advanced usage
- **[System Architecture](docs/pipelines/data_architecture.png)**: Visual overview of system components
- **Experimental Notebooks**: Comprehensive examples demonstrating various use cases and configurations

## Academic Context

This implementation addresses fundamental challenges in data integration and master data management, particularly relevant for:
- Database systems research
- Information retrieval and extraction
- Data quality and cleaning methodologies
- Large-scale data processing systems

The pipeline serves as both a practical tool for entity resolution tasks and a foundation for further research in automated data integration techniques.

## AI Assistance Disclosure

### Development Methodology

This thesis project represents a collaborative approach between human expertise and artificial intelligence assistance. The development process incorporated AI-generated content alongside traditional research and programming methodologies, reflecting modern software development practices and the evolving landscape of academic research.

### AI Contribution Areas

**Code Development**: Portions of the implementation were developed with AI assistance, including:
- Pipeline architecture design and optimization
- Code refactoring and documentation improvements
- Algorithm implementation and testing frameworks
- Performance optimization suggestions

**Documentation and Analysis**: AI tools contributed to:
- Technical documentation writing and structure
- Code commenting and inline documentation
- Experimental notebook creation and organization
- README and markdown file development

**Quality Assurance**: All AI-generated content has been:
- Thoroughly reviewed and validated by the author
- Tested for correctness and performance
- Integrated thoughtfully with human-written components
- Verified against academic and technical standards

### Human Oversight and Validation

The author maintains full responsibility for:
- Research design and methodology selection
- Algorithm logic and implementation decisions
- Experimental design and result interpretation
- Final code review and quality assurance
- Academic integrity and thesis contributions

### Hybrid Development Model

This project demonstrates a hybrid development approach where:
- **Human expertise** provides domain knowledge, research direction, and critical evaluation
- **AI assistance** enhances productivity, code quality, and documentation completeness
- **Collaborative review** ensures accuracy, maintainability, and academic rigor

The resulting codebase contains both human-authored and AI-assisted components, all integrated under human supervision and validated through comprehensive testing. This methodology represents an innovative approach to academic software development, leveraging modern AI tools while maintaining scholarly standards and personal accountability.

### Transparency Statement

This disclosure is provided in the interest of academic transparency and reflects the reality of contemporary software development practices. The use of AI assistance does not diminish the originality, contribution, or academic value of this research, but rather demonstrates the effective integration of emerging technologies in academic work.

---

**Author**: Konstantinos Razgkelis
**Institution**: Aristotle University of Thessaloniki
**Degree Program**: Master's Thesis Project  
**Year**: 2025


## AI Assistance Disclosure

### Development Methodology

This thesis project represents a collaborative approach between human expertise and artificial intelligence assistance. The development process incorporated AI-generated content alongside traditional research and programming methodologies, reflecting modern software development practices and the evolving landscape of academic research.

### AI Contribution Areas

**Code Development**: Portions of the implementation were developed with AI assistance, including:
- Pipeline architecture design and optimization
- Code refactoring and documentation improvements
- Algorithm implementation and testing frameworks
- Performance optimization suggestions

**Documentation and Analysis**: AI tools contributed to:
- Technical documentation writing and structure
- Code commenting and inline documentation
- Experimental notebook creation and organization
- README and markdown file development

**Quality Assurance**: All AI-generated content has been:
- Thoroughly reviewed and validated by the author
- Tested for correctness and performance
- Integrated thoughtfully with human-written components
- Verified against academic and technical standards

### Human Oversight and Validation

The author maintains full responsibility for:
- Research design and methodology selection
- Algorithm logic and implementation decisions
- Experimental design and result interpretation
- Final code review and quality assurance
- Academic integrity and thesis contributions

### Hybrid Development Model

This project demonstrates a hybrid development approach where:
- **Human expertise** provides domain knowledge, research direction, and critical evaluation
- **AI assistance** enhances productivity, code quality, and documentation completeness
- **Collaborative review** ensures accuracy, maintainability, and academic rigor

The resulting codebase contains both human-authored and AI-assisted components, all integrated under human supervision and validated through comprehensive testing. This methodology represents an innovative approach to academic software development, leveraging modern AI tools while maintaining scholarly standards and personal accountability.

### Transparency Statement

This disclosure is provided in the interest of academic transparency and reflects the reality of contemporary software development practices. The use of AI assistance does not diminish the originality, contribution, or academic value of this research, but rather demonstrates the effective integration of emerging technologies in academic work.

---

**Author**: Konstantinos Razgkelis
**Institution**: Aristotle University of Thessaloniki
**Degree Program**: Master's Thesis Project  
**Year**: 2025


