# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese stock market K-line pattern analysis tool that implements the "Ranked KLine" algorithm. The project analyzes historical A-share market data to find similar K-line patterns by ranking OHLC prices within time windows and calculating future return statistics. The tool supports parallel processing for performance optimization and pattern grouping for large-scale analysis.

## Architecture

### Core Components

- **main.py**: Entry point that orchestrates the analysis pipeline
- **src/ranked_kline/analyzer.py**: Core `RankedKLineAnalyzer` class implementing the ranking algorithm with parallel processing support
- **src/utils/cli_parser.py**: Command-line argument parsing and validation with parallel and grouping options
- **src/utils/data_loader.py**: CSV data loading for Chinese stock data format
- **src/utils/output_manager.py**: Result formatting and file output management with grouping support
- **src/utils/pattern_grouper.py**: Pattern similarity analysis and grouping for large-scale data compression

### Analysis Modes

1. **Single Stock Analysis** (`--mode single`): Analyzes one stock file with optional parallel processing
2. **Batch Analysis** (`--mode batch`): Processes entire directories of stock files with parallel execution

### Performance Features

1. **Parallel Processing**: Uses `multiprocessing.Pool` to speed up computation
   - Single stock: Data is split into chunks processed in parallel
   - Batch: Different stocks are processed simultaneously
   - Configurable with `--n_jobs` parameter (default: 1, -1 for all CPU cores)

2. **Pattern Grouping**: Reduces output size for large window lengths
   - Groups patterns with high structural similarity
   - Uses normalized distance-based similarity calculation
   - Configurable threshold with `--correlation_threshold` (default: 0.9)
   - Can achieve 90%+ file size compression while maintaining statistical accuracy

### Data Flow

1. Load CSV files with Chinese column names (股票代码, 交易日期, 开盘价, 最高价, 最低价, 收盘价)
2. Extract price sequences and generate ranking patterns within sliding windows (with optional parallel processing)
3. Calculate future returns for multiple time horizons (1, 3, 5, 10, 20 days)
4. Aggregate pattern statistics and filter by minimum frequency threshold
5. Apply pattern grouping if enabled (groups similar patterns to reduce output size)
6. Output results to CSV files with pattern frequencies and return statistics

## Development Commands

### Running Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Basic analysis
python main.py --mode single --stock_file data_example/sh600082.csv --price_type open -w 5
python main.py --mode batch --data_dir data_example/ --price_type close -w 10

# Parallel processing (recommended for large datasets)
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 15 -j 4
python main.py --mode batch --data_dir data_example/ --price_type open -w 10 -j -1

# Pattern grouping (recommended for large windows, w >= 8)
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 8 --enable_grouping
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 10 --enable_grouping --correlation_threshold 0.8

# Combined features for optimal performance
python main.py --mode batch --data_dir data_example/ --price_type close -w 12 -j -1 --enable_grouping --correlation_threshold 0.9

# Performance testing and validation
# Compare single-threaded vs parallel results (should be identical)
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 6 -j 1
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 6 -j 4

# Parameter reference:
# --price_type: open, high, low, close
# --window: time window size (2-100)
# --min_frequency: minimum pattern frequency threshold (default: 10)
# --n_jobs, -j: number of parallel processes (default: 1, -1 for all CPU cores)
# --enable_grouping: enable pattern grouping to reduce output size
# --correlation_threshold: similarity threshold for grouping (default: 0.9)
# --output_dir: output directory (default: output/)
# --verbose: detailed output
```

### Testing and Validation

No test framework is currently configured. Validation should be done by:

1. **Parallel Processing Validation**: Compare single-threaded vs parallel results
   ```bash
   # Results should be identical
   python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 5 -j 1
   python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 5 -j 4
   ```

2. **Pattern Grouping Validation**: Check grouping results make logical sense
   ```bash
   # High threshold should result in fewer groups
   python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 4 --enable_grouping --correlation_threshold 0.95
   # Lower threshold should result in more compression
   python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 4 --enable_grouping --correlation_threshold 0.5
   ```

3. **Performance Testing**: Monitor execution time and memory usage
   ```bash
   time python main.py --mode batch --data_dir data_example/ --price_type close -w 10 -j -1
   ```

## Data Format

### Input CSV Structure (Chinese columns)
- 股票代码: Stock code
- 交易日期: Trading date (YYYY-MM-DD)
- 开盘价: Open price
- 最高价: High price  
- 最低价: Low price
- 收盘价: Close price

### Output Structure
- `output/single_stock/`: Individual stock analysis results
- `output/market/`: Market-wide batch analysis results

#### Standard Output Format
CSV format with columns: `rank_pattern, return_*d_mean, return_*d_std, frequency`

#### Grouped Output Format (when --enable_grouping is used)
CSV format with columns: `group_id, representative_pattern, pattern_count, return_*d_mean, return_*d_std, frequency`
- `group_id`: Unique identifier for each pattern group
- `representative_pattern`: The pattern representing the entire group
- `pattern_count`: Number of patterns merged into this group
- Other columns: Aggregated statistics across all patterns in the group

## Key Implementation Details

### Core Algorithm
- Rankings are calculated using `numpy.argsort()` where 1 = lowest value, n = highest value
- Future returns are calculated as `(future_price - current_price) / current_price`
- Pattern aggregation uses `collections.defaultdict` for efficient frequency counting
- Batch processing merges results across all stocks before applying frequency thresholds

### Parallel Processing
- Uses `multiprocessing.Pool` for CPU-intensive computations
- Single stock analysis: Data is split into chunks processed across multiple processes
- Batch analysis: Different stocks are processed simultaneously
- All parallel results are verified to match single-threaded output exactly
- Automatic CPU core detection with `-j -1` option

### Pattern Grouping (Fixed Implementation)
- **Previous Issue**: Incorrectly grouped dissimilar patterns using flawed correlation calculation
- **Current Implementation**: Uses normalized distance-based similarity calculation
- **Similarity Metric**: `similarity = 1 - (actual_distance / max_possible_distance)`
- **Distance Calculation**: Sum of absolute differences between pattern elements
- **Grouping Algorithm**: BFS clustering with configurable similarity threshold
- **Performance**: Dramatically reduces output file size for large windows (90%+ compression)
- **Accuracy**: Maintains statistical accuracy by properly merging data from truly similar patterns
- **Validation**: Ensures dissimilar patterns (e.g., (1,2,3,4,5) vs (5,4,3,2,1)) are never grouped together

### Dependencies
- `pandas>=1.5.0`: Data manipulation and CSV handling
- `numpy>=1.21.0`: Numerical computations and ranking
- `scipy>=1.9.0`: Statistical functions for pattern similarity
- `matplotlib>=3.5.0`: Optional visualization support

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Memory Issues with Large Windows
**Problem**: Out of memory errors with large window sizes (w > 15)
**Solutions**:
- Enable pattern grouping: `--enable_grouping`
- Use higher correlation threshold: `--correlation_threshold 0.95`
- Reduce batch size by processing fewer stocks at once
- Increase virtual memory/swap space

#### 2. Slow Performance
**Problem**: Analysis takes too long
**Solutions**:
- Enable parallel processing: `-j -1` (use all CPU cores)
- For single stock: `-j 4` (use 4 cores)
- For batch: Parallel processing is automatic across stocks
- Consider pattern grouping for large windows

#### 3. Pattern Grouping Results Validation
**Problem**: Suspicious grouping results (too many or too few groups)
**Solutions**:
- Check correlation threshold: Higher = fewer groups, Lower = more groups
- Validate with small examples first (w=4, w=5)
- Ensure dissimilar patterns aren't grouped together
- Compare grouped vs ungrouped results for accuracy

#### 4. CSV Encoding Issues
**Problem**: Chinese characters not displaying correctly
**Solutions**:
- Ensure input CSV is in UTF-8 encoding
- Try different encodings: GBK, GB2312
- Use text editor to check/convert file encoding

#### 5. Parallel Processing Validation
**Problem**: Parallel results differ from single-threaded
**Solutions**:
- This should never happen - indicates a bug
- Compare outputs carefully with `diff` command
- Report as a critical bug if results differ

### Performance Optimization Tips

1. **Window Size Selection**:
   - Small windows (w=3-7): Fast, many patterns
   - Medium windows (w=8-12): Good balance, use grouping
   - Large windows (w>12): Use grouping, parallel processing

2. **Memory Management**:
   - Use grouping for w >= 8
   - Higher correlation thresholds for better compression
   - Process data in smaller batches if memory limited

3. **Parallel Processing Guidelines**:
   - Single stock: Benefit starts at w >= 10
   - Batch processing: Always beneficial
   - Use `-j -1` unless memory limited

4. **File Size Management**:
   - Standard output can be very large for w > 10
   - Grouping typically reduces size by 90%+
   - Monitor output directory disk usage