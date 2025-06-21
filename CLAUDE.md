# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Chinese stock market K-line pattern analysis tool that implements the "Ranked KLine" algorithm. The project analyzes historical A-share market data to find similar K-line patterns by ranking OHLC prices within time windows and calculating future return statistics.

## Architecture

### Core Components

- **main.py**: Entry point that orchestrates the analysis pipeline
- **src/ranked_kline/analyzer.py**: Core `RankedKLineAnalyzer` class implementing the ranking algorithm
- **src/utils/cli_parser.py**: Command-line argument parsing and validation
- **src/utils/data_loader.py**: CSV data loading for Chinese stock data format
- **src/utils/output_manager.py**: Result formatting and file output management

### Analysis Modes

1. **Single Stock Analysis** (`--mode single`): Analyzes one stock file
2. **Batch Analysis** (`--mode batch`): Processes entire directories of stock files

### Data Flow

1. Load CSV files with Chinese column names (股票代码, 交易日期, 开盘价, 最高价, 最低价, 收盘价)
2. Extract price sequences and generate ranking patterns within sliding windows
3. Calculate future returns for multiple time horizons (1, 3, 5, 10, 20 days)
4. Aggregate pattern statistics and filter by minimum frequency threshold
5. Output results to CSV files with pattern frequencies and return statistics

## Development Commands

### Running Analysis

```bash
# Install dependencies
pip install -r requirements.txt

# Single stock analysis
python main.py --mode single --stock_file data_example/sh600082.csv --price_type open -w 5

# Batch analysis
python main.py --mode batch --data_dir data_example/ --price_type close -w 10

# Common parameters:
# --price_type: open, high, low, close
# --window: time window size (2-100)
# --min_frequency: minimum pattern frequency threshold (default: 10)
# --output_dir: output directory (default: output/)
# --verbose: detailed output
```

### Testing

No test framework is currently configured. Validation should be done by running the analysis on sample data.

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
- CSV format with columns: rank_pattern, return_*d_mean, return_*d_std, frequency

## Key Implementation Details

- Rankings are calculated using `numpy.argsort()` where 1 = lowest value, n = highest value
- Future returns are calculated as `(future_price - current_price) / current_price`
- Pattern aggregation uses `collections.defaultdict` for efficient frequency counting
- Batch processing merges results across all stocks before applying frequency thresholds