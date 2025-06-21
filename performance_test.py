#!/usr/bin/env python3
"""
性能测试脚本 - 对比优化前后的处理速度
"""
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ranked_kline.analyzer import RankedKLineAnalyzer
from utils.data_loader import DataLoader


def create_test_data(num_days=1000, price_start=100):
    """创建测试用的股票数据"""
    dates = pd.date_range('2020-01-01', periods=num_days, freq='D')
    
    # 生成随机价格数据
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, num_days)  # 日收益率
    prices = [price_start]
    
    for i in range(1, num_days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    # 创建OHLC数据
    data = {
        '股票代码': ['TEST'] * num_days,
        '交易日期': dates,
        '开盘价': prices,
        '最高价': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        '最低价': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        '收盘价': prices,
    }
    
    return pd.DataFrame(data)


def benchmark_window_sizes():
    """测试不同窗口大小的性能"""
    print("=== 性能基准测试 ===")
    
    # 创建测试数据
    test_data = create_test_data(5000)  # 5000天数据
    print(f"创建测试数据: {len(test_data)} 天")
    
    window_sizes = [5, 8, 10, 12, 15]
    
    for window_size in window_sizes:
        print(f"\n--- 测试窗口大小: {window_size} ---")
        
        # 单线程测试
        analyzer_single = RankedKLineAnalyzer(
            window_size=window_size,
            n_jobs=1,
            enable_monitoring=False
        )
        
        start_time = time.time()
        results_single = analyzer_single.analyze_single_stock(test_data, 'close')
        single_time = time.time() - start_time
        
        print(f"单线程处理时间: {single_time:.2f}秒")
        print(f"找到模式数量: {len(results_single)}")
        
        # 多线程测试（如果有多核心）
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        
        if cpu_count > 1:
            analyzer_multi = RankedKLineAnalyzer(
                window_size=window_size,
                n_jobs=min(4, cpu_count),
                enable_monitoring=False
            )
            
            start_time = time.time()
            results_multi = analyzer_multi.analyze_single_stock(test_data, 'close')
            multi_time = time.time() - start_time
            
            print(f"多线程处理时间: {multi_time:.2f}秒 (使用{min(4, cpu_count)}核)")
            print(f"加速比: {single_time/multi_time:.2f}x")
            
            # 验证结果一致性
            if len(results_single) == len(results_multi):
                print("✓ 单线程和多线程结果一致")
            else:
                print("✗ 单线程和多线程结果不一致")


def benchmark_data_sizes():
    """测试不同数据量的性能"""
    print("\n=== 数据量性能测试 ===")
    
    data_sizes = [1000, 2000, 5000, 10000]
    window_size = 10
    
    for size in data_sizes:
        print(f"\n--- 测试数据量: {size} 天 ---")
        
        test_data = create_test_data(size)
        
        analyzer = RankedKLineAnalyzer(
            window_size=window_size,
            n_jobs=4,
            enable_monitoring=True
        )
        
        start_time = time.time()
        results = analyzer.analyze_single_stock(test_data, 'close')
        elapsed = time.time() - start_time
        
        windows_processed = size - window_size + 1
        speed = windows_processed / elapsed
        
        print(f"处理时间: {elapsed:.2f}秒")
        print(f"处理速度: {speed:.0f} 窗口/秒")
        print(f"模式数量: {len(results)}")


def test_memory_efficiency():
    """测试内存使用效率"""
    print("\n=== 内存效率测试 ===")
    
    import psutil
    process = psutil.Process()
    
    # 基准内存使用
    baseline_memory = process.memory_info().rss / 1024 / 1024
    print(f"基准内存使用: {baseline_memory:.1f} MB")
    
    # 测试大窗口模式编码效果
    test_data = create_test_data(3000)
    
    for window_size in [8, 12, 16]:
        print(f"\n--- 窗口大小: {window_size} ---")
        
        # 不使用编码
        analyzer_no_enc = RankedKLineAnalyzer(
            window_size=window_size,
            n_jobs=1,
            enable_monitoring=False
        )
        
        memory_before = process.memory_info().rss / 1024 / 1024
        results_no_enc = analyzer_no_enc.analyze_single_stock(test_data, 'close')
        memory_after = process.memory_info().rss / 1024 / 1024
        
        memory_used = memory_after - memory_before
        print(f"内存使用: {memory_used:.1f} MB")
        print(f"模式数量: {len(results_no_enc)}")
        
        # 清理内存
        del results_no_enc, analyzer_no_enc


if __name__ == "__main__":
    try:
        benchmark_window_sizes()
        benchmark_data_sizes() 
        test_memory_efficiency()
        
        print("\n=== 性能测试完成 ===")
        print("建议:")
        print("1. 对于大数据集，使用并行处理可显著提升性能")
        print("2. 窗口大小越大，处理时间越长，建议启用模式分组")
        print("3. 内存使用随窗口大小指数增长，注意监控")
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()