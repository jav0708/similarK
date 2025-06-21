#!/usr/bin/env python3
"""
相似K线寻找 - 主程序入口
"""
import sys
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.cli_parser import parse_command_line
from utils.data_loader import DataLoader
from utils.output_manager import OutputManager
from ranked_kline.analyzer import RankedKLineAnalyzer


def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_command_line()
        
        # 初始化组件
        data_loader = DataLoader()
        output_manager = OutputManager(args.output_dir)
        analyzer = RankedKLineAnalyzer(
            window_size=args.window,
            min_frequency=args.min_frequency
        )
        
        start_time = time.time()
        
        if args.mode == 'single':
            # 单股分析
            run_single_analysis(args, data_loader, analyzer, output_manager)
        else:
            # 批量分析
            run_batch_analysis(args, data_loader, analyzer, output_manager)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        print(f"\n分析完成！总耗时: {elapsed:.2f}秒")
        
    except KeyboardInterrupt:
        print("\n\n用户中断程序执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {str(e)}")
        if args.verbose if 'args' in locals() else False:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_single_analysis(args, data_loader, analyzer, output_manager):
    """
    运行单股分析
    
    Args:
        args: 命令行参数
        data_loader: 数据加载器
        analyzer: 分析器
        output_manager: 输出管理器
    """
    print("=== 单股K线排序分析 ===")
    
    # 加载股票数据
    print(f"正在加载股票数据: {args.stock_file}")
    df = data_loader.load_single_stock(args.stock_file)
    
    # 获取股票信息
    stock_info = data_loader.get_stock_info(df)
    
    # 打印分析摘要
    output_manager.print_summary(
        stock_info, args.price_type, args.window, {}, "single"
    )
    
    # 执行分析
    results = analyzer.analyze_single_stock(df, args.price_type)
    
    # 保存结果
    stock_code = stock_info.get('stock_code', Path(args.stock_file).stem)
    output_path = output_manager.save_single_stock_result(
        stock_code, args.price_type, args.window, results
    )
    
    print(f"保存结果: {output_path}")
    print("分析完成！")
    
    # 打印统计信息
    print_analysis_stats(results)


def run_batch_analysis(args, data_loader, analyzer, output_manager):
    """
    运行批量分析
    
    Args:
        args: 命令行参数  
        data_loader: 数据加载器
        analyzer: 分析器
        output_manager: 输出管理器
    """
    print("=== 批量K线排序分析 ===")
    
    # 批量加载股票数据
    print(f"正在扫描数据目录: {args.data_dir}")
    stocks_data = data_loader.load_batch_stocks(args.data_dir)
    
    if not stocks_data:
        raise ValueError("没有成功加载任何股票数据")
    
    # 打印分析摘要
    mock_stock_info = {'stock_code': 'MARKET', 'stock_name': '全市场'}
    output_manager.print_summary(
        mock_stock_info, args.price_type, args.window, {}, "batch"
    )
    
    print(f"开始分析 {len(stocks_data)} 只股票...")
    print()
    
    # 执行批量分析
    results = analyzer.analyze_batch_stocks(stocks_data, args.price_type)
    
    print("\n合并分析结果...")
    
    # 保存结果
    output_path = output_manager.save_market_result(
        args.price_type, args.window, results
    )
    
    print(f"保存结果: {output_path}")
    print("批量分析完成！")
    
    # 打印统计信息
    print_batch_stats(stocks_data, results)


def print_analysis_stats(results):
    """
    打印分析统计信息
    
    Args:
        results: 分析结果
    """
    print(f"\n=== 分析统计 ===")
    print(f"有效模式数量: {len(results)}")
    
    if results:
        # 按频率排序
        sorted_results = sorted(results.items(), 
                              key=lambda x: x[1]['frequency'], 
                              reverse=True)
        
        print(f"最高频率模式: {sorted_results[0][1]['frequency']} 次")
        print(f"模式示例: {sorted_results[0][0]}")
        
        # 统计收益率信息
        avg_returns_1d = [stats['returns_1d']['mean'] 
                         for stats in results.values() 
                         if 'returns_1d' in stats]
        
        if avg_returns_1d:
            import numpy as np
            print(f"1日平均收益率范围: {np.min(avg_returns_1d):.4f} ~ {np.max(avg_returns_1d):.4f}")


def print_batch_stats(stocks_data, results):
    """
    打印批量分析统计信息
    
    Args:
        stocks_data: 股票数据字典
        results: 分析结果
    """
    print(f"\n=== 批量分析统计 ===")
    print(f"处理股票数量: {len(stocks_data)}")
    print(f"合并模式数量: {len(results)}")
    
    if results:
        total_frequency = sum(stats['frequency'] for stats in results.values())
        print(f"模式总出现次数: {total_frequency:,}")
        
        # 最高频模式
        max_freq_pattern = max(results.items(), key=lambda x: x[1]['frequency'])
        print(f"最高频模式: {max_freq_pattern[0]} (出现 {max_freq_pattern[1]['frequency']} 次)")


if __name__ == "__main__":
    main()