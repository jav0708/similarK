"""
输出管理模块
"""
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any
import os


class OutputManager:
    """输出管理器类"""
    
    def __init__(self, output_dir: str = "output"):
        """
        初始化输出管理器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = Path(output_dir)
        self._create_output_dirs()
    
    def _create_output_dirs(self):
        """创建输出目录结构"""
        dirs = [
            self.output_dir,
            self.output_dir / "single_stock",
            self.output_dir / "market", 
            self.output_dir / "charts"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_single_stock_result(self, stock_code: str, price_type: str, 
                                window: int, results: Dict[str, Any], min_frequency: int = 10) -> str:
        """
        保存单股分析结果
        
        Args:
            stock_code: 股票代码
            price_type: 价格类型
            window: 窗口长度
            results: 分析结果字典
            
        Returns:
            保存的文件路径
        """
        filename = f"{stock_code}_{price_type}_w{window}.csv"
        file_path = self.output_dir / "single_stock" / filename
        
        # 应用频率过滤
        filtered_results = {pattern: stats for pattern, stats in results.items() 
                           if stats['frequency'] >= min_frequency}
        
        # 转换为DataFrame
        df = self._results_to_dataframe(filtered_results)
        
        # 保存CSV
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        return str(file_path)
    
    def save_market_result(self, price_type: str, window: int, 
                          results: Dict[str, Any], min_frequency: int = 10) -> str:
        """
        保存市场分析结果
        
        Args:
            price_type: 价格类型
            window: 窗口长度
            results: 分析结果字典
            
        Returns:
            保存的文件路径
        """
        filename = f"market_{price_type}_w{window}.csv"
        file_path = self.output_dir / "market" / filename
        
        # 应用频率过滤
        filtered_results = {pattern: stats for pattern, stats in results.items() 
                           if stats['frequency'] >= min_frequency}
        
        # 转换为DataFrame
        df = self._results_to_dataframe(filtered_results)
        
        # 保存CSV
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        return str(file_path)
    
    def _results_to_dataframe(self, results: Dict[str, Any]) -> pd.DataFrame:
        """
        将分析结果转换为DataFrame
        
        Args:
            results: 分析结果字典
            
        Returns:
            格式化的DataFrame
        """
        if not results:
            # 返回空的DataFrame但包含正确的列
            return pd.DataFrame(columns=[
                'rank_pattern', 'return_1d_mean', 'return_1d_std',
                'return_3d_mean', 'return_3d_std', 'return_5d_mean', 'return_5d_std',
                'return_10d_mean', 'return_10d_std', 'return_20d_mean', 'return_20d_std',
                'frequency'
            ])
        
        data = []
        
        for pattern, stats in results.items():
            row = {
                'rank_pattern': str(pattern),
                'return_1d_mean': stats.get('returns_1d', {}).get('mean', 0.0),
                'return_1d_std': stats.get('returns_1d', {}).get('std', 0.0),
                'return_3d_mean': stats.get('returns_3d', {}).get('mean', 0.0),
                'return_3d_std': stats.get('returns_3d', {}).get('std', 0.0),
                'return_5d_mean': stats.get('returns_5d', {}).get('mean', 0.0),
                'return_5d_std': stats.get('returns_5d', {}).get('std', 0.0),
                'return_10d_mean': stats.get('returns_10d', {}).get('mean', 0.0),
                'return_10d_std': stats.get('returns_10d', {}).get('std', 0.0),
                'return_20d_mean': stats.get('returns_20d', {}).get('mean', 0.0),
                'return_20d_std': stats.get('returns_20d', {}).get('std', 0.0),
                'frequency': stats.get('frequency', 0)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # 按频率降序排列
        if len(df) > 0:
            df = df.sort_values('frequency', ascending=False).reset_index(drop=True)
        
        return df
    
    def print_summary(self, stock_info: Dict[str, str], price_type: str, 
                     window: int, results: Dict[str, Any], mode: str, min_frequency: int = 10):
        """
        打印分析摘要
        
        Args:
            stock_info: 股票信息
            price_type: 价格类型
            window: 窗口长度
            results: 分析结果
            mode: 分析模式 (single/batch)
        """
        print("\n" + "="*50)
        
        if mode == "single":
            print("=== 单股K线排序分析 ===")
            print(f"股票: {stock_info.get('stock_code', 'N/A')} ({stock_info.get('stock_name', 'N/A')})")
        else:
            print("=== 批量K线排序分析 ===") 
            
        price_type_map = {
            'open': '开盘价',
            'high': '最高价', 
            'low': '最低价',
            'close': '收盘价'
        }
        
        print(f"价格类型: {price_type_map.get(price_type, price_type)}")
        print(f"时间窗口: {window}")
        
        if mode == "single":
            print(f"数据范围: {stock_info.get('start_date')} 至 {stock_info.get('end_date')}")
        
        print()
        print(f"正在计算{price_type_map.get(price_type, price_type)}排序模式...")
        
        # 计算统计信息
        total_patterns = len(results)
        high_freq_patterns = sum(1 for stats in results.values() 
                               if stats['frequency'] >= min_frequency)
        
        # 计算所有可能的模式数量（仅在window<=10时显示，否则数字太大）
        import math
        if window <= 10:
            max_possible = math.factorial(window)
            print(f"找到排序模式: {max_possible:,} 种可能")
        else:
            print(f"排序模式理论数量: {window}! (数量极大)")
        
        print(f"实际出现模式: {total_patterns:,} 种")
        print(f"高频模式 (≥{min_frequency}次): {high_freq_patterns} 种")
        
        if total_patterns > 0:
            max_freq = max(stats['frequency'] for stats in results.values())
            print(f"最高频率: {max_freq} 次")
        print()
        print("生成分析表格...")
    
    def get_output_path(self, mode: str, stock_code: str = None, 
                       price_type: str = None, window: int = None) -> str:
        """
        获取输出文件路径
        
        Args:
            mode: 分析模式
            stock_code: 股票代码
            price_type: 价格类型
            window: 窗口长度
            
        Returns:
            文件路径
        """
        if mode == "single":
            filename = f"{stock_code}_{price_type}_w{window}.csv"
            return str(self.output_dir / "single_stock" / filename)
        else:
            filename = f"market_{price_type}_w{window}.csv"
            return str(self.output_dir / "market" / filename)