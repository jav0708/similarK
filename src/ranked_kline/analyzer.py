"""
Ranked KLine 排序K线分析器
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import itertools


class RankedKLineAnalyzer:
    """排序K线分析器"""
    
    def __init__(self, window_size: int = 10, min_frequency: int = 10):
        """
        初始化分析器
        
        Args:
            window_size: 时间窗口大小
            min_frequency: 最小频率阈值
        """
        self.window_size = window_size
        self.min_frequency = min_frequency
        self.future_days = [1, 3, 5, 10, 20]  # 预测天数
    
    def analyze_single_stock(self, df: pd.DataFrame, price_type: str) -> Dict[str, Any]:
        """
        分析单只股票
        
        Args:
            df: 股票数据DataFrame
            price_type: 价格类型 ('open', 'high', 'low', 'close')
            
        Returns:
            分析结果字典
        """
        # 价格类型映射
        price_col_map = {
            'open': '开盘价',
            'high': '最高价',
            'low': '最低价', 
            'close': '收盘价'
        }
        
        if price_type not in price_col_map:
            raise ValueError(f"不支持的价格类型: {price_type}")
        
        price_col = price_col_map[price_type]
        if price_col not in df.columns:
            raise ValueError(f"数据中缺少列: {price_col}")
        
        # 提取价格序列
        prices = df[price_col].values
        dates = df['交易日期'].values
        
        # 生成排序模式
        patterns = self._generate_rank_patterns(prices)
        
        # 计算未来收益率
        future_returns = self._calculate_future_returns(prices)
        
        # 统计模式和收益率
        results = self._aggregate_pattern_stats(patterns, future_returns)
        
        # 过滤低频模式
        results = {pattern: stats for pattern, stats in results.items() 
                  if stats['frequency'] >= self.min_frequency}
        
        return results
    
    def analyze_batch_stocks(self, stocks_data: Dict[str, pd.DataFrame], 
                           price_type: str) -> Dict[str, Any]:
        """
        批量分析多只股票
        
        Args:
            stocks_data: 股票代码到DataFrame的字典
            price_type: 价格类型
            
        Returns:
            合并的分析结果
        """
        all_patterns = defaultdict(list)
        all_returns = {day: defaultdict(list) for day in self.future_days}
        
        # 处理每只股票
        for stock_code, df in stocks_data.items():
            try:
                print(f"处理股票: {stock_code}")
                
                # 分析单只股票
                stock_results = self.analyze_single_stock(df, price_type)
                
                # 合并结果
                for pattern, stats in stock_results.items():
                    all_patterns[pattern].append(stats['frequency'])
                    
                    for day in self.future_days:
                        returns_key = f'returns_{day}d'
                        if returns_key in stats:
                            # 将该模式在这只股票中的所有收益率数据加入
                            all_returns[day][pattern].extend(stats[returns_key]['raw_data'])
                
            except Exception as e:
                print(f"处理股票 {stock_code} 时出错: {str(e)}")
                continue
        
        # 聚合所有股票的结果
        merged_results = {}
        
        for pattern in all_patterns:
            # 计算总频率
            total_frequency = sum(all_patterns[pattern])
            
            if total_frequency >= self.min_frequency:
                pattern_stats = {'frequency': total_frequency}
                
                # 计算各期收益率统计
                for day in self.future_days:
                    returns_key = f'returns_{day}d'
                    if pattern in all_returns[day]:
                        returns_data = all_returns[day][pattern]
                        if returns_data:
                            pattern_stats[returns_key] = {
                                'mean': np.mean(returns_data),
                                'std': np.std(returns_data),
                                'count': len(returns_data)
                            }
                        else:
                            pattern_stats[returns_key] = {
                                'mean': 0.0,
                                'std': 0.0,
                                'count': 0
                            }
                
                merged_results[pattern] = pattern_stats
        
        return merged_results
    
    def _generate_rank_patterns(self, prices: np.ndarray) -> List[Tuple]:
        """
        生成排序模式
        
        Args:
            prices: 价格序列
            
        Returns:
            排序模式列表
        """
        patterns = []
        
        for i in range(len(prices) - self.window_size + 1):
            # 提取窗口内的价格
            window_prices = prices[i:i + self.window_size]
            
            # 计算排序（从1到window_size）
            ranks = self._calculate_ranks(window_prices)
            
            patterns.append(tuple(ranks))
        
        return patterns
    
    def _calculate_ranks(self, values: np.ndarray) -> List[int]:
        """
        计算数值的排序
        
        Args:
            values: 数值数组
            
        Returns:
            排序列表（1表示最小值，n表示最大值）
        """
        # 使用argsort获取排序索引，然后计算排名
        sorted_indices = np.argsort(values)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(values) + 1)
        
        return ranks.tolist()
    
    def _calculate_future_returns(self, prices: np.ndarray) -> Dict[int, List[float]]:
        """
        计算未来收益率
        
        Args:
            prices: 价格序列
            
        Returns:
            未来收益率字典
        """
        future_returns = {day: [] for day in self.future_days}
        
        # 为每个窗口计算未来收益率
        for i in range(len(prices) - self.window_size + 1):
            # 窗口结束位置的价格
            current_price = prices[i + self.window_size - 1]
            
            # 计算各期未来收益率
            for day in self.future_days:
                future_idx = i + self.window_size - 1 + day
                if future_idx < len(prices):
                    future_price = prices[future_idx]
                    return_rate = (future_price - current_price) / current_price
                    future_returns[day].append(return_rate)
                else:
                    future_returns[day].append(np.nan)
        
        return future_returns
    
    def _aggregate_pattern_stats(self, patterns: List[Tuple], 
                                future_returns: Dict[int, List[float]]) -> Dict[Tuple, Dict]:
        """
        聚合模式统计信息
        
        Args:
            patterns: 排序模式列表
            future_returns: 未来收益率字典
            
        Returns:
            模式统计字典
        """
        pattern_stats = defaultdict(lambda: {
            'frequency': 0,
            **{f'returns_{day}d': {'data': []} for day in self.future_days}
        })
        
        # 统计每个模式的出现频率和对应的收益率
        for i, pattern in enumerate(patterns):
            pattern_stats[pattern]['frequency'] += 1
            
            # 收集该模式对应的未来收益率
            for day in self.future_days:
                if i < len(future_returns[day]) and not np.isnan(future_returns[day][i]):
                    pattern_stats[pattern][f'returns_{day}d']['data'].append(
                        future_returns[day][i]
                    )
        
        # 计算统计量
        results = {}
        for pattern, stats in pattern_stats.items():
            if stats['frequency'] >= self.min_frequency:
                pattern_result = {'frequency': stats['frequency']}
                
                for day in self.future_days:
                    returns_data = stats[f'returns_{day}d']['data']
                    if returns_data:
                        pattern_result[f'returns_{day}d'] = {
                            'mean': np.mean(returns_data),
                            'std': np.std(returns_data),
                            'count': len(returns_data),
                            'raw_data': returns_data  # 保存原始数据用于批量合并
                        }
                    else:
                        pattern_result[f'returns_{day}d'] = {
                            'mean': 0.0,
                            'std': 0.0,
                            'count': 0,
                            'raw_data': []
                        }
                
                results[pattern] = pattern_result
        
        return results