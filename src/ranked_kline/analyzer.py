"""
Ranked KLine 排序K线分析器
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import itertools
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm


class RankedKLineAnalyzer:
    """排序K线分析器"""
    
    def __init__(self, window_size: int = 10, min_frequency: int = 10, n_jobs: int = 1, 
                 enable_grouping: bool = False, correlation_threshold: float = 0.9):
        """
        初始化分析器
        
        Args:
            window_size: 时间窗口大小
            min_frequency: 最小频率阈值
            n_jobs: 并行处理数量
            enable_grouping: 是否启用分组功能
            correlation_threshold: 相关系数阈值
        """
        self.window_size = window_size
        self.min_frequency = min_frequency
        self.n_jobs = n_jobs
        self.enable_grouping = enable_grouping
        self.correlation_threshold = correlation_threshold
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
        
        if self.n_jobs == 1:
            # 单线程处理
            patterns = self._generate_rank_patterns(prices)
            future_returns = self._calculate_future_returns(prices)
            results = self._aggregate_pattern_stats(patterns, future_returns)
        else:
            # 并行处理
            results = self._analyze_single_stock_parallel(prices)
        
        # 应用分组功能（如果启用）
        if self.enable_grouping and len(results) > 1:
            results = self._apply_grouping(results)
        
        return results
    
    def _analyze_single_stock_parallel(self, prices: np.ndarray) -> Dict[str, Any]:
        """
        并行分析单股数据
        
        Args:
            prices: 价格序列
            
        Returns:
            分析结果字典
        """
        total_windows = len(prices) - self.window_size + 1
        if total_windows <= 0:
            return {}
        
        # 计算每个进程处理的窗口数量
        chunk_size = max(1, total_windows // self.n_jobs)
        chunks = []
        
        for i in range(0, total_windows, chunk_size):
            end_idx = min(i + chunk_size, total_windows)
            # 需要额外的数据来计算未来收益率
            price_end = min(i + chunk_size + self.window_size - 1 + max(self.future_days), len(prices))
            chunk_prices = prices[i:price_end]
            chunks.append((chunk_prices, i, end_idx - i))
        
        # 并行处理每个数据块
        print(f"使用 {self.n_jobs} 个进程处理 {len(chunks)} 个数据块...")
        with Pool(self.n_jobs) as pool:
            worker_func = partial(self._process_chunk, window_size=self.window_size, future_days=self.future_days)
            
            # 使用tqdm监控进度
            chunk_results = []
            with tqdm(total=len(chunks), desc="并行处理数据块", unit="块") as pbar:
                for result in pool.imap(worker_func, chunks):
                    chunk_results.append(result)
                    pbar.update(1)
        
        # 合并结果
        combined_results = defaultdict(lambda: {
            'frequency': 0,
            **{f'returns_{day}d': {'data': []} for day in self.future_days}
        })
        
        for chunk_result in chunk_results:
            for pattern, stats in chunk_result.items():
                combined_results[pattern]['frequency'] += stats['frequency']
                for day in self.future_days:
                    combined_results[pattern][f'returns_{day}d']['data'].extend(
                        stats[f'returns_{day}d']['data']
                    )
        
        # 计算最终统计量
        final_results = {}
        for pattern, stats in combined_results.items():
            pattern_result = {'frequency': stats['frequency']}
            
            for day in self.future_days:
                returns_data = stats[f'returns_{day}d']['data']
                if returns_data:
                    pattern_result[f'returns_{day}d'] = {
                        'mean': np.mean(returns_data),
                        'std': np.std(returns_data),
                        'count': len(returns_data),
                        'raw_data': returns_data
                    }
                else:
                    pattern_result[f'returns_{day}d'] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'count': 0,
                        'raw_data': []
                    }
            
            final_results[pattern] = pattern_result
        
        return final_results
    
    @staticmethod
    def _process_chunk(chunk_data: Tuple[np.ndarray, int, int], window_size: int, future_days: List[int]) -> Dict[Tuple, Dict]:
        """
        处理数据块的静态方法（用于多进程）
        
        Args:
            chunk_data: (价格数据, 起始偏移, 窗口数量)
            window_size: 窗口大小
            future_days: 预测天数列表
            
        Returns:
            该块的分析结果
        """
        prices, offset, num_windows = chunk_data
        
        patterns = []
        future_returns = {day: [] for day in future_days}
        
        # 生成排序模式和计算未来收益率
        for i in range(num_windows):
            if i + window_size > len(prices):
                break
                
            # 提取窗口内的价格
            window_prices = prices[i:i + window_size]
            
            # 计算排序
            ranks = RankedKLineAnalyzer._calculate_ranks_static(window_prices)
            pattern = tuple(ranks)
            patterns.append(pattern)
            
            # 计算未来收益率
            current_price = prices[i + window_size - 1]
            
            for day in future_days:
                future_idx = i + window_size - 1 + day
                if future_idx < len(prices):
                    future_price = prices[future_idx]
                    return_rate = (future_price - current_price) / current_price
                    future_returns[day].append(return_rate)
                else:
                    future_returns[day].append(np.nan)
        
        # 统计模式
        pattern_stats = defaultdict(lambda: {
            'frequency': 0,
            **{f'returns_{day}d': {'data': []} for day in future_days}
        })
        
        for i, pattern in enumerate(patterns):
            pattern_stats[pattern]['frequency'] += 1
            
            for day in future_days:
                if i < len(future_returns[day]) and not np.isnan(future_returns[day][i]):
                    pattern_stats[pattern][f'returns_{day}d']['data'].append(
                        future_returns[day][i]
                    )
        
        return dict(pattern_stats)
    
    @staticmethod
    def _calculate_ranks_static(values: np.ndarray) -> List[int]:
        """
        静态方法：计算数值的排序
        
        Args:
            values: 数值数组
            
        Returns:
            排序列表（1表示最小值，n表示最大值）
        """
        sorted_indices = np.argsort(values)
        ranks = np.empty_like(sorted_indices)
        ranks[sorted_indices] = np.arange(1, len(values) + 1)
        return ranks.tolist()
    
    def _apply_grouping(self, results: Dict[Tuple, Dict]) -> Dict[str, Dict]:
        """
        对分析结果应用分组
        
        Args:
            results: 原始分析结果
            
        Returns:
            分组后的结果
        """
        # 动态导入模式分组器
        import sys
        from pathlib import Path
        
        # 添加项目根目录到sys.path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        from src.utils.pattern_grouper import PatternGrouper
        
        grouper = PatternGrouper(
            correlation_threshold=self.correlation_threshold,
            batch_size=5000  # 默认批量大小，可以根据内存情况调整
        )
        grouped_results = grouper.group_patterns(results)
        
        return grouped_results
    
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
        if self.n_jobs == 1:
            # 单线程处理
            all_patterns, all_returns = self._process_stocks_sequential(stocks_data, price_type)
        else:
            # 并行处理
            all_patterns, all_returns = self._process_stocks_parallel(stocks_data, price_type)
        
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
        
        # 应用分组功能（如果启用）
        if self.enable_grouping and len(merged_results) > 1:
            merged_results = self._apply_grouping(merged_results)
        
        return merged_results
    
    def _process_stocks_sequential(self, stocks_data: Dict[str, pd.DataFrame], 
                                 price_type: str) -> Tuple[defaultdict, Dict]:
        """
        串行处理股票数据
        
        Args:
            stocks_data: 股票数据字典
            price_type: 价格类型
            
        Returns:
            模式和收益率数据
        """
        all_patterns = defaultdict(list)
        all_returns = {day: defaultdict(list) for day in self.future_days}
        
        # 处理每只股票
        with tqdm(total=len(stocks_data), desc="处理股票", unit="只") as pbar:
            for stock_code, df in stocks_data.items():
                try:
                    pbar.set_description(f"处理股票: {stock_code}")
                    
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
                finally:
                    pbar.update(1)  # 无论成功还是失败都更新进度条
        
        return all_patterns, all_returns
    
    def _process_stocks_parallel(self, stocks_data: Dict[str, pd.DataFrame], 
                               price_type: str) -> Tuple[defaultdict, Dict]:
        """
        并行处理股票数据
        
        Args:
            stocks_data: 股票数据字典
            price_type: 价格类型
            
        Returns:
            模式和收益率数据
        """
        # 准备股票列表
        stock_items = list(stocks_data.items())
        
        # 并行处理股票
        print(f"使用 {self.n_jobs} 个进程并行处理 {len(stock_items)} 只股票...")
        with Pool(self.n_jobs) as pool:
            worker_func = partial(self._process_single_stock_worker, 
                                price_type=price_type, 
                                window_size=self.window_size,
                                future_days=self.future_days)
            
            # 使用tqdm监控进度
            results = []
            with tqdm(total=len(stock_items), desc="并行处理股票", unit="只") as pbar:
                for result in pool.imap(worker_func, stock_items):
                    results.append(result)
                    pbar.update(1)
        
        # 合并结果
        all_patterns = defaultdict(list)
        all_returns = {day: defaultdict(list) for day in self.future_days}
        
        for stock_code, stock_results in results:
            if stock_results is None:
                continue
                
            print(f"合并股票结果: {stock_code}")
            
            for pattern, stats in stock_results.items():
                all_patterns[pattern].append(stats['frequency'])
                
                for day in self.future_days:
                    returns_key = f'returns_{day}d'
                    if returns_key in stats:
                        all_returns[day][pattern].extend(stats[returns_key]['raw_data'])
        
        return all_patterns, all_returns
    
    @staticmethod
    def _process_single_stock_worker(stock_item: Tuple[str, pd.DataFrame], 
                                   price_type: str, window_size: int, 
                                   future_days: List[int]) -> Tuple[str, Dict]:
        """
        处理单只股票的工作函数（用于多进程）
        
        Args:
            stock_item: (股票代码, DataFrame)
            price_type: 价格类型
            window_size: 窗口大小
            future_days: 预测天数列表
            
        Returns:
            (股票代码, 分析结果)
        """
        stock_code, df = stock_item
        
        try:
            # 创建临时分析器（n_jobs=1避免递归并行）
            temp_analyzer = RankedKLineAnalyzer(window_size=window_size, n_jobs=1)
            
            # 分析单只股票
            stock_results = temp_analyzer.analyze_single_stock(df, price_type)
            
            return stock_code, stock_results
            
        except Exception as e:
            print(f"处理股票 {stock_code} 时出错: {str(e)}")
            return stock_code, None
    
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