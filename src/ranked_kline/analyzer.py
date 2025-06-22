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
import psutil
import time


class RankedKLineAnalyzer:
    """排序K线分析器 - 性能优化版本"""
    
    def __init__(self, window_size: int = 10, min_frequency: int = 10, n_jobs: int = 1, 
                 enable_grouping: bool = False, correlation_threshold: float = 0.9,
                 enable_monitoring: bool = True, batch_size: int = 1000):
        """
        初始化分析器
        
        Args:
            window_size: 时间窗口大小
            min_frequency: 最小频率阈值
            n_jobs: 并行处理数量
            enable_grouping: 是否启用分组功能
            correlation_threshold: 相关系数阈值
            enable_monitoring: 是否启用性能监控
            batch_size: 批处理大小
        """
        self.window_size = window_size
        self.min_frequency = min_frequency
        self.n_jobs = n_jobs
        self.enable_grouping = enable_grouping
        self.correlation_threshold = correlation_threshold
        self.enable_monitoring = enable_monitoring
        self.batch_size = batch_size
        self.future_days = [1, 3, 5, 10, 20]  # 预测天数
        
        # 性能监控相关
        self.start_time = None
        self.processed_stocks = 0
        self.total_stocks = 0
    
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
        处理数据块的静态方法（用于多进程）- 优化版本
        
        Args:
            chunk_data: (价格数据, 起始偏移, 窗口数量)
            window_size: 窗口大小
            future_days: 预测天数列表
            
        Returns:
            该块的分析结果
        """
        prices, offset, num_windows = chunk_data
        
        if num_windows <= 0 or len(prices) < window_size:
            return {}
        
        # 向量化批量排序计算
        patterns_array = RankedKLineAnalyzer._calculate_ranks_vectorized(prices, window_size, num_windows)
        
        # 向量化计算未来收益率
        future_returns = RankedKLineAnalyzer._calculate_future_returns_vectorized(
            prices, window_size, num_windows, future_days
        )
        
        # 使用numpy数组优化模式统计
        pattern_stats = RankedKLineAnalyzer._aggregate_patterns_optimized(
            patterns_array, future_returns, future_days, use_encoding=True
        )
        
        return pattern_stats
    
    @staticmethod
    def _calculate_ranks_vectorized(prices: np.ndarray, window_size: int, num_windows: int) -> np.ndarray:
        """
        向量化批量计算排序模式
        
        Args:
            prices: 价格序列
            window_size: 窗口大小
            num_windows: 窗口数量
            
        Returns:
            排序模式数组，形状为 (num_windows, window_size)
        """
        # 使用更兼容的方式创建滑动窗口矩阵
        try:
            # 尝试使用numpy 1.20+的sliding_window_view
            windows_matrix = np.lib.stride_tricks.sliding_window_view(prices[:num_windows + window_size - 1], window_size)
        except AttributeError:
            # 如果不可用，使用手动实现
            windows_matrix = np.array([prices[i:i + window_size] for i in range(num_windows)])
        
        # 批量计算排序
        sorted_indices = np.argsort(windows_matrix, axis=1)
        ranks = np.empty_like(sorted_indices)
        
        # 向量化计算排名
        for i in range(window_size):
            ranks[np.arange(len(windows_matrix)), sorted_indices[:, i]] = i + 1
        
        return ranks
    
    @staticmethod
    def _calculate_future_returns_vectorized(prices: np.ndarray, window_size: int, 
                                           num_windows: int, future_days: List[int]) -> Dict[int, np.ndarray]:
        """
        向量化计算未来收益率
        
        Args:
            prices: 价格序列
            window_size: 窗口大小
            num_windows: 窗口数量
            future_days: 预测天数列表
            
        Returns:
            未来收益率字典
        """
        future_returns = {}
        
        # 当前价格 (窗口结束位置)
        current_prices = prices[window_size - 1:window_size - 1 + num_windows]
        
        for day in future_days:
            future_indices = np.arange(window_size - 1 + day, window_size - 1 + day + num_windows)
            
            # 处理超出范围的索引
            valid_mask = future_indices < len(prices)
            returns = np.full(num_windows, np.nan)
            
            if np.any(valid_mask):
                valid_future_prices = prices[future_indices[valid_mask]]
                valid_current_prices = current_prices[valid_mask]
                returns[valid_mask] = (valid_future_prices - valid_current_prices) / valid_current_prices
            
            future_returns[day] = returns
        
        return future_returns
    
    @staticmethod
    def _aggregate_patterns_optimized(patterns_array: np.ndarray, future_returns: Dict[int, np.ndarray], 
                                    future_days: List[int], use_encoding: bool = True) -> Dict[Tuple, Dict]:
        """
        优化的模式统计聚合
        
        Args:
            patterns_array: 排序模式数组
            future_returns: 未来收益率字典
            future_days: 预测天数列表
            use_encoding: 是否使用整数编码优化内存
            
        Returns:
            模式统计字典
        """
        if use_encoding:
            # 使用整数编码节省内存
            window_size = patterns_array.shape[1]
            encoded_patterns = np.array([RankedKLineAnalyzer.encode_pattern(tuple(pattern), window_size) 
                                       for pattern in patterns_array])
            
            # 使用numpy数组优化统计
            unique_patterns, inverse_indices, counts = np.unique(encoded_patterns, return_inverse=True, return_counts=True)
            
            pattern_stats = {}
            
            for i, encoded_pattern in enumerate(unique_patterns):
                pattern = RankedKLineAnalyzer.decode_pattern(encoded_pattern, window_size)
                mask = inverse_indices == i
                
                pattern_stats[pattern] = {
                    'frequency': int(counts[i])
                }
                
                # 收集该模式对应的所有收益率
                for day in future_days:
                    returns_data = future_returns[day][mask]
                    valid_returns = returns_data[~np.isnan(returns_data)]
                    pattern_stats[pattern][f'returns_{day}d'] = {
                        'data': valid_returns.tolist()
                    }
        else:
            # 原始实现
            pattern_stats = defaultdict(lambda: {
                'frequency': 0,
                **{f'returns_{day}d': {'data': []} for day in future_days}
            })
            
            for i in range(len(patterns_array)):
                pattern = tuple(patterns_array[i])
                pattern_stats[pattern]['frequency'] += 1
                
                for day in future_days:
                    return_value = future_returns[day][i]
                    if not np.isnan(return_value):
                        pattern_stats[pattern][f'returns_{day}d']['data'].append(return_value)
            
            pattern_stats = dict(pattern_stats)
        
        return pattern_stats
    
    @staticmethod
    def encode_pattern(pattern: Tuple[int, ...], window_size: int) -> int:
        """
        将排序模式编码为单个整数，节省内存
        
        Args:
            pattern: 排序模式元组
            window_size: 窗口大小
            
        Returns:
            编码后的整数
        """
        encoded = 0
        base = window_size + 1
        for i, rank in enumerate(pattern):
            encoded += rank * (base ** i)
        return encoded
    
    @staticmethod
    def decode_pattern(encoded: int, window_size: int) -> Tuple[int, ...]:
        """
        将编码的整数解码回排序模式
        
        Args:
            encoded: 编码的整数
            window_size: 窗口大小
            
        Returns:
            解码后的排序模式
        """
        pattern = []
        base = window_size + 1
        for _ in range(window_size):
            pattern.append(encoded % base)
            encoded //= base
        return tuple(pattern)
    
    def _log_memory_usage(self, stage: str):
        """记录内存使用情况"""
        if not self.enable_monitoring:
            return
        
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"[{stage}] 内存使用: {memory_mb:.1f} MB")
    
    def _estimate_completion_time(self, processed: int, total: int, start_time: float):
        """估算完成时间"""
        if not self.enable_monitoring or processed == 0:
            return
        
        elapsed = time.time() - start_time
        rate = processed / elapsed
        remaining = total - processed
        eta_seconds = remaining / rate if rate > 0 else 0
        
        print(f"处理进度: {processed}/{total} ({processed/total*100:.1f}%), "
              f"速度: {rate:.1f} 股票/秒, 预计剩余时间: {eta_seconds/60:.1f} 分钟")
    
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
            batch_size=5000,  # 默认批量大小，可以根据内存情况调整
            min_frequency=self.min_frequency  # 传递频率阈值
        )
        grouped_results = grouper.group_patterns(results)
        
        return grouped_results
    
    def analyze_batch_stocks(self, stocks_data: Dict[str, pd.DataFrame], 
                           price_type: str) -> Dict[str, Any]:
        """
        批量分析多只股票 - 优化版本，支持分批处理
        
        Args:
            stocks_data: 股票代码到DataFrame的字典
            price_type: 价格类型
            
        Returns:
            合并的分析结果
        """
        self.total_stocks = len(stocks_data)
        self.processed_stocks = 0
        self.start_time = time.time()
        
        self._log_memory_usage("开始批量分析")
        
        # 判断是否需要分批处理
        if len(stocks_data) > self.batch_size:
            print(f"数据量大（{len(stocks_data)}只股票），启用分批处理（批大小：{self.batch_size}）")
            merged_results = self._process_stocks_in_batches(stocks_data, price_type)
        else:
            print(f"数据量小（{len(stocks_data)}只股票），使用普通处理")
            if self.n_jobs == 1:
                all_patterns, all_returns = self._process_stocks_sequential(stocks_data, price_type)
            else:
                all_patterns, all_returns = self._process_stocks_parallel(stocks_data, price_type)
            
            merged_results = self._merge_pattern_results(all_patterns, all_returns)
        
        self._log_memory_usage("分析完成")
        
        # 应用分组功能（如果启用）
        if self.enable_grouping and len(merged_results) > 1:
            self._log_memory_usage("开始模式分组")
            merged_results = self._apply_grouping(merged_results)
            self._log_memory_usage("模式分组完成")
        
        return merged_results
    
    def _process_stocks_in_batches(self, stocks_data: Dict[str, pd.DataFrame], 
                                 price_type: str) -> Dict[str, Any]:
        """
        分批处理股票数据，避免内存过载
        
        Args:
            stocks_data: 股票数据字典
            price_type: 价格类型
            
        Returns:
            合并的分析结果
        """
        stock_items = list(stocks_data.items())
        total_batches = (len(stock_items) + self.batch_size - 1) // self.batch_size
        
        # 总的模式统计
        global_patterns = defaultdict(list)
        global_returns = {day: defaultdict(list) for day in self.future_days}
        
        print(f"将{len(stock_items)}只股票分为{total_batches}个批次处理")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(stock_items))
            batch_stocks = dict(stock_items[start_idx:end_idx])
            
            print(f"\n处理第 {batch_idx + 1}/{total_batches} 批：股票 {start_idx + 1}-{end_idx}")
            self._log_memory_usage(f"批次 {batch_idx + 1} 开始")
            
            # 处理当前批次
            if self.n_jobs == 1:
                batch_patterns, batch_returns = self._process_stocks_sequential(batch_stocks, price_type)
            else:
                batch_patterns, batch_returns = self._process_stocks_parallel(batch_stocks, price_type)
            
            # 合并到全局结果
            for pattern, frequencies in batch_patterns.items():
                global_patterns[pattern].extend(frequencies)
            
            for day in self.future_days:
                for pattern, returns_data in batch_returns[day].items():
                    global_returns[day][pattern].extend(returns_data)
            
            # 释放当前批次的内存
            del batch_stocks, batch_patterns, batch_returns
            
            self._log_memory_usage(f"批次 {batch_idx + 1} 完成")
            self.processed_stocks = end_idx
            self._estimate_completion_time(self.processed_stocks, self.total_stocks, self.start_time)
        
        # 合并所有批次的结果
        print("\n合并所有批次的结果...")
        merged_results = self._merge_pattern_results(global_patterns, global_returns)
        
        return merged_results
    
    def _merge_pattern_results(self, all_patterns: defaultdict, all_returns: Dict) -> Dict[str, Any]:
        """
        合并模式统计结果
        注意：当启用分组功能时，不在此处进行频率过滤，而是在分组后再过滤
        
        Args:
            all_patterns: 所有模式频率数据
            all_returns: 所有模式收益率数据
            
        Returns:
            合并后的结果
        """
        merged_results = {}
        total_patterns = len(all_patterns)
        processed_patterns = 0
        
        print(f"合并{total_patterns}个独特模式的统计结果...")
        
        for pattern in all_patterns:
            # 计算总频率
            total_frequency = sum(all_patterns[pattern])
            
            # 重要修改：如果启用分组，则不在此处过滤，而是保留所有模式用于分组
            if not self.enable_grouping and total_frequency < self.min_frequency:
                processed_patterns += 1
                continue
                
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
            
            processed_patterns += 1
            if processed_patterns % 10000 == 0:
                progress = processed_patterns / total_patterns * 100
                print(f"合并进度: {processed_patterns}/{total_patterns} ({progress:.1f}%)")
        
        if self.enable_grouping:
            print(f"分组模式：保留{len(merged_results)}个模式用于分组（将在分组后应用频率过滤）")
        else:
            filtered_count = len(merged_results)
            print(f"过滤后保留{filtered_count}个模式（频率>={self.min_frequency}）")
        
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
        生成排序模式 - 优化版本
        
        Args:
            prices: 价格序列
            
        Returns:
            排序模式列表
        """
        num_windows = len(prices) - self.window_size + 1
        if num_windows <= 0:
            return []
        
        # 使用向量化批量排序
        patterns_array = RankedKLineAnalyzer._calculate_ranks_vectorized(prices, self.window_size, num_windows)
        
        # 转换为tuple列表
        patterns = [tuple(pattern) for pattern in patterns_array]
        
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
        计算未来收益率 - 优化版本
        
        Args:
            prices: 价格序列
            
        Returns:
            未来收益率字典
        """
        num_windows = len(prices) - self.window_size + 1
        if num_windows <= 0:
            return {day: [] for day in self.future_days}
        
        # 使用向量化计算
        future_returns_dict = RankedKLineAnalyzer._calculate_future_returns_vectorized(
            prices, self.window_size, num_windows, self.future_days
        )
        
        # 转换为list格式以保持兼容性
        result = {}
        for day in self.future_days:
            result[day] = future_returns_dict[day].tolist()
        
        return result
    
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