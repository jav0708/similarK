"""
排列模式分组器
基于相关系数将相似的排列模式归组，减少输出文件大小
"""
import numpy as np
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict
import scipy.stats as stats
from tqdm import tqdm


class PatternGrouper:
    """排列模式分组器"""
    
    def __init__(self, correlation_threshold: float = 0.9, batch_size: int = 5000, min_frequency: int = 10):
        """
        初始化分组器
        
        Args:
            correlation_threshold: 相关系数阈值，默认0.9
            batch_size: 增量分组时的批量大小，默认10000
            min_frequency: 最小频率阈值，用于分组后过滤
        """
        self.correlation_threshold = correlation_threshold
        self.batch_size = batch_size
        self.min_frequency = min_frequency
    
    def group_patterns(self, results: Dict[Tuple, Dict]) -> Dict[str, Dict]:
        """
        对排列模式进行分组（增量分组算法，避免内存爆炸）
        
        Args:
            results: 原始分析结果，格式为 {pattern: stats}
            
        Returns:
            分组后的结果，格式为 {group_id: group_stats}
        """
        if not results:
            return {}
        
        total_patterns = len(results)
        print(f"开始增量分组 {total_patterns} 个模式...")
        
        # 检查是否需要使用增量分组
        if total_patterns <= 2000:
            # 小数据集直接使用原算法
            print("数据量较小，使用完整矩阵算法...")
            return self._group_patterns_full_matrix(results)
        else:
            # 大数据集使用增量分组
            print("数据量较大，使用增量分组算法...")
            return self._group_patterns_incremental(results)
    
    def _group_patterns_full_matrix(self, results: Dict[Tuple, Dict]) -> Dict[str, Dict]:
        """原始的完整矩阵分组算法"""
        patterns = list(results.keys())
        similarity_matrix = self._calculate_pattern_similarity_matrix(patterns, results)
        groups = self._cluster_patterns(patterns, similarity_matrix)
        grouped_results = self._merge_group_stats(groups, results)
        
        # 在分组后应用频率过滤
        filtered_grouped_results = {group_id: stats for group_id, stats in grouped_results.items() 
                                   if stats.get('frequency', 0) >= self.min_frequency}
        
        print(f"分组完成：{len(results)} 个模式 -> {len(grouped_results)} 个组")
        print(f"频率过滤：{len(grouped_results)} 个组 -> {len(filtered_grouped_results)} 个组（频率>={self.min_frequency}）")
        return filtered_grouped_results
    
    def _group_patterns_incremental(self, results: Dict[Tuple, Dict]) -> Dict[str, Dict]:
        """
        增量分组算法，避免内存爆炸
        
        Args:
            results: 原始分析结果
            
        Returns:
            分组后的结果
        """
        patterns = list(results.keys())
        total_patterns = len(patterns)
        
        # 存储已建立的组
        established_groups = []  # 每个元素是一个组，包含模式索引列表
        group_representatives = []  # 每个组的代表模式
        
        print(f"使用批量大小: {self.batch_size}")
        
        # 分批处理
        total_batches = (total_patterns + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=total_batches, desc="增量分组进度", unit="批次") as batch_pbar:
            for batch_idx, batch_start in enumerate(range(0, total_patterns, self.batch_size)):
                batch_end = min(batch_start + self.batch_size, total_patterns)
                batch_patterns = patterns[batch_start:batch_end]
                batch_size_actual = len(batch_patterns)
                
                print(f"\n处理批次 {batch_idx + 1}/{total_batches}: 模式 {batch_start+1}-{batch_end} ({batch_size_actual} 个)")
                
                if not established_groups:
                    # 第一批：直接分组
                    similarity_matrix = self._calculate_pattern_similarity_matrix(batch_patterns, results)
                    batch_groups = self._cluster_patterns(batch_patterns, similarity_matrix)
                    
                    # 建立初始组
                    for group_indices in batch_groups:
                        absolute_indices = [batch_start + i for i in group_indices]
                        established_groups.append(absolute_indices)
                        # 选择第一个模式作为代表
                        group_representatives.append(patterns[absolute_indices[0]])
                    
                    print(f"  建立初始 {len(batch_groups)} 个组")
                else:
                    # 后续批次：逐个模式尝试加入现有组或创建新组
                    new_groups = []
                    
                    # 为当前批次的模式匹配添加进度条
                    with tqdm(total=batch_size_actual, desc="  模式匹配", unit="模式", leave=False) as pattern_pbar:
                        for i, pattern in enumerate(batch_patterns):
                            pattern_index = batch_start + i
                            assigned = False
                            
                            # 尝试加入现有组
                            for group_idx, representative in enumerate(group_representatives):
                                similarity = self._calculate_pattern_similarity(pattern, representative)
                                
                                if similarity >= self.correlation_threshold:
                                    established_groups[group_idx].append(pattern_index)
                                    assigned = True
                                    break
                            
                            # 如果无法加入现有组，创建新组
                            if not assigned:
                                new_groups.append(pattern_index)
                            
                            pattern_pbar.update(1)
                    
                    # 处理无法分配的模式（在它们之间进行分组）
                    if new_groups:
                        if len(new_groups) == 1:
                            # 只有一个模式，直接创建组
                            established_groups.append(new_groups)
                            group_representatives.append(patterns[new_groups[0]])
                        else:
                            # 多个模式，计算它们之间的相似性并分组
                            print(f"  对 {len(new_groups)} 个未分配模式进行内部分组...")
                            new_patterns = [patterns[idx] for idx in new_groups]
                            similarity_matrix = self._calculate_pattern_similarity_matrix(new_patterns, results)
                            internal_groups = self._cluster_patterns(new_patterns, similarity_matrix)
                            
                            for internal_group in internal_groups:
                                absolute_indices = [new_groups[i] for i in internal_group]
                                established_groups.append(absolute_indices)
                                group_representatives.append(patterns[absolute_indices[0]])
                    
                    print(f"  当前总组数: {len(established_groups)}")
                
                batch_pbar.update(1)
        
        print(f"增量分组完成：{total_patterns} 个模式 -> {len(established_groups)} 个组")
        
        # 合并统计数据
        grouped_results = self._merge_group_stats(established_groups, results)
        
        # 在分组后应用频率过滤
        filtered_grouped_results = {group_id: stats for group_id, stats in grouped_results.items() 
                                   if stats.get('frequency', 0) >= self.min_frequency}
        
        print(f"频率过滤：{len(grouped_results)} 个组 -> {len(filtered_grouped_results)} 个组（频率>={self.min_frequency}）")
        return filtered_grouped_results
    
    def _calculate_pattern_similarity_matrix(self, patterns: List[Tuple], 
                                           results: Dict[Tuple, Dict]) -> np.ndarray:
        """
        计算排列模式之间的结构相似性矩阵
        使用Spearman秩相关系数来衡量排列的结构相似性
        
        Args:
            patterns: 排列模式列表
            results: 分析结果字典（未使用，保持接口一致）
            
        Returns:
            相似性矩阵
        """
        n = len(patterns)
        similarity_matrix = np.eye(n)  # 对角线为1
        
        print(f"计算 {n}x{n} 排列结构相似性矩阵...")
        
        # 使用tqdm显示进度条
        with tqdm(total=n, desc="计算相似性矩阵", unit="行") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    try:
                        pattern_i = np.array(patterns[i])
                        pattern_j = np.array(patterns[j])
                        
                        # 计算排列之间的距离相似性
                        # 使用归一化的逆距离作为相似性度量
                        max_possible_diff = 2 * sum(range(1, len(pattern_i) + 1))  # 更准确的最大距离
                        actual_distance = np.sum(np.abs(pattern_i - pattern_j))
                        normalized_distance = actual_distance / max_possible_diff
                        similarity = 1.0 - normalized_distance  # 距离越小，相似性越高
                        
                        similarity_matrix[i, j] = max(0.0, similarity)
                        similarity_matrix[j, i] = max(0.0, similarity)
                            
                    except Exception as e:
                        similarity_matrix[i, j] = 0.0
                        similarity_matrix[j, i] = 0.0
                
                pbar.update(1)  # 更新进度条
        
        return similarity_matrix
    
    def _calculate_pattern_similarity(self, pattern1: Tuple, pattern2: Tuple) -> float:
        """
        计算两个模式之间的相似性
        
        Args:
            pattern1: 第一个模式
            pattern2: 第二个模式
            
        Returns:
            相似性分数 (0.0-1.0)
        """
        try:
            pattern_i = np.array(pattern1)
            pattern_j = np.array(pattern2)
            
            # 计算归一化距离相似性
            max_possible_diff = 2 * sum(range(1, len(pattern_i) + 1))
            actual_distance = np.sum(np.abs(pattern_i - pattern_j))
            normalized_distance = actual_distance / max_possible_diff
            similarity = 1.0 - normalized_distance
            
            return max(0.0, similarity)
        except Exception:
            return 0.0
    
    def _cluster_patterns(self, patterns: List[Tuple], 
                         similarity_matrix: np.ndarray) -> List[List[int]]:
        """
        基于相关系数矩阵进行聚类
        
        Args:
            patterns: 排列模式列表
            similarity_matrix: 相关系数矩阵
            
        Returns:
            分组列表，每个组包含模式的索引
        """
        n = len(patterns)
        visited = [False] * n
        groups = []
        
        for i in range(n):
            if visited[i]:
                continue
            
            # 开始新的组
            group = [i]
            visited[i] = True
            
            # 使用BFS找到所有高相关性的模式
            queue = [i]
            
            while queue:
                current = queue.pop(0)
                
                # 查找与当前模式高度相关的未访问模式
                for j in range(n):
                    if (not visited[j] and 
                        similarity_matrix[current, j] >= self.correlation_threshold):
                        group.append(j)
                        visited[j] = True
                        queue.append(j)
            
            groups.append(group)
        
        return groups
    
    def _merge_group_stats(self, groups: List[List[int]], 
                          results: Dict[Tuple, Dict]) -> Dict[str, Dict]:
        """
        合并每组的统计数据
        
        Args:
            groups: 分组列表
            results: 原始分析结果
            
        Returns:
            合并后的组统计数据
        """
        patterns = list(results.keys())
        grouped_results = {}
        future_days = [1, 3, 5, 10, 20]
        
        for group_idx, group in enumerate(groups):
            group_id = f"Group_{group_idx + 1:04d}"
            
            # 收集组内所有模式的数据
            group_patterns = [patterns[i] for i in group]
            total_frequency = 0
            combined_returns = {day: [] for day in future_days}
            
            # 合并组内所有模式的数据
            for pattern_idx in group:
                pattern = patterns[pattern_idx]
                pattern_stats = results[pattern]
                
                total_frequency += pattern_stats.get('frequency', 0)
                
                # 合并收益率数据
                for day in future_days:
                    returns_key = f'returns_{day}d'
                    if returns_key in pattern_stats:
                        raw_data = pattern_stats[returns_key].get('raw_data', [])
                        combined_returns[day].extend(raw_data)
            
            # 计算组的统计数据
            group_stats = {
                'group_id': group_id,
                'pattern_count': len(group_patterns),
                'representative_pattern': str(group_patterns[0]),  # 使用第一个作为代表
                'all_patterns': [str(p) for p in group_patterns],
                'frequency': total_frequency  # 使用frequency保持与非分组结果的一致性
            }
            
            # 调试信息：显示分组详情
            if len(group_patterns) > 5:  # 只显示大组的详情
                print(f"  {group_id}: {len(group_patterns)} 个模式")
                print(f"    代表模式: {group_patterns[0]}")
                print(f"    前5个模式: {group_patterns[:5]}")
                print(f"    总频率: {total_frequency}")
            
            # 计算各期收益率统计
            for day in future_days:
                returns_data = combined_returns[day]
                returns_key = f'returns_{day}d'
                
                if returns_data:
                    group_stats[returns_key] = {
                        'mean': np.mean(returns_data),
                        'std': np.std(returns_data),
                        'count': len(returns_data)
                    }
                else:
                    group_stats[returns_key] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'count': 0
                    }
            
            grouped_results[group_id] = group_stats
        
        return grouped_results
    
    def get_grouping_summary(self, original_count: int, grouped_count: int) -> Dict[str, Any]:
        """
        获取分组摘要信息
        
        Args:
            original_count: 原始模式数量
            grouped_count: 分组后数量
            
        Returns:
            分组摘要信息
        """
        reduction_rate = (original_count - grouped_count) / original_count if original_count > 0 else 0
        
        return {
            'original_patterns': original_count,
            'grouped_patterns': grouped_count,
            'reduction_rate': reduction_rate,
            'compression_ratio': original_count / grouped_count if grouped_count > 0 else 0,
            'correlation_threshold': self.correlation_threshold
        }