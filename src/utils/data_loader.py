"""
数据加载和预处理模块
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
import glob


class DataLoader:
    """数据加载器类"""
    
    def __init__(self):
        """初始化数据加载器"""
        self.column_mapping = {
            'stock_code': '股票代码',
            'stock_name': '股票简称', 
            'date': '交易日期',
            'open': '开盘价',
            'high': '最高价',
            'low': '最低价', 
            'close': '收盘价',
            'pre_close': '前收盘价',
            'volume': '成交量'
        }
    
    def load_single_stock(self, file_path: str) -> pd.DataFrame:
        """
        加载单个股票文件
        
        Args:
            file_path: 股票数据文件路径
            
        Returns:
            处理后的DataFrame
        """
        try:
            # 尝试不同编码读取CSV文件
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
            df = None
            
            for encoding in encodings:
                try:
                    # 先尝试直接读取
                    df = pd.read_csv(file_path, encoding=encoding)
                    # 检查是否需要跳过第一行
                    if '股票代码' not in df.columns and len(df.columns) > 0:
                        # 第一行可能是说明文字，跳过第一行重新读取
                        df = pd.read_csv(file_path, encoding=encoding, skiprows=1)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("无法解码文件，尝试了多种编码格式都失败")
            
            # 检查必需列是否存在
            required_cols = ['交易日期', '开盘价', '最高价', '最低价', '收盘价']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"缺少必需列: {missing_cols}")
            
            # 数据预处理
            df = self._preprocess_data(df)
            
            return df
            
        except Exception as e:
            raise Exception(f"加载文件 {file_path} 失败: {str(e)}")
    
    def load_batch_stocks(self, data_dir: str) -> Dict[str, pd.DataFrame]:
        """
        批量加载股票文件
        
        Args:
            data_dir: 数据文件夹路径
            
        Returns:
            股票代码到DataFrame的字典
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        
        # 查找所有CSV文件
        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"目录 {data_dir} 中没有找到CSV文件")
        
        stocks_data = {}
        failed_files = []
        
        for file_path in csv_files:
            try:
                df = self.load_single_stock(str(file_path))
                # 从文件名提取股票代码
                stock_code = file_path.stem
                stocks_data[stock_code] = df
                print(f"成功加载: {stock_code}")
                
            except Exception as e:
                failed_files.append((str(file_path), str(e)))
                print(f"加载失败: {file_path.name} - {str(e)}")
        
        if failed_files:
            print(f"\n警告: {len(failed_files)} 个文件加载失败")
        
        print(f"\n成功加载 {len(stocks_data)} 个股票文件")
        return stocks_data
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            预处理后的DataFrame
        """
        # 复制数据避免修改原始数据
        df = df.copy()
        
        # 转换日期列
        df['交易日期'] = pd.to_datetime(df['交易日期'])
        
        # 确保价格列为数值类型
        price_cols = ['开盘价', '最高价', '最低价', '收盘价']
        for col in price_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 删除包含NaN的行
        df = df.dropna(subset=price_cols)
        
        # 按日期排序
        df = df.sort_values('交易日期').reset_index(drop=True)
        
        # 验证数据有效性
        self._validate_data(df)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        验证数据有效性
        
        Args:
            df: 待验证的DataFrame
        """
        if len(df) == 0:
            raise ValueError("数据为空")
        
        # 检查价格是否为正数
        price_cols = ['开盘价', '最高价', '最低价', '收盘价']
        for col in price_cols:
            if col in df.columns:
                if (df[col] <= 0).any():
                    raise ValueError(f"{col} 包含非正数值")
        
        # 检查高低价关系
        if '最高价' in df.columns and '最低价' in df.columns:
            if (df['最高价'] < df['最低价']).any():
                raise ValueError("存在最高价小于最低价的异常数据")
    
    def get_stock_info(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        获取股票基本信息
        
        Args:
            df: 股票数据DataFrame
            
        Returns:
            股票信息字典
        """
        info = {}
        
        if '股票代码' in df.columns:
            info['stock_code'] = df['股票代码'].iloc[0]
        if '股票简称' in df.columns:
            info['stock_name'] = df['股票简称'].iloc[0]
        
        info['start_date'] = df['交易日期'].min().strftime('%Y-%m-%d')
        info['end_date'] = df['交易日期'].max().strftime('%Y-%m-%d')
        info['total_days'] = len(df)
        
        return info