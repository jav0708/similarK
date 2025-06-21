"""
命令行参数解析模块
"""
import argparse
import sys
from pathlib import Path


class CLIParser:
    """命令行参数解析器"""
    
    def __init__(self):
        """初始化解析器"""
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """创建参数解析器"""
        parser = argparse.ArgumentParser(
            description="相似K线寻找 - Ranked KLine算法",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
使用示例:
  # 单股分析
  python main.py --mode single --stock_file data_example/sh600082.csv --price_type open -w 10
  
  # 批量分析
  python main.py --mode batch --data_dir data_example/ --price_type close -w 20
  
  # 使用并行处理加速分析
  python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 15 -j 4
  
  # 使用所有CPU核心
  python main.py --mode batch --data_dir data_example/ --price_type open -w 10 -j -1
  
  # 启用分组功能减少文件大小
  python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 8 --enable_grouping
  
  # 自定义相关系数阈值
  python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 6 --enable_grouping --correlation_threshold 0.8
            """
        )
        
        # 必需参数
        parser.add_argument(
            '--mode',
            choices=['single', 'batch'],
            required=True,
            help='处理模式: single=单股分析, batch=批量分析'
        )
        
        parser.add_argument(
            '--price_type', '-p',
            choices=['open', 'high', 'low', 'close'],
            required=True,
            help='价格类型: open=开盘价, high=最高价, low=最低价, close=收盘价'
        )
        
        parser.add_argument(
            '--window', '-w',
            type=int,
            required=True,
            help='时间窗口长度'
        )
        
        # 条件必需参数
        parser.add_argument(
            '--stock_file',
            type=str,
            help='单个股票文件路径 (single模式必需)'
        )
        
        parser.add_argument(
            '--data_dir',
            type=str,
            help='数据文件夹路径 (batch模式必需)'
        )
        
        # 可选参数
        parser.add_argument(
            '--output_dir', '-o',
            type=str,
            default='output',
            help='输出目录 (默认: output/)'
        )
        
        parser.add_argument(
            '--min_frequency',
            type=int,
            default=10,
            help='最小模式频率阈值 (默认: 10)'
        )
        
        parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='详细输出模式'
        )
        
        parser.add_argument(
            '--n_jobs', '-j',
            type=int,
            default=1,
            help='并行处理数量 (默认: 1, -1表示使用所有CPU核心)'
        )
        
        parser.add_argument(
            '--enable_grouping',
            action='store_true',
            help='启用模式分组功能，减少输出文件大小'
        )
        
        parser.add_argument(
            '--correlation_threshold',
            type=float,
            default=0.9,
            help='分组时的相关系数阈值 (默认: 0.9)'
        )
        
        return parser
    
    def parse_args(self) -> argparse.Namespace:
        """解析命令行参数"""
        args = self.parser.parse_args()
        
        # 验证参数
        self._validate_args(args)
        
        return args
    
    def _validate_args(self, args: argparse.Namespace) -> None:
        """验证参数有效性"""
        
        # 验证模式相关参数
        if args.mode == 'single':
            if not args.stock_file:
                self.parser.error("single模式需要指定 --stock_file 参数")
            
            # 检查文件是否存在
            stock_path = Path(args.stock_file)
            if not stock_path.exists():
                self.parser.error(f"股票文件不存在: {args.stock_file}")
            
            if not stock_path.suffix.lower() == '.csv':
                self.parser.error(f"股票文件必须是CSV格式: {args.stock_file}")
                
        elif args.mode == 'batch':
            if not args.data_dir:
                self.parser.error("batch模式需要指定 --data_dir 参数")
            
            # 检查目录是否存在
            data_path = Path(args.data_dir)
            if not data_path.exists():
                self.parser.error(f"数据目录不存在: {args.data_dir}")
            
            if not data_path.is_dir():
                self.parser.error(f"数据路径不是目录: {args.data_dir}")
        
        # 验证窗口大小
        if args.window < 2:
            self.parser.error("窗口长度必须大于等于2")
        
        if args.window > 100:
            self.parser.error("窗口长度不能超过100")
        
        # 验证最小频率
        if args.min_frequency < 1:
            self.parser.error("最小频率必须大于等于1")
        
        # 验证并行数量
        if args.n_jobs < -1 or args.n_jobs == 0:
            self.parser.error("并行数量必须为正数或-1（表示使用所有CPU核心）")
        
        # 处理-1的情况（使用所有CPU核心）
        if args.n_jobs == -1:
            import multiprocessing
            args.n_jobs = multiprocessing.cpu_count()
        
        # 验证相关系数阈值
        if args.correlation_threshold < 0.0 or args.correlation_threshold > 1.0:
            self.parser.error("相关系数阈值必须在0.0到1.0之间")
    
    def print_help(self):
        """打印帮助信息"""
        self.parser.print_help()


def parse_command_line() -> argparse.Namespace:
    """
    解析命令行参数的便捷函数
    
    Returns:
        解析后的参数对象
    """
    parser = CLIParser()
    return parser.parse_args()