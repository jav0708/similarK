# 相似K线寻找

## 项目概述

这个项目专注于从A股市场历史数据中寻找相似的K线模式，并通过统计分析来预测未来收益率。通过Ranked KLine算法和模型，我们能够识别出具有统计显著性的K线模式，为量化交易提供数据支持。

## 项目目标

- 识别具有预测价值的K线模式
- 统计不同模式对应的未来收益率分布
- 提供可配置的时间窗口和预测周期
- 支持大规模数据批量处理

## 子项目架构

这个项目采用Ranked KLine（排序K线分析）方法，通过统计不同排序组合的出现频率以及未来n天收益率，来识别具有预测价值的K线模式。

**核心思想**: 通过对时间窗口内的开高收低价格进行排序，识别具有相似排序模式的K线组合。

**实现方法**:
- 设定滑动时间窗口长度（可配置，默认20根K线）
- 对窗口内每根K线的开高收低价格分别进行排序
- 统计不同排序组合的出现频率
- 计算每种模式对应的未来n天收益率（n=1-20）

**示例说明**:
假设时间窗口长度为10，某个窗口内开盘价排序为：[10,5,3,1,2,4,7,6,8,9]，表示第1根K线开盘价最高，第4根最低。

## 安装和使用

### 环境要求

- Python 3.8 或更高版本
- 内存建议 8GB 以上（处理大量股票数据）
- 硬盘空间 10GB 以上（用于数据存储）

### 安装步骤

1. 克隆项目
```bash
git clone https://github.com/your-username/similar_K.git
cd similar_K
```

2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或者 venv\Scripts\activate  # Windows
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

### 快速开始

#### 1. 数据准备
将K线数据放入 `data/` 目录，支持的格式：
- CSV文件：包含日期、开盘价、最高价、最低价、收盘价、成交量列
- HDF5文件：pandas DataFrame格式

#### 2. 运行排序K线分析
```bash
python main.py --method ranked_kline --window 10 --future_days 5 --data_path data/stock_data.csv
```

#### 3. 查看结果
结果将保存在 `output/` 目录下，包括：
- 统计报告（CSV格式）
- 可视化图表（PNG格式）
- 详细分析日志

## API文档

### 命令行参数

```bash
python main.py [OPTIONS]
```

#### 通用参数
- `--method, -m`: 分析方法 (ranked_kline, pattern_matching, clustering, deep_learning)
- `--data_path, -d`: 数据文件路径
- `--output_dir, -o`: 输出目录 (默认: output/)
- `--verbose, -v`: 详细输出模式
- `--config, -c`: 配置文件路径 (可选)

#### Ranked KLine 特定参数
- `--window, -w`: 时间窗口长度 (默认: 10)
- `--future_days, -f`: 预测天数 (默认: 1-20)
- `--min_frequency`: 最小模式频率阈值 (默认: 10)
- `--price_types`: 价格类型组合 (open, high, low, close)

#### Pattern Matching 参数
- `--similarity_threshold`: 相似度阈值 (0-1, 默认: 0.8)
- `--dtw_window`: DTW窗口大小 (默认: 10)
- `--feature_weights`: 特征权重配置

#### Clustering 参数
- `--n_clusters, -k`: 聚类数量 (默认: 自动确定)
- `--algorithm`: 聚类算法 (kmeans, hierarchical, dbscan)
- `--features`: 特征选择列表

#### Deep Learning 参数
- `--model_type`: 模型类型 (cnn, lstm, autoencoder)
- `--epochs`: 训练轮数 (默认: 100)
- `--batch_size`: 批处理大小 (默认: 32)
- `--learning_rate`: 学习率 (默认: 0.001)

### 配置文件示例

```yaml
# config.yaml
data:
  path: "data/stock_data.csv"
  date_column: "date"
  price_columns: ["open", "high", "low", "close"]
  volume_column: "volume"

analysis:
  method: "ranked_kline"
  window_size: 10
  future_days: [1, 3, 5, 10, 20]
  min_frequency: 10

output:
  directory: "output/"
  formats: ["csv", "json", "html"]
  visualization: true
```

## 使用示例

### 示例1：基础排序K线分析

```bash
# 分析平安银行（000001）的10日窗口K线模式
python main.py -m ranked_kline -d data/000001.csv -w 10 -f 5 -o results/
```

预期输出：
```
正在处理股票数据: 000001.csv
时间窗口长度: 10
预测天数: 5
找到 1247 个有效的K线模式
生成统计报告...
保存结果到: results/000001_ranked_analysis.csv
```

### 示例2：批量处理多只股票

```bash
# 批量处理data目录下所有股票文件
python batch_process.py -d data/ -m ranked_kline -w 15 --parallel 4
```

### 示例3：使用配置文件

```bash
python main.py -c configs/advanced_analysis.yaml
```

## 数据格式说明

### 输入数据格式

#### CSV格式 (推荐)
```csv
date,open,high,low,close,volume
2023-01-01,10.25,10.58,10.12,10.45,1250000
2023-01-02,10.46,10.89,10.35,10.78,1380000
2023-01-03,10.79,10.95,10.52,10.63,1120000
...
```

**必需列**:
- `date`: 日期 (YYYY-MM-DD 格式)
- `open`: 开盘价
- `high`: 最高价  
- `low`: 最低价
- `close`: 收盘价
- `volume`: 成交量 (可选)

#### HDF5格式
```python
import pandas as pd

# 读取示例
df = pd.read_hdf('data/stock_data.h5', key='stock_000001')
print(df.head())
```

### 输出结果格式

#### 统计报告 (CSV)
```csv
pattern_id,open_rank,high_rank,low_rank,close_rank,frequency,avg_return_1d,avg_return_3d,avg_return_5d,std_return_1d,sharpe_ratio
1,"[1,2,3,4,5,6,7,8,9,10]","[3,1,5,2,8,4,9,6,10,7]",...,156,0.0125,0.0287,0.0456,0.0234,0.534
2,"[2,4,1,5,3,7,6,9,8,10]","[1,3,2,6,4,8,5,10,7,9]",...,98,0.0089,0.0201,0.0334,0.0198,0.449
...
```

**输出列说明**:
- `pattern_id`: 模式唯一标识
- `*_rank`: 各价格的排序序列
- `frequency`: 模式出现频率
- `avg_return_*d`: 未来N天平均收益率
- `std_return_*d`: 收益率标准差
- `sharpe_ratio`: 夏普比率

#### 可视化图表
- **频率分布图**: 显示各模式的出现频率
- **收益率分布图**: 不同模式的收益率分布对比
- **时间序列图**: 模式在时间轴上的分布
- **热力图**: 模式相关性矩阵

## 算法原理和技术细节

### Ranked KLine 算法详解

#### 1. 滑动窗口机制
```python
def sliding_window(data, window_size):
    """生成滑动窗口"""
    for i in range(len(data) - window_size + 1):
        yield data[i:i + window_size]
```

#### 2. 价格排序算法
对于时间窗口内的每根K线，计算其开高收低价格在窗口内的相对排序：

```
排序公式: rank(price_i) = sum(price_j <= price_i for j in window) 
其中 i 为当前K线位置，j 为窗口内所有位置
```

#### 3. 模式哈希化
将排序序列转换为唯一的模式标识符：
```python
def pattern_hash(open_ranks, high_ranks, low_ranks, close_ranks):
    return f"{open_ranks}_{high_ranks}_{low_ranks}_{close_ranks}"
```

#### 4. 收益率计算
```python
def calculate_returns(prices, days):
    """计算未来N天收益率"""
    returns = []
    for i in range(len(prices) - days):
        ret = (prices[i + days] - prices[i]) / prices[i]
        returns.append(ret)
    return returns
```

### Pattern Matching 算法

#### 动态时间规整 (DTW)
用于计算两个K线序列的相似度：

```python
def dtw_distance(seq1, seq2):
    """计算DTW距离"""
    n, m = len(seq1), len(seq2)
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(seq1[i-1] - seq2[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )
    
    return dtw_matrix[n, m]
```

### 性能优化策略

#### 1. 并行处理
- 使用多进程处理不同股票
- 向量化计算减少循环开销
- 内存映射处理大文件

#### 2. 缓存机制
```python
from functools import lru_cache

@lru_cache(maxsize=10000)
def compute_pattern_stats(pattern_key):
    """缓存模式统计结果"""
    pass
```

#### 3. 数据预处理
- 预计算技术指标
- 数据标准化和归一化
- 异常值检测和处理

### 统计显著性检验

#### t检验
验证模式收益率是否显著异于随机：
```python
from scipy import stats

def significance_test(returns, alpha=0.05):
    """进行t检验"""
    t_stat, p_value = stats.ttest_1samp(returns, 0)
    return p_value < alpha
```

#### 夏普比率计算
```python
def sharpe_ratio(returns, risk_free_rate=0):
    """计算夏普比率"""
    excess_returns = np.array(returns) - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)
```

## 项目结构

```
similar_K/
├── main.py                 # 主入口程序
├── batch_process.py        # 批量处理工具
├── requirements.txt        # 依赖包列表
├── config.yaml            # 默认配置文件
├── README.md              # 项目文档
│
├── src/                   # 源代码目录
│   ├── __init__.py
│   ├── ranked_kline/      # 排序K线模块
│   │   ├── __init__.py
│   │   ├── analyzer.py
│   │   └── utils.py
│   ├── pattern_matching/  # 模式匹配模块
│   │   ├── __init__.py
│   │   ├── dtw.py
│   │   └── features.py
│   ├── clustering/        # 聚类分析模块
│   │   ├── __init__.py
│   │   └── kmeans.py
│   ├── deep_learning/     # 深度学习模块
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── training.py
│   └── utils/             # 通用工具
│       ├── __init__.py
│       ├── data_loader.py
│       ├── visualizer.py
│       └── statistics.py
│
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   └── examples/         # 示例数据
│
├── output/               # 输出结果
│   ├── reports/          # 分析报告
│   ├── charts/           # 图表文件
│   └── models/           # 训练模型
│
├── configs/              # 配置文件
│   ├── ranked_kline.yaml
│   ├── pattern_matching.yaml
│   └── deep_learning.yaml
│
├── tests/                # 测试代码
│   ├── __init__.py
│   ├── test_ranked_kline.py
│   └── test_utils.py
│
└── docs/                 # 详细文档
    ├── algorithm_details.md
    ├── performance_tuning.md
    └── api_reference.md
```

## 开发计划

### 版本 1.0 (当前开发中)
- [x] Ranked KLine 基础实现
- [x] 命令行接口设计
- [x] 基础数据处理和统计功能
- [ ] 单元测试覆盖
- [ ] 性能优化

### 版本 1.1 (计划中)
- [ ] Pattern Matching 模块
- [ ] 多进程并行处理
- [ ] Web界面支持
- [ ] 更多可视化选项

### 版本 2.0 (长期规划)
- [ ] 深度学习模块
- [ ] 实时数据接口
- [ ] 云端部署支持
- [ ] API服务接口

## 贡献指南

### 如何贡献
1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范
- 使用 Python PEP 8 编码规范
- 添加适当的类型提示
- 编写单元测试
- 更新相关文档

### 报告问题
如果发现bug或有功能建议，请在 [Issues](https://github.com/your-username/similar_K/issues) 中创建新的问题。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 联系方式

- 项目主页: https://github.com/your-username/similar_K
- 问题反馈: https://github.com/your-username/similar_K/issues
- 邮箱: your-email@example.com

## 致谢

感谢以下开源项目和研究为本项目提供的灵感和支持：
- pandas: 数据处理的强大工具
- numpy: 高性能数值计算
- scikit-learn: 机器学习算法库
- matplotlib: 数据可视化
- 相关学术论文和研究成果

---

**免责声明**: 本项目仅用于学术研究和教育目的，不构成任何投资建议。使用本项目进行投资决策的风险由用户自行承担。

