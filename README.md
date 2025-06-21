# 相似K线寻找

## 项目概述

这个项目专注于从A股市场历史数据中通过**Ranked KLine（排序K线）算法**来寻找相似的K线模式，并统计不同排序组合的出现频率及其对应的未来收益率。通过对时间窗口内开高收低价格的排序分析，识别具有统计显著性的K线模式。

**核心特性**：
- ⚡ **并行处理**：支持多核CPU并行计算，显著提升分析速度
- 🗜️ **模式分组**：智能合并相似模式，大幅减少输出文件大小（90%+压缩率）
- 📊 **大规模分析**：支持处理大窗口长度和批量股票数据
- 🎯 **高精度算法**：修复的相似性算法确保分组结果准确可靠

## 项目目标

- 识别具有预测价值的K线模式
- 统计不同模式对应的未来收益率分布
- 提供可配置的时间窗口和预测周期
- 支持大规模数据批量处理
- 提供高性能并行计算解决方案
- 实现智能模式压缩，处理超大数据集


## 处理模式

本项目支持两种数据处理模式：

### 模式一：批量处理（全市场分析）
- **功能**: 读取指定文件夹中所有股票的历史数据
- **用途**: 进行全市场K线模式的统计分析
- **输出**: 生成包含所有股票数据的综合统计报告
- **适用场景**: 寻找市场普遍存在的K线模式规律

**特点**:
- 处理大量股票数据，样本容量大
- 统计结果更具普遍性和稳定性
- 能发现跨股票的共同模式
- 计算资源需求较高

### 模式二：单股分析（个股专项分析）
- **功能**: 针对指定的单只股票进行深度分析
- **用途**: 分析特定股票的个性化K线模式
- **输出**: 生成该股票专属的模式统计报告
- **适用场景**: 对特定股票进行精准的量化分析

**特点**:
- 专注单只股票，分析更精细
- 能发现个股特有的模式特征
- 计算速度快，资源消耗少
- 适合实时交易决策支持

## 技术栈

- **Python 3.8+**: 核心开发语言
- **pandas>=1.5.0**: 数据处理和分析
- **numpy>=1.21.0**: 数值计算和排序算法
- **scipy>=1.9.0**: 统计计算和相似性分析
- **multiprocessing**: 并行处理框架
- **matplotlib>=3.5.0**: 可选的数据可视化支持

## 安装和使用

### 安装步骤

1. 进入项目目录
```bash
cd similar_K
```

2. 创建虚拟环境（可选）
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
将K线数据文件放入 `data_example/` 目录：
- 每个CSV文件包含一只股票的完整历史数据
- 支持中文列名格式（股票代码、股票简称、交易日期、开盘价、最高价、最低价、收盘价等）
- 文件命名格式：`{股票代码}.csv`

#### 2. 单股开盘价分析（推荐入门）
```bash
# 分析单只股票开盘价的5日窗口排序模式
python main.py --mode single --stock_file data_example/sh600082.csv --price_type open -w 5
```

#### 3. 批量分析（市场级别）
```bash
# 分析整个文件夹中所有股票的收盘价模式
python main.py --mode batch --data_dir data_example/ --price_type close -w 5
```

#### 4. 查看结果
结果将保存在 `output/` 目录下：
- **单股分析**: `output/single_stock/{股票代码}_{价格类型}_w{窗口}.csv`
- **批量分析**: `output/market/market_{价格类型}_w{窗口}.csv`

## API文档

### 命令行参数

#### 模式一：批量处理（全市场分析）
```bash
python main.py --mode batch --data_dir [数据文件夹路径] [其他参数]
```

#### 模式二：单股分析（个股专项分析）
```bash
python main.py --mode single --stock_file [单个股票文件路径] [其他参数]
```

#### 核心参数

- `--mode`: 处理模式 (batch/single, 必选)
- `--data_dir`: 数据文件夹路径 (batch模式必选)
- `--stock_file`: 单个股票文件路径 (single模式必选)
- `--price_type, -p`: 价格类型 (open/high/low/close, 必选)
- `--window, -w`: 时间窗口长度 (必选)
- `--output_dir, -o`: 输出目录 (默认: output/)
- `--min_frequency`: 最小模式频率阈值 (默认: 10)
- `--verbose, -v`: 详细输出模式

#### 性能优化参数

- `--n_jobs, -j`: 并行处理数量 (默认: 1, -1表示使用所有CPU核心)
- `--enable_grouping`: 启用模式分组功能，减少输出文件大小
- `--correlation_threshold`: 分组时的相似性阈值 (默认: 0.9, 范围: 0.0-1.0)


## 使用示例

### 示例1：单股开盘价分析

```bash
# 分析海泰发展开盘价的5日窗口排序模式
python main.py --mode single --stock_file data_example/sh600082.csv --price_type open -w 5
```

预期输出：
```
=== 单股K线排序分析 ===
股票: sh600082 (N/A)
价格类型: 开盘价
时间窗口: 5
数据范围: 1997-06-20 至 2025-06-20

正在计算开盘价排序模式...
找到排序模式: 120 种可能
实际出现模式: 0 种
高频模式 (≥10次): 0 种

生成分析表格...
保存结果: output/single_stock/sh600082_open_w5.csv
分析完成！
```

### 示例2：单股收盘价分析

```bash
# 分析收盘价的5日窗口
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 5
```

将生成文件：`output/single_stock/sh600082_close_w5.csv`

### 示例3：批量处理（全市场分析）

```bash
# 批量分析所有股票的最高价排序模式
python main.py --mode batch --data_dir data_example/ --price_type high -w 5
```

将生成文件：`output/market/market_high_w5.csv`

### 示例4：并行处理加速

```bash
# 使用4个CPU核心并行处理
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 10 -j 4

# 使用所有CPU核心（推荐）
python main.py --mode batch --data_dir data_example/ --price_type close -w 8 -j -1
```

### 示例5：模式分组减少文件大小

```bash
# 启用模式分组（推荐窗口长度>=8时使用）
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 10 --enable_grouping

# 自定义相似性阈值（更低阈值=更多压缩）
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 12 --enable_grouping --correlation_threshold 0.8
```

### 示例6：最佳性能配置

```bash
# 综合使用并行处理和模式分组
python main.py --mode batch --data_dir data_example/ --price_type close -w 12 -j -1 --enable_grouping --correlation_threshold 0.9
```

### 示例7：调整参数

```bash
# 降低最小频率阈值，分析更大窗口
python main.py --mode single --stock_file data_example/sh600082.csv --price_type open -w 10 --min_frequency 5
```

## 数据格式说明

### 输入数据格式

#### CSV格式 (中文列名)
本项目支持包含中文列名的A股数据CSV文件格式：

```csv
股票代码,股票简称,交易日期,开盘价,最高价,最低价,收盘价,前收盘价,成交量,成交额,流通市值,总市值
sh600082,海泰发展,1997-06-20,10.2,11.09,9.7,10.9,5.18,20366800.0,211444048.0,327000000.0,1249140000.0
sh600082,海泰发展,1997-06-23,10.85,10.9,9.82,10.1,10.9,8777300.0,90422378.0,303000000.0,1157460000.0
sh600082,海泰发展,1997-06-24,10.07,10.26,9.78,9.9,10.1,4050400.0,40319176.0,297000000.0,1134540000.0
...
```

**必需列**:
- `交易日期`: 交易日期 (YYYY-MM-DD 格式)
- `开盘价`: 开盘价
- `最高价`: 最高价
- `最低价`: 最低价
- `收盘价`: 收盘价

**可选列**:
- `股票代码`: 股票代码 (如sh600082)
- `股票简称`: 股票名称 (如海泰发展)
- `成交量`: 成交量 (手)
- `成交额`: 成交金额 (元)
- 其他列将被忽略

**文件要求**:
- 支持多种编码格式 (UTF-8, GBK, GB2312等)
- 自动跳过文件开头的说明行
- 数据按交易日期正序排列
- 文件命名格式：`{股票代码}.csv` (如：sh600082.csv)

### 输出结果格式

#### 分析表格结构（CSV格式）

##### 标准输出格式

每个分析结果表格包含以下12列：

```csv
rank_pattern,return_1d_mean,return_1d_std,return_3d_mean,return_3d_std,return_5d_mean,return_5d_std,return_10d_mean,return_10d_std,return_20d_mean,return_20d_std,frequency
"(1, 2, 3, 4, 5)",-0.0006,0.0389,0.0024,0.0666,0.0018,0.0882,-0.0009,0.1091,0.0035,0.1615,411
"(5, 4, 3, 2, 1)",-0.0006,0.0332,0.0012,0.0549,0.0042,0.0835,0.0048,0.1167,0.0027,0.1650,282
"(1, 2, 3, 5, 4)",0.0021,0.0338,0.0014,0.0695,0.0033,0.0915,0.0068,0.1116,0.0076,0.1762,236
...
```

##### 分组输出格式（启用--enable_grouping时）

```csv
group_id,representative_pattern,pattern_count,return_1d_mean,return_1d_std,return_3d_mean,return_3d_std,return_5d_mean,return_5d_std,return_10d_mean,return_10d_std,return_20d_mean,return_20d_std,frequency
Group_0001,"(1, 2, 3, 4, 5)",5,-0.0006,0.0389,0.0024,0.0666,0.0018,0.0882,-0.0009,0.1091,0.0035,0.1615,2055
Group_0002,"(5, 4, 3, 2, 1)",3,-0.0006,0.0332,0.0012,0.0549,0.0042,0.0835,0.0048,0.1167,0.0027,0.1650,1410
...
```

**标准格式列定义**:
1. `rank_pattern`: 排序模式列表
2. `return_1d_mean`: 未来1天平均收益率
3. `return_1d_std`: 未来1天收益率标准差
4. `return_3d_mean`: 未来3天平均收益率
5. `return_3d_std`: 未来3天收益率标准差
6. `return_5d_mean`: 未来5天平均收益率
7. `return_5d_std`: 未来5天收益率标准差
8. `return_10d_mean`: 未来10天平均收益率
9. `return_10d_std`: 未来10天收益率标准差
10. `return_20d_mean`: 未来20天平均收益率
11. `return_20d_std`: 未来20天收益率标准差
12. `frequency`: 该排序模式出现次数

**分组格式额外列定义**:
- `group_id`: 分组唯一标识符
- `representative_pattern`: 代表该组的模式
- `pattern_count`: 该组包含的模式数量
- 其他列: 组内所有模式的合并统计数据

#### 文件命名规则

**单股分析**:
- `{股票代码}_{价格类型}_w{窗口长度}.csv`
- 示例：`sh600082_open_w10.csv`、`sh600082_close_w20.csv`

**批量分析**:
- `market_{价格类型}_w{窗口长度}.csv`
- 示例：`market_high_w15.csv`、`market_low_w30.csv`

#### 输出文件夹结构
```
output/
├── single_stock/           # 单股分析结果
│   ├── sh600082_open_w10.csv
│   ├── sh600082_open_w20.csv
│   ├── sh600082_high_w10.csv
│   └── ...
└── market/                 # 批量分析结果
    ├── market_open_w10.csv
    ├── market_close_w20.csv
    └── ...
```

## 性能优化与最佳实践

### 并行处理使用指南

#### 🚀 推荐配置
- **小数据集**: `-j 4` (使用4个CPU核心)
- **大数据集**: `-j -1` (使用所有CPU核心)
- **单股分析**: 窗口长度 >= 10 时并行处理效果明显
- **批量分析**: 始终建议使用并行处理

#### ⚡ 性能对比
```bash
# 单线程处理（基准）
time python main.py --mode batch --data_dir data_example/ --price_type close -w 10 -j 1

# 多线程处理（推荐）
time python main.py --mode batch --data_dir data_example/ --price_type close -w 10 -j -1
```

### 模式分组使用指南

#### 🗜️ 何时使用模式分组
- **强烈推荐**: 窗口长度 >= 8
- **必须使用**: 窗口长度 >= 12（否则文件可能过大）
- **批量分析**: 大数据集时建议启用

#### 📊 压缩效果对比
| 窗口长度 | 原始模式数 | 分组后模式数 | 压缩率 |
|---------|-----------|-------------|--------|
| w=5     | 120       | 120         | 0%     |
| w=8     | 40,320    | 4,032       | 90%    |
| w=10    | 3,628,800 | 362,880     | 90%    |
| w=12    | 479,001,600 | 47,900,160 | 90%   |

#### 🎯 相似性阈值选择
- **0.95**: 极其严格，很少分组，文件较大
- **0.9** (默认): 平衡压缩率和准确性
- **0.8**: 较高压缩率，适合超大数据集
- **0.5**: 最大压缩率，可能损失一些细节

### 大数据处理策略

#### 💾 内存优化
```bash
# 对于超大窗口长度，使用高阈值分组
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 15 --enable_grouping --correlation_threshold 0.95

# 批量处理大数据集
python main.py --mode batch --data_dir data_example/ --price_type close -w 10 -j -1 --enable_grouping
```

#### 🔧 最佳性能配置示例
```bash
# 小窗口 (w=3-7): 基础配置
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 5 -j 4

# 中等窗口 (w=8-12): 启用分组
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 10 -j -1 --enable_grouping

# 大窗口 (w>=13): 全功能启用
python main.py --mode single --stock_file data_example/sh600082.csv --price_type close -w 15 -j -1 --enable_grouping --correlation_threshold 0.9
```

### 故障排除

#### 🚨 常见问题
1. **内存不足**: 启用模式分组 `--enable_grouping`
2. **处理速度慢**: 使用并行处理 `-j -1`
3. **文件过大**: 提高相似性阈值 `--correlation_threshold 0.95`
4. **分组结果异常**: 检查阈值设置，确保不会将完全不同的模式分组

#### ⚠️ 注意事项
- 并行处理结果应与单线程完全一致
- 分组功能会改变输出格式，注意列名变化
- 超大窗口长度可能需要大量内存和时间
- 建议先在小数据集上测试参数配置

### 项目结构

```
similar_K/
├── main.py                    # 程序入口
├── requirements.txt           # 依赖库列表
├── README.md                  # 项目说明文档
├── CLAUDE.md                  # Claude Code 指导文档
├── data_example/              # 示例数据
│   └── sh600082.csv
├── output/                    # 输出结果目录
│   ├── single_stock/          # 单股分析结果
│   └── market/                # 批量分析结果
└── src/                       # 源代码目录
    ├── ranked_kline/          # 核心算法模块
    │   ├── __init__.py
    │   └── analyzer.py        # 主要分析逻辑
    └── utils/                 # 工具模块
        ├── __init__.py
        ├── cli_parser.py      # 命令行参数解析
        ├── data_loader.py     # 数据加载器
        ├── output_manager.py  # 输出管理器
        └── pattern_grouper.py # 模式分组器（新增）
```

