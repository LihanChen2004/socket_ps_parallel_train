# socket_ps_parallel_train

基于Socket通信的分布式参数服务器（Parameter Server, PS）架构，支持多Worker节点的数据并行训练。用于 SMBU 并行与分布式计算课程大作业。

## 目录结构

```txt
.
├── config.py                # 配置文件
├── model.py                 # 模型结构与参数转换工具
├── parameter_server.py      # 参数服务器主程序
├── worker.py                # Worker节点主程序
├── utils.py                 # 工具函数与可视化
├── run_experiment.py        # 一键实验脚本
├── results/                 # 日志、模型、图表输出目录
└── README.md
```

## 主要特性

- **分布式通信**：基于Python原生socket实现Worker与参数服务器的数据同步
- **模型同步**：支持参数平均同步，保证多Worker训练一致性
- **多模型支持**：支持MLP、CNN等常见结构，易于扩展
- **性能测试**：自动记录训练速度、吞吐量、收敛步数等指标
- **可视化**：支持matplotlib和Plotly两种训练过程可视化
- **实验自动化**：一键启动完整分布式实验

## 快速开始

### 1. 安装依赖

```bash
pip install torch torchvision matplotlib plotly
```

### 2. 配置参数

编辑 config.py，可自定义数据集、模型类型、优化器、批量大小等参数。

### 3. 运行实验

以4个Worker为例：

```bash
python run_experiment.py 4
```

或手动启动：

```bash
# 启动参数服务器
python parameter_server.py 4

# 分别启动4个Worker（可用不同终端）
python worker.py 0 4
python worker.py 1 4
python worker.py 2 4
python worker.py 3 4
```

### 4. 查看结果

- 日志、模型、图表等输出在 results 目录下
- 训练过程曲线（静态PNG和交互式HTML）在 figures
- 实验报告在 logs

## 文件说明

- **parameter_server.py**：参数服务器，负责聚合参数、同步模型、记录日志和可视化
- **worker.py**：Worker节点，负责本地训练、与服务器通信
- **model.py**：定义MLP/CNN模型结构及参数向量化工具
- **utils.py**：通信、日志、可视化、性能分析等工具函数
- **run_experiment.py**：自动化实验脚本，支持多进程一键启动
- **config.py**：全局配置文件
