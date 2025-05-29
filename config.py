# 通用配置
import os

# 实验配置
EXPERIMENT_NAME = "parallel_training_experiment"
MAX_STEPS = 500
SAVE_INTERVAL = 50

# 网络配置
SERVER_HOST = "localhost"
SERVER_PORT = 12345
BUFFER_SIZE = 4096

# 数据集配置
DATASET_NAME = "MNIST"
BATCH_SIZE = 512

# 模型配置
MODEL_TYPE = "CNN"  # 可选: MLP, CNN
MODEL_CONFIG = {
    "MLP": {"input_dim": 784, "hidden_dim": 128, "output_dim": 10},
    "CNN": {
        "input_channels": 1,
        "conv1_out": 16,
        "conv2_out": 32,
        "fc1_out": 128,
        "output_dim": 10,
    },
}

# 优化器配置
OPTIMIZER = "Adam"  # 可选: SGD, Adam
LEARNING_RATE = 0.005
MOMENTUM = 0.9

# 日志和结果目录
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "models"), exist_ok=True)
