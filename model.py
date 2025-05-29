import torch
import torch.nn as nn
import torch.nn.functional as F

from config import MODEL_CONFIG


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        config = MODEL_CONFIG["MLP"]
        self.fc1 = nn.Linear(config["input_dim"], config["hidden_dim"])
        self.fc2 = nn.Linear(config["hidden_dim"], config["output_dim"])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        config = MODEL_CONFIG["CNN"]
        self.conv1 = nn.Conv2d(config["input_channels"], config["conv1_out"], 3, 1)
        self.conv2 = nn.Conv2d(config["conv1_out"], config["conv2_out"], 3, 1)
        self.fc1 = nn.Linear(5 * 5 * config["conv2_out"], config["fc1_out"])
        self.fc2 = nn.Linear(config["fc1_out"], config["output_dim"])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5 * 5 * MODEL_CONFIG["CNN"]["conv2_out"])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_model(model_type="MLP"):
    if model_type == "MLP":
        return MLP()
    elif model_type == "CNN":
        return CNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def model_parameters_to_vector(model):
    """将模型参数转换为向量"""
    return torch.cat([param.view(-1) for param in model.parameters()])


def vector_to_model_parameters(vector, model):
    """将向量转换为模型参数"""
    pointer = 0
    for param in model.parameters():
        num_param = param.numel()
        param.data = vector[pointer : pointer + num_param].view(param.size())
        pointer += num_param
