import socket

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import (
    BATCH_SIZE,
    DATASET_NAME,
    LEARNING_RATE,
    MODEL_TYPE,
    MOMENTUM,
    OPTIMIZER,
    SERVER_HOST,
    SERVER_PORT,
)
from model import create_model, model_parameters_to_vector, vector_to_model_parameters
from utils import log_message, recv_data, send_data


class Worker:
    def __init__(self, worker_id, num_workers):
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model(MODEL_TYPE).to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # 设置优化器
        if OPTIMIZER == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM
            )
        elif OPTIMIZER == "Adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        else:
            raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

        # 加载数据集
        self.train_loader, self.test_loader = self.load_data()

        # 连接服务器
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((SERVER_HOST, SERVER_PORT))
        log_message(f"Worker {worker_id} connected to server", worker_id)

    def load_data(self):
        """加载并划分数据集"""
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                if DATASET_NAME == "MNIST"
                else transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        if DATASET_NAME == "MNIST":
            full_dataset = datasets.MNIST(
                "./data", train=True, download=True, transform=transform
            )
        elif DATASET_NAME == "CIFAR10":
            full_dataset = datasets.CIFAR10(
                "./data", train=True, download=True, transform=transform
            )
        else:
            raise ValueError(f"Unsupported dataset: {DATASET_NAME}")

        # 划分数据集给各个工作节点
        total_size = len(full_dataset)
        worker_size = total_size // self.num_workers
        start_idx = self.worker_id * worker_size
        end_idx = (
            (self.worker_id + 1) * worker_size
            if self.worker_id < self.num_workers - 1
            else total_size
        )

        indices = list(range(start_idx, end_idx))
        worker_dataset = Subset(full_dataset, indices)

        # 创建数据加载器
        train_loader = DataLoader(worker_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 测试集（所有工作节点使用相同的测试集）
        test_dataset = (
            datasets.MNIST("./data", train=False, transform=transform)
            if DATASET_NAME == "MNIST"
            else datasets.CIFAR10("./data", train=False, transform=transform)
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        log_message(
            f"Loaded {len(worker_dataset)} samples for worker {self.worker_id}",
            self.worker_id,
        )
        return train_loader, test_loader

    def train(self):
        """训练循环"""
        data_iter = iter(self.train_loader)

        while True:
            try:
                # 获取下一批数据
                images, labels = next(data_iter)
            except StopIteration:
                # 重新开始迭代
                data_iter = iter(self.train_loader)
                images, labels = next(data_iter)

            # 移动数据到设备
            images, labels = images.to(self.device), labels.to(self.device)

            # 前向传播
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # 计算准确率
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)

            # 反向传播和优化
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 发送更新后的参数到服务器
            send_data(
                self.sock,
                {
                    "params": model_parameters_to_vector(self.model),
                    "loss": loss.item(),
                    "accuracy": accuracy,
                },
            )

            # 接收全局参数
            global_data = recv_data(self.sock)
            if global_data is None:
                break

            # 更新本地模型
            params_tensor = global_data["params"].to(self.device)
            vector_to_model_parameters(params_tensor, self.model)

            log_message(
                f"Step {global_data['step']} completed, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}",
                self.worker_id,
            )

    def run(self):
        try:
            self.train()
        finally:
            self.sock.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python worker.py <worker_id> <num_workers>")
        sys.exit(1)

    worker_id = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    worker = Worker(worker_id, num_workers)
    worker.run()
