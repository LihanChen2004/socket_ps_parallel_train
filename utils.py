import os
import pickle
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio
from matplotlib.animation import FuncAnimation
from plotly.subplots import make_subplots

from config import DATASET_NAME, EXPERIMENT_NAME, MODEL_TYPE, OPTIMIZER, RESULTS_DIR


def send_data(sock, data):
    """发送数据到socket"""
    data_pkl = pickle.dumps(data)
    data_len = struct.pack(">I", len(data_pkl))
    sock.sendall(data_len + data_pkl)


def recv_data(sock):
    """从socket接收数据"""
    data_len = recv_all(sock, 4)
    if not data_len:
        return None
    data_len = struct.unpack(">I", data_len)[0]
    return pickle.loads(recv_all(sock, data_len))


def recv_all(sock, n):
    """接收指定长度的数据"""
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data


def log_message(message, worker_id=None):
    """记录日志信息"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    prefix = "[Server] " if worker_id is None else f"[Worker {worker_id}] "
    log_entry = f"{timestamp} {prefix}{message}"
    print(log_entry)

    # 保存到日志文件
    log_file = os.path.join(RESULTS_DIR, "logs", f"{EXPERIMENT_NAME}.log")
    with open(log_file, "a") as f:
        f.write(log_entry + "\n")


class RealTimePlotter:
    """实时训练过程可视化"""

    def __init__(self, num_workers, title="Training Progress"):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle(title)

        # 损失曲线
        self.loss_lines = []
        for i in range(num_workers):
            (line,) = self.ax1.plot([], [], label=f"Worker {i} Loss")
            self.loss_lines.append(line)
        (self.avg_loss_line,) = self.ax1.plot(
            [], [], "k-", linewidth=2, label="Avg Loss"
        )

        # 准确率曲线
        self.acc_lines = []
        for i in range(num_workers):
            (line,) = self.ax2.plot([], [], label=f"Worker {i} Acc")
            self.acc_lines.append(line)
        (self.avg_acc_line,) = self.ax2.plot([], [], "k-", linewidth=2, label="Avg Acc")

        # 设置坐标轴
        self.ax1.set_xlabel("Step")
        self.ax1.set_ylabel("Loss")
        self.ax1.set_title("Training Loss")
        self.ax1.legend()
        self.ax1.grid(True)

        self.ax2.set_xlabel("Step")
        self.ax2.set_ylabel("Accuracy")
        self.ax2.set_title("Training Accuracy")
        self.ax2.legend()
        self.ax2.grid(True)

        # 数据存储
        self.steps = []
        self.losses = [[] for _ in range(num_workers)]
        self.accuracies = [[] for _ in range(num_workers)]
        self.avg_losses = []
        self.avg_accuracies = []

        self.animation = FuncAnimation(self.fig, self.update_plot, interval=1000)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)

    def update_plot(self, frame):
        """更新图表"""
        for i, line in enumerate(self.loss_lines):
            line.set_data(self.steps, self.losses[i])

        self.avg_loss_line.set_data(self.steps, self.avg_losses)

        for i, line in enumerate(self.acc_lines):
            line.set_data(self.steps, self.accuracies[i])

        self.avg_acc_line.set_data(self.steps, self.avg_accuracies)

        # 调整坐标轴范围
        if self.steps:
            self.ax1.set_xlim(0, self.steps[-1] + 1)
            self.ax2.set_xlim(0, self.steps[-1] + 1)

            min_loss = min(min(loss) for loss in self.losses if loss) or 0
            max_loss = max(max(loss) for loss in self.losses if loss) or 1
            self.ax1.set_ylim(min_loss * 0.9, max_loss * 1.1)

            min_acc = min(min(acc) for acc in self.accuracies if acc) or 0
            max_acc = max(max(acc) for acc in self.accuracies if acc) or 1
            self.ax2.set_ylim(min_acc * 0.9, min(max_acc * 1.1, 1.0))

        return (
            self.loss_lines
            + [self.avg_loss_line]
            + self.acc_lines
            + [self.avg_acc_line]
        )

    def add_data(self, step, worker_losses, worker_accuracies):
        """添加新数据点"""
        self.steps.append(step)

        for i in range(len(worker_losses)):
            self.losses[i].append(worker_losses[i])
            self.accuracies[i].append(worker_accuracies[i])

        self.avg_losses.append(np.mean(worker_losses))
        self.avg_accuracies.append(np.mean(worker_accuracies))

    def save_plot(self, filename):
        """保存图表到文件"""
        self.fig.savefig(os.path.join(RESULTS_DIR, "figures", filename))
        plt.close(self.fig)


def calculate_throughput(start_time, end_time, num_samples):
    """计算吞吐量（样本/秒）"""
    elapsed_time = end_time - start_time
    return num_samples / elapsed_time if elapsed_time > 0 else 0


def analyze_results(results):
    """分析实验结果并生成报告"""
    report = "# Parallel Training Experiment Report\n\n"
    report += "## Experiment Summary\n"
    report += f"- **Experiment Name**: {EXPERIMENT_NAME}\n"
    report += f"- **Dataset**: {DATASET_NAME}\n"
    report += f"- **Model**: {MODEL_TYPE}\n"
    report += f"- **Optimizer**: {OPTIMIZER}\n"
    report += f"- **Number of Workers**: {results['num_workers']}\n"
    report += f"- **Total Steps**: {results['total_steps']}\n"
    report += f"- **Final Accuracy**: {results['final_accuracy']:.4f}\n"
    report += f"- **Total Time**: {results['total_time']:.2f} seconds\n"
    report += f"- **Throughput**: {results['throughput']:.2f} samples/sec\n\n"

    report += "## Performance Metrics\n"
    report += "| Metric | Value |\n"
    report += "|--------|-------|\n"
    report += f"| Training Time | {results['total_time']:.2f} sec |\n"
    report += f"| Throughput | {results['throughput']:.2f} samples/sec |\n"
    report += f"| Convergence Steps | {results['convergence_step']} |\n"
    report += f"| Final Loss | {results['final_loss']:.4f} |\n"
    report += f"| Final Accuracy | {results['final_accuracy']:.4f} |\n\n"

    report += "## Loss and Accuracy Curves\n"
    report += f"![Training Progress](figures/{EXPERIMENT_NAME}_progress.png)\n\n"

    report += "## Detailed Results\n"
    report += "### Loss per Worker\n"
    for i, loss in enumerate(results["worker_losses"]):
        report += f"- Worker {i}: Final loss = {loss[-1]:.4f}\n"

    report += "\n### Accuracy per Worker\n"
    for i, acc in enumerate(results["worker_accuracies"]):
        report += f"- Worker {i}: Final accuracy = {acc[-1]:.4f}\n"

    # 保存报告
    report_file = os.path.join(RESULTS_DIR, "logs", f"{EXPERIMENT_NAME}_report.md")
    with open(report_file, "w") as f:
        f.write(report)

    return report


class PlotlyPlotter:
    """基于Plotly的训练过程可视化（静态，实验结束后生成交互式HTML）"""

    def __init__(self, num_workers, title="Training Progress"):
        self.num_workers = num_workers
        self.title = title
        self.steps = []
        self.losses = [[] for _ in range(num_workers)]
        self.accuracies = [[] for _ in range(num_workers)]
        self.avg_losses = []
        self.avg_accuracies = []

    def add_data(self, step, worker_losses, worker_accuracies):
        self.steps.append(step)
        for i in range(self.num_workers):
            self.losses[i].append(worker_losses[i])
            self.accuracies[i].append(worker_accuracies[i])
        self.avg_losses.append(np.mean(worker_losses))
        self.avg_accuracies.append(np.mean(worker_accuracies))

    def save_plot(self, filename):
        """保存交互式曲线到HTML文件"""
        fig = make_subplots(rows=2, cols=1, subplot_titles=("Loss", "Accuracy"))
        # 损失曲线
        for i in range(self.num_workers):
            fig.add_trace(
                go.Scatter(
                    x=self.steps,
                    y=self.losses[i],
                    mode="lines",
                    name=f"Worker {i} Loss",
                ),
                row=1,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=self.steps,
                y=self.avg_losses,
                mode="lines",
                name="Avg Loss",
                line=dict(width=3, color="black"),
            ),
            row=1,
            col=1,
        )
        # 准确率曲线
        for i in range(self.num_workers):
            fig.add_trace(
                go.Scatter(
                    x=self.steps,
                    y=self.accuracies[i],
                    mode="lines",
                    name=f"Worker {i} Acc",
                ),
                row=2,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=self.steps,
                y=self.avg_accuracies,
                mode="lines",
                name="Avg Acc",
                line=dict(width=3, color="black"),
            ),
            row=2,
            col=1,
        )
        fig.update_layout(title_text=self.title, height=800)
        fig.update_xaxes(title_text="Step", row=1, col=1)
        fig.update_xaxes(title_text="Step", row=2, col=1)
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        # 保存为HTML
        save_path = os.path.join(
            RESULTS_DIR, "figures", filename.replace(".png", ".html")
        )
        pio.write_html(fig, file=save_path, auto_open=False)
