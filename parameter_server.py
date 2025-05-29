import os
import socket
import threading
import time

import torch

from config import (
    BATCH_SIZE,
    EXPERIMENT_NAME,
    MAX_STEPS,
    MODEL_TYPE,
    RESULTS_DIR,
    SAVE_INTERVAL,
    SERVER_HOST,
    SERVER_PORT,
)
from model import create_model, model_parameters_to_vector
from utils import (
    PlotlyPlotter,
    analyze_results,
    calculate_throughput,
    log_message,
    recv_data,
    send_data,
)


class ParameterServer:
    def __init__(self, num_workers):
        self.num_workers = num_workers
        self.global_model = create_model(MODEL_TYPE)
        self.global_params = model_parameters_to_vector(self.global_model)
        self.connections = []
        self.worker_params = [None] * num_workers
        self.worker_losses = [None] * num_workers
        self.worker_accuracies = [None] * num_workers
        self.current_step = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.plotter = PlotlyPlotter(
            num_workers, f"{MODEL_TYPE} Training with {num_workers} Workers"
        )
        self.running = True

        # 初始化服务器socket
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((SERVER_HOST, SERVER_PORT))
        self.server_socket.listen(num_workers)
        log_message(
            f"Parameter server started on {SERVER_HOST}:{SERVER_PORT}, waiting for {num_workers} workers..."
        )

    def start(self):
        # 接受工作节点连接
        for i in range(self.num_workers):
            conn, addr = self.server_socket.accept()
            self.connections.append(conn)
            log_message(f"Worker {i} connected from {addr}")
            threading.Thread(target=self.handle_worker, args=(conn, i)).start()

        # 发送初始全局参数
        self.broadcast_params()

        # 主循环
        try:
            while self.current_step < MAX_STEPS and self.running:
                time.sleep(0.1)  # 避免忙等待
        except KeyboardInterrupt:
            log_message("Server interrupted, shutting down...")
            self.running = False

        # 训练结束
        total_time = time.time() - self.start_time
        log_message(f"Training completed in {total_time:.2f} seconds")

        # 保存最终模型
        self.save_model(f"final_model_step_{self.current_step}.pth")

        # 保存图表
        self.plotter.save_plot(f"{EXPERIMENT_NAME}_progress.png")

        # 分析结果
        results = {
            "num_workers": self.num_workers,
            "total_steps": self.current_step,
            "total_time": total_time,
            "throughput": calculate_throughput(
                self.start_time,
                time.time(),
                self.current_step * BATCH_SIZE * self.num_workers,
            ),
            "worker_losses": self.plotter.losses,
            "worker_accuracies": self.plotter.accuracies,
            "avg_losses": self.plotter.avg_losses,
            "avg_accuracies": self.plotter.avg_accuracies,
            "final_loss": self.plotter.avg_losses[-1] if self.plotter.avg_losses else 0,
            "final_accuracy": self.plotter.avg_accuracies[-1]
            if self.plotter.avg_accuracies
            else 0,
            "convergence_step": self.find_convergence_step(),
        }

        report = analyze_results(results)
        log_message("Experiment results:\n" + report)

        # 关闭所有连接
        for conn in self.connections:
            conn.close()
        self.server_socket.close()

    def find_convergence_step(self):
        """找到收敛步骤（当准确率停止显著提高时）"""
        if len(self.plotter.avg_accuracies) < 20:
            return len(self.plotter.avg_accuracies) - 1

        # 检查最近20步的准确率变化
        last_acc = self.plotter.avg_accuracies[-20:]
        max_acc = max(last_acc)
        if max_acc - min(last_acc) < 0.005:  # 变化小于0.5%
            return len(self.plotter.avg_accuracies) - 20

        return len(self.plotter.avg_accuracies) - 1

    def handle_worker(self, conn, worker_id):
        """处理工作节点的通信"""
        try:
            while self.running and self.current_step < MAX_STEPS:
                # 接收工作节点更新的参数
                data = recv_data(conn)
                if data is None:
                    break

                with self.lock:
                    self.worker_params[worker_id] = data["params"]
                    self.worker_losses[worker_id] = data["loss"]
                    self.worker_accuracies[worker_id] = data["accuracy"]

                    # 检查是否所有工作节点都已完成当前步骤
                    if (
                        all(p is not None for p in self.worker_params)
                        and all(l is not None for l in self.worker_losses)
                        and all(a is not None for a in self.worker_accuracies)
                    ):
                        # 聚合参数（平均）
                        avg_params = torch.mean(torch.stack(self.worker_params), dim=0)
                        self.global_params = avg_params

                        # 更新图表
                        self.plotter.add_data(
                            self.current_step,
                            self.worker_losses,
                            self.worker_accuracies,
                        )

                        # 广播新参数
                        self.broadcast_params()

                        # 重置状态
                        self.worker_params = [None] * self.num_workers
                        self.worker_losses = [None] * self.num_workers
                        self.worker_accuracies = [None] * self.num_workers

                        # 保存模型
                        if (self.current_step + 1) % SAVE_INTERVAL == 0:
                            self.save_model(f"model_step_{self.current_step + 1}.pth")

                        self.current_step += 1
                        log_message(f"Completed step {self.current_step}/{MAX_STEPS}")
        except (ConnectionResetError, BrokenPipeError):
            log_message(f"Worker {worker_id} disconnected unexpectedly")
        finally:
            conn.close()

    def broadcast_params(self):
        """广播全局参数到所有工作节点"""
        for conn in self.connections:
            send_data(conn, {"step": self.current_step, "params": self.global_params})

    def save_model(self, filename):
        """保存模型到文件"""
        model_path = os.path.join(RESULTS_DIR, "models", filename)
        torch.save(
            {
                "step": self.current_step,
                "model_state_dict": self.global_model.state_dict(),
                "params": self.global_params,
            },
            model_path,
        )
        log_message(f"Model saved to {model_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python parameter_server.py <num_workers>")
        sys.exit(1)

    num_workers = int(sys.argv[1])
    server = ParameterServer(num_workers)
    server.start()
