import multiprocessing
import subprocess
import sys
import time


def run_parameter_server(num_workers):
    """运行参数服务器"""
    subprocess.run(["python", "parameter_server.py", str(num_workers)])


def run_worker(worker_id, num_workers):
    """运行工作节点"""
    subprocess.run(["python", "worker.py", str(worker_id), str(num_workers)])


def run_experiment(num_workers):
    """运行完整实验"""
    # 启动参数服务器
    server_process = multiprocessing.Process(
        target=run_parameter_server, args=(num_workers,)
    )
    server_process.start()

    # 等待服务器启动
    time.sleep(2)

    # 启动工作节点
    worker_processes = []
    for worker_id in range(num_workers):
        p = multiprocessing.Process(target=run_worker, args=(worker_id, num_workers))
        p.start()
        worker_processes.append(p)

    # 等待所有工作节点完成
    for p in worker_processes:
        p.join()

    # 终止服务器进程
    server_process.terminate()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_experiment.py <num_workers>")
        sys.exit(1)

    num_workers = int(sys.argv[1])
    print(f"Starting experiment with {num_workers} workers...")
    run_experiment(num_workers)
    print("Experiment completed!")
