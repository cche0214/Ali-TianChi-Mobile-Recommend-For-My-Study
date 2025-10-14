# -*- coding: utf-8 -*-

import os
import sys
import platform
import multiprocessing
import subprocess
import time
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def detect_dsw_environment():
    # 环境判定（汇总输出）
    dsw_indicators = []

    if 'JUPYTER_SERVER_ROOT' in os.environ:
        dsw_indicators.append("jupyter")
    if 'KUBERNETES_SERVICE_HOST' in os.environ:
        dsw_indicators.append("k8s")

    python_path = sys.executable
    if 'conda' in python_path.lower() or 'miniconda' in python_path.lower():
        dsw_indicators.append("conda")

    hostname = platform.node()
    if any(k in hostname.lower() for k in ['dsw', 'pai', 'aliyun']):
        dsw_indicators.append("host")

    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            dsw_indicators.append("gpu")
    except Exception:
        pass

    is_dsw = len(dsw_indicators) >= 2
    print(f"环境判定: {'DSW' if is_dsw else '非DSW/未知'} (命中指标{len(dsw_indicators)})")
    return is_dsw


def get_cpu_info():
    # CPU信息（汇总输出）
    cpu_count = multiprocessing.cpu_count()
    if cpu_count >= 48:
        grade = "顶配"
    elif cpu_count >= 32:
        grade = "高配"
    elif cpu_count >= 16:
        grade = "中配"
    else:
        grade = "偏低"

    physical_cores = psutil.cpu_count(logical=False) if HAS_PSUTIL else None
    cpu_freq = psutil.cpu_freq() if HAS_PSUTIL else None

    t0 = time.time()
    _ = [i * i for i in range(100000)]
    test_time = time.time() - t0
    if test_time < 0.1:
        perf = "优秀"
    elif test_time < 0.5:
        perf = "良好"
    else:
        perf = "偏低"

    parts = [f"CPU: {cpu_count}核", f"评级:{grade}", f"测试:{test_time:.3f}s({perf})"]
    if physical_cores:
        parts.insert(1, f"物理{physical_cores}核")
    if cpu_freq and cpu_freq.max:
        parts.insert(2, f"频率上限{cpu_freq.max:.0f}MHz")
    print("，".join(parts))
    return cpu_count


def get_memory_info():
    # 内存信息（汇总输出）
    if not HAS_PSUTIL:
        print("内存: 未安装psutil，无法检测")
        return None

    mem = psutil.virtual_memory()
    memory_gb = mem.total / (1024 ** 3)

    if memory_gb >= 350:
        grade = "超大"
    elif memory_gb >= 200:
        grade = "大"
    elif memory_gb >= 100:
        grade = "中"
    else:
        grade = "偏低"

    bandwidth_str = "N/A"
    try:
        import numpy as np
        start = time.time()
        test_size = min(100000000, int(mem.available * 0.1 / 8))
        arr = np.random.random(test_size)
        _ = float(np.sum(arr))
        dt = time.time() - start
        gb = test_size * 8 / (1024 ** 3)
        bw = gb / dt if dt > 0 else 0.0
        if bw > 50:
            bw_grade = "优秀"
        elif bw > 20:
            bw_grade = "良好"
        else:
            bw_grade = "偏低"
        bandwidth_str = f"{bw:.1f}GB/s({bw_grade})"
        del arr
    except Exception:
        pass

    print(
        f"内存: {memory_gb:.1f}GB，可用{mem.available / (1024 ** 3):.1f}GB，"
        f"评级:{grade}，带宽:{bandwidth_str}"
    )
    return memory_gb


def get_disk_info():
    # 磁盘信息（汇总输出）
    if not HAS_PSUTIL:
        print("磁盘: 未安装psutil，无法检测")
        return None

    free_gb = None
    try:
        usage = psutil.disk_usage(".")
        free_gb = usage.free / (1024 ** 3)
        if free_gb >= 500:
            grade = "充足"
        elif free_gb >= 200:
            grade = "良好"
        elif free_gb >= 100:
            grade = "一般"
        else:
            grade = "不足"
    except Exception:
        print("磁盘: 获取使用情况失败")
        return None

    try:
        test_file = "dsw_io_test.tmp"
        size_mb = 100
        block = b"0" * (1024 * 1024)

        t0 = time.time()
        with open(test_file, "wb") as f:
            for _ in range(size_mb):
                f.write(block)
        write_time = time.time() - t0

        t1 = time.time()
        with open(test_file, "rb") as f:
            _ = f.read()
        read_time = time.time() - t1

        os.remove(test_file)

        write_speed = size_mb / write_time if write_time > 0 else 0.0
        read_speed = size_mb / read_time if read_time > 0 else 0.0
        if write_speed > 100 and read_speed > 100:
            io_grade = "优秀"
        elif write_speed > 50 and read_speed > 50:
            io_grade = "良好"
        else:
            io_grade = "偏低"

        print(
            f"磁盘: 可用{free_gb:.0f}GB，评级:{grade}，I/O 写{write_speed:.0f}MB/s 读{read_speed:.0f}MB/s({io_grade})"
        )
    except Exception:
        print(f"磁盘: 可用{free_gb:.0f}GB，评级:{grade}，I/O测试失败")

    return free_gb


def get_python_info():
    # Python与依赖（汇总输出）
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    required = ["pandas", "numpy", "sklearn", "psutil", "matplotlib", "pickle", "multiprocessing"]
    missing = []
    for m in required:
        try:
            __import__(m)
        except Exception:
            missing.append(m)
    if missing:
        print(f"Python: {py_ver}，依赖缺失: {', '.join(missing)}")
    else:
        print(f"Python: {py_ver}，依赖完整")


def test_dsw_optimizations():
    # 并行与数据聚合快测（汇总输出）
    cpu_cores = multiprocessing.cpu_count()
    os.environ['TIANCHI_MEMORY_MODE'] = 'normal'
    os.environ['TIANCHI_CPU_CORES'] = str(min(40, cpu_cores))
    os.environ['TIANCHI_CHUNK_SIZE'] = '500000'

    n_jobs = min(30, cpu_cores)
    os.environ['OMP_NUM_THREADS'] = str(n_jobs)
    os.environ['MKL_NUM_THREADS'] = str(n_jobs)
    os.environ['OPENBLAS_NUM_THREADS'] = str(n_jobs)
    print(f"并行线程: OMP/MKL/OPENBLAS={n_jobs}")

    try:
        import pandas as pd
        import numpy as np
        t0 = time.time()
        df = pd.DataFrame({
            'user_id': np.random.randint(1, 100000, 1000000),
            'item_id': np.random.randint(1, 50000, 1000000),
            'value': np.random.random(1000000),
        })
        _ = df.groupby('user_id')['value'].agg(['count', 'sum', 'mean'])
        t = time.time() - t0
        level = "优秀" if t < 1.0 else ("良好" if t < 3.0 else "需优化")
        print(f"数据聚合测试: 100万行 {t:.3f}s({level})")
        del df
    except Exception as e:
        print(f"数据聚合测试: 失败({e.__class__.__name__})")

    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        memory_gb = mem.total / (1024 ** 3)
        if memory_gb >= 300:
            rec_chunk = 800000
        elif memory_gb >= 200:
            rec_chunk = 500000
        elif memory_gb >= 100:
            rec_chunk = 300000
        else:
            rec_chunk = 100000
        est_mem = rec_chunk * 30 * 8 / (1024 ** 3)
        safe = "安全" if est_mem < memory_gb * 0.1 else "偏大"
        print(f"推荐chunk_size: {rec_chunk:,}，单chunk约{est_mem:.2f}GB({safe})")


def calculate_dsw_performance_recommendations(cpu_count, memory_gb, free_disk_gb, is_dsw):
    # 综合建议（汇总输出）
    if cpu_count >= 48:
        rec_jobs = 35
    elif cpu_count >= 32:
        rec_jobs = 25
    elif cpu_count >= 16:
        rec_jobs = 14
    else:
        rec_jobs = max(1, cpu_count // 2)

    if memory_gb is None:
        rec_chunk = 300000
    elif memory_gb >= 300:
        rec_chunk = 800000
    elif memory_gb >= 200:
        rec_chunk = 500000
    elif memory_gb >= 100:
        rec_chunk = 300000
    else:
        rec_chunk = 100000

    if cpu_count >= 32 and (memory_gb or 0) >= 200:
        flow = "高配流程"
    elif cpu_count >= 16 and (memory_gb or 0) >= 100:
        flow = "中配流程"
    else:
        flow = "简化流程"

    env_tag = "DSW" if is_dsw else "通用"
    print(f"建议({env_tag}): n_jobs={rec_jobs}，chunk_size={rec_chunk:,}，流程:{flow}")
    print(
        f"环境变量: TIANCHI_CPU_CORES={min(40, cpu_count)}，"
        f"TIANCHI_CHUNK_SIZE={rec_chunk}，OMP_NUM_THREADS={min(30, cpu_count)}"
    )


def main():
    print("DSW环境检测（精简输出）")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        is_dsw = detect_dsw_environment()
        cpu_count = get_cpu_info()
        memory_gb = get_memory_info()
        free_disk_gb = get_disk_info()
        get_python_info()
        test_dsw_optimizations()
        calculate_dsw_performance_recommendations(cpu_count, memory_gb, free_disk_gb, is_dsw)

        if is_dsw and cpu_count >= 32 and (memory_gb or 0) >= 200:
            conclusion = "结论: 配置强大，支持全量运行"
        elif cpu_count >= 16 and (memory_gb or 0) >= 100:
            conclusion = "结论: 配置良好，建议分阶段运行"
        else:
            conclusion = "结论: 配置一般，建议采样与降规模"
        print(conclusion)
        print("检测完成")
    except KeyboardInterrupt:
        print("检测中断")
    except Exception as e:
        print(f"错误: {e.__class__.__name__}: {e}")


if __name__ == "__main__":
    main()