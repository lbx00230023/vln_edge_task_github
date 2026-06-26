#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_real_vlfm_one_episode.py

功能：
    test_vlfm_1 的真实 VLFM 单 episode 运行入口。

作用：
    1. 在 VS Code 中一键运行官方 VLFM；
    2. 使用已经筛选好的单场景、单目标、单 episode 小数据集；
    3. 开启 TEST_VLFM_1_ENABLE，使 vlfm_trainer.py 中的 LiveVLFMRecorder 生效；
    4. 将运行日志保存到 results/test_vlfm_1/logs/；
    5. 运行过程中实时显示 RGB / Depth / Obstacle Map / Value Map 面板。

重要说明：
    - 本脚本不实现新的导航策略；
    - 本脚本不训练模型；
    - 本脚本只是调用官方 VLFM 推理 / evaluation 流程；
    - 真正的实时可视化逻辑由 live_vlfm_recorder.py 和 vlfm_trainer.py 补丁完成。
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path("/home/nd/vln_edge_task")
VLFM_ROOT = PROJECT_ROOT / "external/vlfm"
RESULT_DIR = PROJECT_ROOT / "results/test_vlfm_1"

ONE_EPISODE_DATASET = (
    RESULT_DIR
    / "configs/objectnav_mp3d_one_episode/val/val.json.gz"
)


def make_dirs() -> None:
    """
    创建 test_vlfm_1 标准结果目录。

    即使前面已经创建过，这里再次 mkdir 也不会破坏已有结果。
    """
    for subdir in [
        "logs",
        "videos",
        "frames",
        "snapshots",
        "configs",
        "metrics",
    ]:
        (RESULT_DIR / subdir).mkdir(parents=True, exist_ok=True)


def is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """
    检查本地端口是否可连接。

    VLFM 需要四个本地 VLM server：
        12181 GroundingDINO
        12182 BLIP2ITM
        12183 MobileSAM
        12184 YOLOv7
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_vlm_servers() -> bool:
    """
    检查四个 VLM server 是否已经启动。

    返回：
        True:
            四个 server 都在线。
        False:
            至少一个 server 不在线。
    """
    servers = {
        12181: "GroundingDINO",
        12182: "BLIP2ITM",
        12183: "MobileSAM",
        12184: "YOLOv7",
    }

    print("===== 检查 VLM servers =====")

    all_ok = True

    for port, name in servers.items():
        ok = is_port_open("127.0.0.1", port)
        status = "OK" if ok else "MISSING"
        print(f"{port} {name:14s}: {status}")

        if not ok:
            all_ok = False

    print()

    if not all_ok:
        print("[ERROR] 至少一个 VLM server 没有启动。")
        print()
        print("请在 Linux 辅助终端中先启动四个 VLM servers：")
        print()
        print("cd ~/vln_edge_task/external/vlfm")
        print("source ~/vln_edge_task/miniforge3/bin/activate ~/vln_edge_task/envs/vlfm_standard")
        print("export PYTHONPATH=$PWD:$PWD/yolov7:$PYTHONPATH")
        print("export VLFM_PYTHON=$(which python)")
        print("export HF_HOME=~/vln_edge_task/hf_cache")
        print("export HUGGINGFACE_HUB_CACHE=~/vln_edge_task/hf_cache/hub")
        print("export HF_ENDPOINT=https://hf-mirror.com")
        print("bash scripts/launch_vlm_servers.sh")
        print()
        print("等待后检查：")
        print('ss -ltnp | grep -E "12181|12182|12183|12184"')
        print()

    return all_ok


def build_env(args: argparse.Namespace) -> Dict[str, str]:
    """
    构造运行官方 vlfm.run 所需的环境变量。

    重点：
        TEST_VLFM_1_ENABLE=1
            开启我们在 vlfm_trainer.py 中加入的实时 recorder。

        TEST_VLFM_1_DIR
            指定结果保存目录。

        PYTHONPATH
            必须包含：
            - /home/nd/vln_edge_task
            - /home/nd/vln_edge_task/external/vlfm
            - /home/nd/vln_edge_task/external/vlfm/yolov7
    """
    env = os.environ.copy()

    python_paths = [
        str(PROJECT_ROOT),
        str(VLFM_ROOT),
        str(VLFM_ROOT / "yolov7"),
    ]

    old_pythonpath = env.get("PYTHONPATH", "")
    if old_pythonpath:
        python_paths.append(old_pythonpath)

    env["PYTHONPATH"] = ":".join(python_paths)

    env["HYDRA_FULL_ERROR"] = "1"

    env["HF_HOME"] = str(PROJECT_ROOT / "hf_cache")
    env["HUGGINGFACE_HUB_CACHE"] = str(PROJECT_ROOT / "hf_cache/hub")
    env["HF_ENDPOINT"] = "https://hf-mirror.com"

    env["TEST_VLFM_1_ENABLE"] = "0"#自己的可视化改造开关（05用）
    env["TEST_VLFM_1_PROJECT_ROOT"] = str(PROJECT_ROOT)
    env["TEST_VLFM_1_DIR"] = str(RESULT_DIR)
    env["TEST_VLFM_1_LIVE"] = "0" if args.no_live else "1"
    env["TEST_VLFM_1_SAVE_EVERY"] = str(args.save_every)
    env["TEST_VLFM_1_SCENE"] = "Z6MFQCViBuw"
    env["TEST_VLFM_1_TARGET"] = "chair"

    return env


def build_vlfm_command(args: argparse.Namespace) -> List[str]:
    """
    构造官方 VLFM 运行命令。

    等价于在 external/vlfm 目录下运行：

        python -u -m vlfm.run
            habitat.dataset.data_path=<单 episode 数据集>
            habitat_baselines.test_episode_count=1

    注意：
        不设置 video_option，因为本实验先不保存 mp4 视频；
        实时显示和 frames 保存由 LiveVLFMRecorder 负责。
    """
    hydra_dir = RESULT_DIR / "configs/hydra_run"

    cmd = [
        sys.executable,
        "-u",
        "-m",
        "vlfm.run",
        f"habitat.dataset.data_path={ONE_EPISODE_DATASET}",
        "habitat_baselines.test_episode_count=1",
        f"hydra.run.dir={hydra_dir}",
    ]

    if args.extra_override:
        cmd.extend(args.extra_override)

    return cmd


def stream_process_to_log(cmd: List[str], env: Dict[str, str], log_path: Path) -> int:
    """
    启动子进程，并把输出同时打印到 VS Code 终端和日志文件。

    参数：
        cmd:
            要运行的命令。

        env:
            环境变量。

        log_path:
            日志保存路径。

    返回：
        子进程退出码。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("===== 运行命令 =====")
    print(" ".join(str(x) for x in cmd))
    print()
    print(f"===== 日志保存到 =====")
    print(log_path)
    print()

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("===== command =====\n")
        log_file.write(" ".join(str(x) for x in cmd) + "\n\n")
        log_file.write("===== output =====\n")
        log_file.flush()

        process = subprocess.Popen(
            cmd,
            cwd=str(VLFM_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

        log_file.write("\n===== return code =====\n")
        log_file.write(str(return_code) + "\n")

    return return_code


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="在 VS Code 中运行 test_vlfm_1 真实 VLFM 单 episode 实时可视化实验"
    )

    parser.add_argument(
        "--skip-server-check",
        action="store_true",
        help="跳过 VLM server 端口检查。一般不建议使用。",
    )

    parser.add_argument(
        "--no-live",
        action="store_true",
        help="不弹出实时窗口，只保存 frames 和 snapshots。",
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="每隔多少步保存一张 frame。默认 5。",
    )

    parser.add_argument(
        "--extra-override",
        nargs="*",
        default=[],
        help=(
            "额外传给 Hydra 的 override。"
            "例如 habitat_baselines.eval.video_option=[]"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    主入口。
    """
    args = parse_args()

    make_dirs()

    if not ONE_EPISODE_DATASET.exists():
        raise FileNotFoundError(
            "没有找到单 episode 数据集：\n"
            f"{ONE_EPISODE_DATASET}\n"
            "请先运行 prepare_one_episode_dataset.py 生成小数据集。"
        )

    if not args.skip_server_check:
        ok = check_vlm_servers()
        if not ok:
            raise RuntimeError(
                "VLM servers 未全部启动。请先启动 12181/12182/12183/12184。"
            )

    env = build_env(args)
    cmd = build_vlfm_command(args)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = RESULT_DIR / "logs" / f"real_vlfm_one_episode_{timestamp}.log"

    return_code = stream_process_to_log(
        cmd=cmd,
        env=env,
        log_path=log_path,
    )

    print()
    print("===== VLFM run finished =====")
    print(f"return code: {return_code}")
    print(f"log path:    {log_path}")
    print(f"frames:      {RESULT_DIR / 'frames'}")
    print(f"snapshots:   {RESULT_DIR / 'snapshots'}")
    print(f"metrics:     {RESULT_DIR / 'metrics'}")

    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main()