#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_one_episode_dataset.py

功能：
    为 test_vlfm_1 实验准备一个“单场景、单 episode”的 Habitat ObjectNav 数据集。

为什么需要这个脚本：
    官方 MP3D ObjectNav val 数据集包含很多 scene 和很多 episode。
    默认直接运行 VLFM 时，会尝试跑完整 val 集合，例如 2195 个 episode。
    现在我们只想先做一个可控的小实验：
        1. 固定一个 MP3D scene；
        2. 固定一个目标物体类别，例如 chair / table / bed；
        3. 只运行一个 episode；
        4. 方便观察实时运动过程、保存地图和可视化结果。

输入：
    /home/nd/vln_edge_task/external/vlfm/data/datasets/objectnav/mp3d/val/val.json.gz
    /home/nd/vln_edge_task/external/vlfm/data/datasets/objectnav/mp3d/val/content/*.json.gz

输出：
    /home/nd/vln_edge_task/results/test_vlfm_1/configs/objectnav_one_episode/val/val.json.gz
    /home/nd/vln_edge_task/results/test_vlfm_1/configs/objectnav_one_episode/val/content/<scene_id>.json.gz
    /home/nd/vln_edge_task/results/test_vlfm_1/configs/selected_episode_summary.json

注意：
    本脚本不会修改官方数据集。
    它只是在 results/test_vlfm_1/configs/ 下生成一个小数据集副本。
    liunx终端运行：python src/experiments/test_vlfm_1/prepare_one_episode_dataset.py --list-only \
  2>&1 | tee results/test_vlfm_1/logs/list_episodes_$(date +%Y%m%d_%H%M%S).log
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json_gz(path: Path) -> Dict[str, Any]:
    """
    读取 .json.gz 文件。

    参数：
        path:
            输入 json.gz 文件路径。

    返回：
        Python 字典。
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def save_json_gz(data: Dict[str, Any], path: Path) -> None:
    """
    保存 .json.gz 文件。

    参数：
        data:
            要保存的数据。
        path:
            输出路径。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(data, f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    """
    保存普通 json 文件。

    主要用于保存 selected_episode_summary.json，
    方便后续查看本次实验选择了哪个 scene、哪个目标、哪个 episode。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def short_scene_id(scene_id: str) -> str:
    """
    从 Habitat episode 的 scene_id 字段中提取短 scene 名。

    例子：
        data/scene_datasets/mp3d/Z6MFQCViBuw/Z6MFQCViBuw.glb
    提取为：
        Z6MFQCViBuw
    """
    if not scene_id:
        return "UNKNOWN_SCENE"

    path = Path(scene_id)

    # 常见情况：scene_id 以 .glb 结尾
    if path.suffix == ".glb":
        return path.stem

    # 兜底情况：返回最后一级路径名
    return path.name


def get_episode_target_category(episode: Dict[str, Any]) -> str:
    """
    从 episode 中提取目标物体类别。

    不同版本的 ObjectNav 数据格式可能略有差异。
    常见字段包括：
        - object_category
        - goal_name
        - target
        - category
        - goals[0]["object_category"]
        - goals[0]["category"]

    本函数尽量兼容这些格式。
    """
    direct_keys = [
        "object_category",
        "goal_name",
        "target",
        "category",
    ]

    for key in direct_keys:
        value = episode.get(key)
        if isinstance(value, str) and value:
            return value

    goals = episode.get("goals", [])
    if isinstance(goals, list) and len(goals) > 0:
        first_goal = goals[0]
        if isinstance(first_goal, dict):
            goal_keys = [
                "object_category",
                "category",
                "object_name",
                "name",
            ]
            for key in goal_keys:
                value = first_goal.get(key)
                if isinstance(value, str) and value:
                    return value

    return "UNKNOWN_TARGET"


def load_all_content_files(src_val_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """
    读取官方 val/content 目录下所有 scene 的 episode 文件。

    参数：
        src_val_dir:
            官方 MP3D ObjectNav val 目录，例如：
            /home/nd/vln_edge_task/external/vlfm/data/datasets/objectnav/mp3d/val

    返回：
        一个列表：
            [
                (content_file_path, content_file_data),
                ...
            ]
    """
    content_dir = src_val_dir / "content"

    if not content_dir.exists():
        raise FileNotFoundError(f"找不到 content 目录: {content_dir}")

    content_files = sorted(content_dir.glob("*.json.gz"))
    if not content_files:
        raise FileNotFoundError(f"content 目录下没有 json.gz 文件: {content_dir}")

    loaded_files: List[Tuple[Path, Dict[str, Any]]] = []

    for path in content_files:
        try:
            data = load_json_gz(path)
            loaded_files.append((path, data))
        except Exception as exc:
            print(f"[WARN] 读取失败，跳过: {path} | {repr(exc)}")

    if not loaded_files:
        raise RuntimeError("没有成功读取任何 content/*.json.gz 文件")

    return loaded_files


def print_dataset_summary(src_val_dir: Path, max_scenes: int = 40) -> None:
    """
    打印数据集摘要。

    作用：
        先帮助我们确认当前数据集中有哪些 scene、有哪些目标类别，
        然后再决定 test_vlfm_1 选择哪个 scene 和 target。
    """
    loaded_files = load_all_content_files(src_val_dir)

    scene_counter: Counter[str] = Counter()
    target_counter: Counter[str] = Counter()
    scene_target_counter: Dict[str, Counter[str]] = defaultdict(Counter)

    total_episodes = 0

    for _path, data in loaded_files:
        episodes = data.get("episodes", [])

        if not isinstance(episodes, list):
            continue

        for episode in episodes:
            scene = short_scene_id(str(episode.get("scene_id", "")))
            target = get_episode_target_category(episode)

            scene_counter[scene] += 1
            target_counter[target] += 1
            scene_target_counter[scene][target] += 1
            total_episodes += 1

    print("===== Dataset summary =====")
    print(f"val dir:        {src_val_dir}")
    print(f"scene files:    {len(loaded_files)}")
    print(f"total episodes: {total_episodes}")
    print()

    print("===== Top scenes =====")
    for scene, count in scene_counter.most_common(max_scenes):
        targets = scene_target_counter[scene].most_common(8)
        target_preview = ", ".join(f"{target}:{num}" for target, num in targets)
        print(f"{scene:16s} episodes={count:4d} targets=({target_preview})")

    print()

    print("===== Top target categories =====")
    for target, count in target_counter.most_common(40):
        print(f"{target:24s} {count}")


def find_matching_episodes(
    src_val_dir: Path,
    scene_id: Optional[str],
    target_category: Optional[str],
) -> List[Tuple[Path, Dict[str, Any], Dict[str, Any]]]:
    """
    查找符合条件的 episode。

    参数：
        src_val_dir:
            官方 val 目录。
        scene_id:
            指定 scene 短名，例如 Z6MFQCViBuw。
            如果为 None，则不限制 scene。
        target_category:
            指定目标类别，例如 chair。
            如果为 None，则不限制目标类别。

    返回：
        列表，每个元素为：
            (来源 content 文件路径, 来源 content 文件数据, episode 数据)
    """
    loaded_files = load_all_content_files(src_val_dir)

    matches: List[Tuple[Path, Dict[str, Any], Dict[str, Any]]] = []

    for path, data in loaded_files:
        episodes = data.get("episodes", [])

        if not isinstance(episodes, list):
            continue

        for episode in episodes:
            episode_scene = short_scene_id(str(episode.get("scene_id", "")))
            episode_target = get_episode_target_category(episode)

            if scene_id is not None and episode_scene != scene_id:
                continue

            if target_category is not None and episode_target != target_category:
                continue

            matches.append((path, data, episode))

    return matches


def build_one_episode_dataset(
    src_val_dir: Path,
    dst_val_dir: Path,
    summary_path: Path,
    scene_id: Optional[str],
    target_category: Optional[str],
    episode_index: int,
) -> Dict[str, Any]:
    """
    构建一个单 episode 小数据集。

    输出结构：
        dst_val_dir/
        ├── val.json.gz
        └── content/
            └── <scene_id>.json.gz

    说明：
        val.json.gz 保留官方顶层元信息，但 episodes 置为空。
        content/<scene_id>.json.gz 里只保留一个 episode。
    """
    src_val_json = src_val_dir / "val.json.gz"

    if not src_val_json.exists():
        raise FileNotFoundError(f"找不到官方 val.json.gz: {src_val_json}")

    top_level_data = load_json_gz(src_val_json)

    matches = find_matching_episodes(
        src_val_dir=src_val_dir,
        scene_id=scene_id,
        target_category=target_category,
    )

    if not matches:
        raise RuntimeError(
            "没有找到符合条件的 episode。"
            f" scene_id={scene_id}, target_category={target_category}。"
            " 建议先运行 --list-only 查看可用 scene 和 target。"
        )

    if episode_index < 0 or episode_index >= len(matches):
        raise IndexError(
            f"episode_index={episode_index} 超出范围，"
            f"匹配到的 episode 数量为 {len(matches)}"
        )

    src_content_path, src_content_data, selected_episode = matches[episode_index]

    selected_scene = short_scene_id(str(selected_episode.get("scene_id", "")))
    selected_target = get_episode_target_category(selected_episode)
    selected_episode_id = selected_episode.get("episode_id", "UNKNOWN_EPISODE")

    print("===== Selected episode =====")
    print(f"source content file: {src_content_path}")
    print(f"scene:               {selected_scene}")
    print(f"target:              {selected_target}")
    print(f"episode_id:          {selected_episode_id}")
    print(f"matched count:       {len(matches)}")
    print()

    # 清理旧输出，保证本次小数据集是干净的
    if dst_val_dir.exists():
        shutil.rmtree(dst_val_dir)

    dst_content_dir = dst_val_dir / "content"
    dst_content_dir.mkdir(parents=True, exist_ok=True)

    # 顶层 val.json.gz：保留元信息，清空 episodes
    new_top_level_data = dict(top_level_data)
    new_top_level_data["episodes"] = []

    # scene content 文件：保留元信息，只放一个 episode
    new_content_data = dict(src_content_data)
    new_content_data["episodes"] = [selected_episode]

    dst_val_json = dst_val_dir / "val.json.gz"
    dst_content_json = dst_content_dir / f"{selected_scene}.json.gz"

    save_json_gz(new_top_level_data, dst_val_json)
    save_json_gz(new_content_data, dst_content_json)

    summary = {
        "experiment": "test_vlfm_1",
        "src_val_dir": str(src_val_dir),
        "dst_val_dir": str(dst_val_dir),
        "dst_val_json": str(dst_val_json),
        "dst_content_json": str(dst_content_json),
        "selected_scene": selected_scene,
        "selected_target": selected_target,
        "selected_episode_id": selected_episode_id,
        "matched_episode_count": len(matches),
        "episode_index": episode_index,
        "scene_id_raw": selected_episode.get("scene_id", ""),
        "object_category": selected_target,
    }

    save_json(summary, summary_path)

    print("===== Output files =====")
    print(dst_val_json)
    print(dst_content_json)
    print(summary_path)

    return summary


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(
        description="为 test_vlfm_1 准备单场景单 episode ObjectNav 数据集"
    )

    parser.add_argument(
        "--src-val-dir",
        type=Path,
        default=Path.home()
        / "vln_edge_task/external/vlfm/data/datasets/objectnav/mp3d/val",
        help="官方 MP3D ObjectNav val 目录",
    )

    parser.add_argument(
        "--dst-val-dir",
        type=Path,
        default=Path.home()
        / "vln_edge_task/results/test_vlfm_1/configs/objectnav_one_episode/val",
        help="输出的小数据集 val 目录",
    )

    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path.home()
        / "vln_edge_task/results/test_vlfm_1/configs/selected_episode_summary.json",
        help="保存所选 episode 摘要信息的位置",
    )

    parser.add_argument(
        "--scene-id",
        type=str,
        default=None,
        help="指定 scene 短名，例如 Z6MFQCViBuw。不填则不限制 scene。",
    )

    parser.add_argument(
        "--target-category",
        type=str,
        default=None,
        help="指定目标类别，例如 chair / table / bed。不填则不限制目标类别。",
    )

    parser.add_argument(
        "--episode-index",
        type=int,
        default=0,
        help="当匹配到多个 episode 时，选择第几个。默认 0。",
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="只打印数据集摘要，不生成小数据集。",
    )

    return parser.parse_args()


def main() -> None:
    """
    主函数。
    """
    args = parse_args()

    if args.list_only:
        print_dataset_summary(args.src_val_dir)
        return

    build_one_episode_dataset(
        src_val_dir=args.src_val_dir,
        dst_val_dir=args.dst_val_dir,
        summary_path=args.summary_path,
        scene_id=args.scene_id,
        target_category=args.target_category,
        episode_index=args.episode_index,
    )


if __name__ == "__main__":
    main()