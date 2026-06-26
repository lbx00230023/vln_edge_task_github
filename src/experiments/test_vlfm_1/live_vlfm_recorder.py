#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
live_vlfm_recorder.py

test_vlfm_1 实验专用实时可视化与结果保存模块。

功能：
    1. 实时显示 VLFM 运行过程；
    2. 保存 RGB、Depth、Top-down Map、VLFM 附加可视化图；
    3. 保存 frames、snapshots、logs、metrics；
    4. 尽量从 policy_info / infos 中自动提取可视化字段；
    5. 不参与导航决策，不改变 VLFM 策略。

当前优先识别的 VLFM 字段：
    top_down_map:
        Habitat / VLFM 返回的俯视地图，作为左下角地图显示。

    render_below_images:
        VLFM 官方可视化附加图，作为右下角 Value Map / Debug Map 区域显示。

    target_object:
        当前目标类别，例如 chair。

    target_detected / stop_called:
        用于推断当前 mode，例如 navigate / target_detected / stop。

输出目录：
    results/test_vlfm_1/
    ├── frames/
    ├── snapshots/
    ├── logs/
    └── metrics/
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class LiveVLFMRecorder:
    """
    VLFM 实时可视化与保存器。

    这个类只负责“显示”和“保存”，不参与 VLFM 决策。
    因此即使 recorder 出错，也不应该影响主程序导航。
    """

    def __init__(
        self,
        result_dir: str | Path,
        live: bool = True,
        save_every: int = 5,
        window_name: str = "test_vlfm_1_live",
        panel_cell_size: Tuple[int, int] = (480, 270),
    ) -> None:
        """
        初始化 recorder。

        参数：
            result_dir:
                test_vlfm_1 结果目录。

            live:
                是否实时弹出 OpenCV 窗口。

            save_every:
                每隔多少 step 保存一张 frame。

            window_name:
                OpenCV 窗口名。

            panel_cell_size:
                四宫格里每个子图的大小，格式为 (width, height)。
        """
        self.result_dir = Path(result_dir).expanduser().resolve()
        self.live = bool(live)
        self.save_every = max(1, int(save_every))
        self.window_name = window_name
        self.panel_cell_size = panel_cell_size

        self.frames_dir = self.result_dir / "frames"
        self.snapshots_dir = self.result_dir / "snapshots"
        self.logs_dir = self.result_dir / "logs"
        self.metrics_dir = self.result_dir / "metrics"

        for directory in [
            self.frames_dir,
            self.snapshots_dir,
            self.logs_dir,
            self.metrics_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        self.step_log_path = self.logs_dir / "live_steps.jsonl"
        self.meta_path = self.logs_dir / "live_recorder_meta.json"
        self.policy_keys_path = self.logs_dir / "policy_info_keys.txt"
        self.debug_once_path = self.logs_dir / "policy_info_debug_once.txt"

        self.start_time = time.time()
        self.last_panel: Optional[np.ndarray] = None
        self.stop_requested = False

        self._policy_keys_written = False
        self._debug_once_written = False

        self._write_meta()

    def _write_meta(self) -> None:
        """
        写入 recorder 基础信息，方便后续复现实验。
        """
        meta = {
            "result_dir": str(self.result_dir),
            "live": self.live,
            "save_every": self.save_every,
            "window_name": self.window_name,
            "panel_cell_size": list(self.panel_cell_size),
            "created_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.meta_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """
        将 numpy / torch 等对象尽量转成可写入 json 的简单类型。
        """
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        if isinstance(value, np.generic):
            return value.item()

        if isinstance(value, np.ndarray):
            return {
                "type": "ndarray",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }

        if isinstance(value, dict):
            return {str(k): LiveVLFMRecorder._json_safe(v) for k, v in value.items()}

        if isinstance(value, (list, tuple)):
            return [LiveVLFMRecorder._json_safe(v) for v in value]

        return str(value)

    @staticmethod
    def _extract_from_dict(data: Optional[Dict[str, Any]], candidate_keys: list[str]) -> Any:
        """
        从字典中按照候选 key 提取第一个存在的值。
        """
        if not isinstance(data, dict):
            return None

        for key in candidate_keys:
            if key in data:
                return data[key]

        return None

    @staticmethod
    def _call_visualize_if_possible(value: Any) -> Optional[np.ndarray]:
        """
        尽量把任意对象转换成可显示图像。

        支持：
            1. np.ndarray；
            2. 带 visualize() 方法的对象；
            3. Habitat top_down_map 字典；
            4. render_below_images 这类 list / tuple；
            5. 嵌套 dict。
        """
        if value is None:
            return None

        # 1. 直接是 numpy 图像
        if isinstance(value, np.ndarray):
            return value

        # 2. 对象带 visualize()
        visualize = getattr(value, "visualize", None)
        if callable(visualize):
            try:
                img = visualize()
                if isinstance(img, np.ndarray):
                    return img
            except Exception as exc:
                print(f"[LiveVLFMRecorder][WARN] visualize() 调用失败: {repr(exc)}")

        # 3. Habitat top_down_map 字典
        if isinstance(value, dict) and "map" in value and isinstance(value["map"], np.ndarray):
            return LiveVLFMRecorder.render_top_down_map(value)

        # 4. list / tuple：例如 render_below_images
        if isinstance(value, (list, tuple)):
            images: list[np.ndarray] = []
            for item in value:
                img = LiveVLFMRecorder._call_visualize_if_possible(item)
                if img is not None:
                    images.append(img)

            if images:
                return LiveVLFMRecorder.stack_images_horizontally(images)

            return None

        # 5. 普通 dict：递归找图像
        if isinstance(value, dict):
            images: list[np.ndarray] = []
            for item in value.values():
                img = LiveVLFMRecorder._call_visualize_if_possible(item)
                if img is not None:
                    images.append(img)

            if not images:
                return None

            if len(images) == 1:
                return images[0]

            return LiveVLFMRecorder.stack_images_horizontally(images)

        return None

    @staticmethod
    def stack_images_horizontally(images: list[np.ndarray]) -> Optional[np.ndarray]:
        """
        将多张图横向拼接。

        用途：
            render_below_images 可能包含多张官方调试图。
            第一版先全部横向拼起来，确保信息不丢。
        """
        if not images:
            return None

        processed: list[np.ndarray] = []

        for img in images:
            arr = np.asarray(img)

            # CHW -> HWC
            if arr.ndim == 3 and arr.shape[0] in [1, 3, 4] and arr.shape[-1] not in [1, 3, 4]:
                arr = np.transpose(arr, (1, 2, 0))

            if arr.ndim == 2:
                arr = arr.astype(np.float32)
                if arr.max() <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = arr[:, :, 0].astype(np.float32)
                if arr.max() <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

            elif arr.ndim == 3 and arr.shape[2] in [3, 4]:
                if arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                arr = arr.astype(np.float32)
                if arr.max() <= 1.0:
                    arr = arr * 255.0
                arr = np.clip(arr, 0, 255).astype(np.uint8)

            else:
                continue

            processed.append(arr)

        if not processed:
            return None

        # 统一高度
        target_height = min(300, max(img.shape[0] for img in processed))
        resized: list[np.ndarray] = []

        for img in processed:
            h, w = img.shape[:2]
            new_w = max(1, int(w * target_height / max(1, h)))
            resized.append(cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA))

        return np.hstack(resized)

    @staticmethod
    def render_top_down_map(top_down_map: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        将 Habitat 的 top_down_map 字典转换为 RGB 图像。

        常见字段：
            map:
                2D 地图。

            fog_of_war_mask:
                探索区域 mask。

            agent_map_coord:
                agent 坐标。

            agent_angle:
                agent 朝向。
        """
        map_arr = top_down_map.get("map")
        if not isinstance(map_arr, np.ndarray):
            return None

        arr = np.asarray(map_arr)

        # 如果已经是彩色图，直接处理
        if arr.ndim == 3 and arr.shape[2] in [3, 4]:
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            return np.clip(arr, 0, 255).astype(np.uint8)

        if arr.ndim != 2:
            return None

        arr = arr.astype(np.float32)

        if arr.max() > arr.min():
            norm = (arr - arr.min()) / (arr.max() - arr.min())
        else:
            norm = np.zeros_like(arr, dtype=np.float32)

        gray = (norm * 255).astype(np.uint8)

        # OpenCV colormap 返回 BGR，转 RGB 供后续统一处理
        vis_bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
        vis = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)

        # 未探索区域压暗
        fog = top_down_map.get("fog_of_war_mask")
        if isinstance(fog, np.ndarray) and fog.shape[:2] == arr.shape[:2]:
            mask = fog.astype(bool)
            vis[~mask] = (vis[~mask] * 0.35).astype(np.uint8)

        # 画 agent 位置
        agent_coord = top_down_map.get("agent_map_coord")
        if agent_coord is not None:
            try:
                y, x = int(agent_coord[0]), int(agent_coord[1])
                cv2.circle(vis, (x, y), 6, (255, 255, 255), -1)
                cv2.circle(vis, (x, y), 9, (0, 0, 0), 2)

                angle = top_down_map.get("agent_angle")
                if angle is not None:
                    length = 20
                    dx = int(length * np.cos(float(angle)))
                    dy = int(length * np.sin(float(angle)))
                    cv2.line(vis, (x, y), (x + dx, y + dy), (255, 255, 255), 2)
            except Exception:
                pass

        return vis

    @classmethod
    def extract_rgb(
        cls,
        observations: Optional[Dict[str, Any]] = None,
        rgb: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        提取 RGB 图像。
        """
        if isinstance(rgb, np.ndarray):
            return rgb

        value = cls._extract_from_dict(
            observations,
            ["rgb", "robot_rgb", "head_rgb", "color", "image"],
        )

        if isinstance(value, np.ndarray):
            return value

        return None

    @classmethod
    def extract_depth(
        cls,
        observations: Optional[Dict[str, Any]] = None,
        depth: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """
        提取 Depth 图像。
        """
        if isinstance(depth, np.ndarray):
            return depth

        value = cls._extract_from_dict(
            observations,
            ["depth", "robot_depth", "head_depth"],
        )

        if isinstance(value, np.ndarray):
            return value

        return None

    @classmethod
    def extract_obstacle_map(
        cls,
        policy_info: Optional[Dict[str, Any]] = None,
        obstacle_map: Any = None,
    ) -> Optional[np.ndarray]:
        """
        提取左下角地图图像。

        当前优先使用：
            top_down_map
        """
        explicit = cls._call_visualize_if_possible(obstacle_map)
        if explicit is not None:
            return explicit

        value = cls._extract_from_dict(
            policy_info,
            [
                "top_down_map",
                "obstacle_map",
                "obstacle_map_vis",
                "obstacle_map_visualization",
                "map",
                "map_vis",
            ],
        )

        return cls._call_visualize_if_possible(value)

    @classmethod
    def extract_value_map(
        cls,
        policy_info: Optional[Dict[str, Any]] = None,
        value_map: Any = None,
    ) -> Optional[np.ndarray]:
        """
        提取右下角 Value Map / VLFM 附加可视化图。

        当前优先使用：
            render_below_images
        """
        explicit = cls._call_visualize_if_possible(value_map)
        if explicit is not None:
            return explicit

        value = cls._extract_from_dict(
            policy_info,
            [
                "render_below_images",
                "value_map",
                "value_map_vis",
                "value_map_visualization",
                "semantic_value_map",
                "semantic_value_map_vis",
            ],
        )

        return cls._call_visualize_if_possible(value)

    @staticmethod
    def to_uint8_bgr(
        image: Optional[np.ndarray],
        default_text: str,
        cell_size: Tuple[int, int],
        is_depth: bool = False,
    ) -> np.ndarray:
        """
        将任意图像转换为 OpenCV 可显示的 uint8 BGR 图像。
        """
        width, height = cell_size

        if image is None:
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 245
            cv2.putText(
                canvas,
                default_text,
                (20, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (80, 80, 80),
                2,
                cv2.LINE_AA,
            )
            return canvas

        arr = np.asarray(image)

        # CHW -> HWC
        if arr.ndim == 3 and arr.shape[0] in [1, 3, 4] and arr.shape[-1] not in [1, 3, 4]:
            arr = np.transpose(arr, (1, 2, 0))

        if is_depth:
            arr = np.squeeze(arr).astype(np.float32)
            valid = np.isfinite(arr)

            if valid.any():
                min_val = float(np.nanpercentile(arr[valid], 2))
                max_val = float(np.nanpercentile(arr[valid], 98))
                if max_val <= min_val:
                    max_val = min_val + 1e-6
                norm = (arr - min_val) / (max_val - min_val)
                norm = np.clip(norm, 0.0, 1.0)
            else:
                norm = np.zeros_like(arr, dtype=np.float32)

            gray = (norm * 255).astype(np.uint8)
            bgr = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

        if arr.ndim == 2:
            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0].astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        elif arr.ndim == 3 and arr.shape[2] in [3, 4]:
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]

            arr = arr.astype(np.float32)
            if arr.max() <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

            # Habitat RGB 默认按 RGB 处理，OpenCV 显示需要 BGR
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        else:
            return LiveVLFMRecorder.to_uint8_bgr(None, default_text, cell_size)

        return cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def add_title(image: np.ndarray, title: str) -> np.ndarray:
        """
        给子图添加标题条。
        """
        title_bar_height = 32
        h, w = image.shape[:2]

        canvas = np.zeros((h + title_bar_height, w, 3), dtype=np.uint8)
        canvas[:title_bar_height, :, :] = (40, 40, 40)
        canvas[title_bar_height:, :, :] = image

        cv2.putText(
            canvas,
            title,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def compose_panel(
        self,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        obstacle_map: Optional[np.ndarray] = None,
        value_map: Optional[np.ndarray] = None,
        text_info: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        拼接四宫格面板。
        """
        rgb_img = self.to_uint8_bgr(rgb, "No RGB", self.panel_cell_size)
        depth_img = self.to_uint8_bgr(depth, "No Depth", self.panel_cell_size, is_depth=True)
        obstacle_img = self.to_uint8_bgr(obstacle_map, "No Obstacle Map", self.panel_cell_size)
        value_img = self.to_uint8_bgr(value_map, "No Value Map", self.panel_cell_size)

        rgb_img = self.add_title(rgb_img, "RGB")
        depth_img = self.add_title(depth_img, "Depth")
        obstacle_img = self.add_title(obstacle_img, "Top-down Map / Trajectory")
        value_img = self.add_title(value_img, "VLFM Render / Value Map")

        top = np.hstack([rgb_img, depth_img])
        bottom = np.hstack([obstacle_img, value_img])
        panel = np.vstack([top, bottom])

        info_bar_height = 70
        h, w = panel.shape[:2]
        final_panel = np.ones((h + info_bar_height, w, 3), dtype=np.uint8) * 255
        final_panel[:h, :, :] = panel

        text_info = text_info or {}

        line1 = (
            f"step={text_info.get('step', 'NA')} | "
            f"mode={text_info.get('mode', 'NA')} | "
            f"action={text_info.get('action', 'NA')} | "
            f"target={text_info.get('target', 'NA')}"
        )

        line2 = (
            f"scene={text_info.get('scene_id', 'NA')} | "
            f"episode={text_info.get('episode_id', 'NA')} | "
            f"elapsed={float(text_info.get('elapsed_sec', 0.0)):.1f}s | "
            f"press q to request stop"
        )

        cv2.putText(
            final_panel,
            line1,
            (12, h + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            final_panel,
            line2,
            (12, h + 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

        return final_panel

    def _write_policy_info_keys_once(self, policy_info: Optional[Dict[str, Any]]) -> None:
        """
        第一次接收到 policy_info 时，记录所有字段名。
        """
        if self._policy_keys_written:
            return

        if not isinstance(policy_info, dict):
            return

        keys = sorted(str(k) for k in policy_info.keys())
        self.policy_keys_path.write_text("\n".join(keys), encoding="utf-8")
        self._policy_keys_written = True

    def _write_policy_info_debug_once(self, policy_info: Optional[Dict[str, Any]]) -> None:
        """
        第一次接收到 policy_info 时，记录 key、类型、shape 等调试信息。
        """
        if self._debug_once_written:
            return

        if not isinstance(policy_info, dict):
            return

        lines: list[str] = []
        lines.append("===== policy_info debug once =====")

        for key in sorted(policy_info.keys()):
            value = policy_info[key]
            lines.append(f"{key}: type={type(value)}")

            if isinstance(value, dict):
                lines.append(f"  dict_keys={list(value.keys())}")

            if isinstance(value, (list, tuple)):
                lines.append(f"  len={len(value)}")
                if len(value) > 0:
                    lines.append(f"  first_type={type(value[0])}")
                    first = value[0]
                    if hasattr(first, "shape"):
                        lines.append(f"  first_shape={getattr(first, 'shape', None)}")

            if hasattr(value, "shape"):
                lines.append(f"  shape={getattr(value, 'shape', None)}")
                lines.append(f"  dtype={getattr(value, 'dtype', None)}")

        self.debug_once_path.write_text("\n".join(lines), encoding="utf-8")
        self._debug_once_written = True

    def _append_step_log(self, record: Dict[str, Any]) -> None:
        """
        写入每一步 jsonl 日志。
        """
        safe_record = self._json_safe(record)
        with self.step_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe_record, ensure_ascii=False) + "\n")

    def save_snapshot_images(
        self,
        panel: np.ndarray,
        rgb: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        obstacle_map: Optional[np.ndarray],
        value_map: Optional[np.ndarray],
    ) -> None:
        """
        保存最新截图。
        """
        cv2.imwrite(str(self.snapshots_dir / "latest_panel.jpg"), panel)

        if rgb is not None:
            cv2.imwrite(
                str(self.snapshots_dir / "latest_rgb.jpg"),
                self.to_uint8_bgr(rgb, "No RGB", self.panel_cell_size),
            )

        if depth is not None:
            cv2.imwrite(
                str(self.snapshots_dir / "latest_depth.png"),
                self.to_uint8_bgr(depth, "No Depth", self.panel_cell_size, is_depth=True),
            )

        if obstacle_map is not None:
            cv2.imwrite(
                str(self.snapshots_dir / "latest_obstacle_map.jpg"),
                self.to_uint8_bgr(obstacle_map, "No Obstacle Map", self.panel_cell_size),
            )

        if value_map is not None:
            cv2.imwrite(
                str(self.snapshots_dir / "latest_value_map.jpg"),
                self.to_uint8_bgr(value_map, "No Value Map", self.panel_cell_size),
            )

    @staticmethod
    def infer_mode(policy_info: Optional[Dict[str, Any]], fallback: Optional[str]) -> str:
        """
        从 policy_info 中推断 mode。
        """
        if fallback is not None and str(fallback) not in ["", "None", "UNKNOWN_MODE"]:
            return str(fallback)

        if isinstance(policy_info, dict):
            for key in ["mode", "nav_mode", "state"]:
                value = policy_info.get(key)
                if value is not None:
                    return str(value)

            if policy_info.get("stop_called", 0):
                return "stop"

            if policy_info.get("target_detected", 0):
                return "target_detected"

        return "navigate"

    @staticmethod
    def infer_target(policy_info: Optional[Dict[str, Any]], fallback: Optional[str]) -> str:
        """
        从 policy_info 或环境变量中推断 target。
        """
        if fallback is not None and str(fallback) not in ["", "None", "UNKNOWN_TARGET"]:
            return str(fallback)

        if isinstance(policy_info, dict):
            value = policy_info.get("target_object")
            if isinstance(value, (list, tuple)) and len(value) > 0:
                value = value[0]
            if value is not None:
                return str(value)

        env_target = os.environ.get("TEST_VLFM_1_TARGET")
        if env_target:
            return env_target

        return "UNKNOWN_TARGET"

    def update(
        self,
        step: int,
        observations: Optional[Dict[str, Any]] = None,
        infos: Optional[Dict[str, Any]] = None,
        policy_info: Optional[Dict[str, Any]] = None,
        action: Any = None,
        mode: Optional[str] = None,
        target: Optional[str] = None,
        scene_id: Optional[str] = None,
        episode_id: Optional[str] = None,
        rgb: Optional[np.ndarray] = None,
        depth: Optional[np.ndarray] = None,
        obstacle_map: Any = None,
        value_map: Any = None,
    ) -> None:
        """
        每一步更新实时窗口和保存结果。
        """
        elapsed_sec = time.time() - self.start_time

        self._write_policy_info_keys_once(policy_info)
        self._write_policy_info_debug_once(policy_info)

        rgb_img = self.extract_rgb(observations, rgb)
        depth_img = self.extract_depth(observations, depth)
        obstacle_img = self.extract_obstacle_map(policy_info, obstacle_map)
        value_img = self.extract_value_map(policy_info, value_map)

        mode_str = self.infer_mode(policy_info, mode)
        target_str = self.infer_target(policy_info, target)

        if scene_id is None:
            scene_id = os.environ.get("TEST_VLFM_1_SCENE", "UNKNOWN_SCENE")

        text_info = {
            "step": step,
            "mode": mode_str,
            "action": action,
            "target": target_str,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "elapsed_sec": elapsed_sec,
        }

        panel = self.compose_panel(
            rgb=rgb_img,
            depth=depth_img,
            obstacle_map=obstacle_img,
            value_map=value_img,
            text_info=text_info,
        )

        self.last_panel = panel

        record = {
            "step": int(step),
            "mode": mode_str,
            "action": str(action),
            "target": target_str,
            "scene_id": str(scene_id),
            "episode_id": str(episode_id),
            "elapsed_sec": float(elapsed_sec),
            "has_rgb": rgb_img is not None,
            "has_depth": depth_img is not None,
            "has_top_down_map": obstacle_img is not None,
            "has_render_below_images": value_img is not None,
        }

        self._append_step_log(record)

        self.save_snapshot_images(
            panel=panel,
            rgb=rgb_img,
            depth=depth_img,
            obstacle_map=obstacle_img,
            value_map=value_img,
        )

        if int(step) % self.save_every == 0:
            frame_path = self.frames_dir / f"step_{int(step):06d}.jpg"
            cv2.imwrite(str(frame_path), panel)

        if self.live:
            cv2.imshow(self.window_name, panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stop_requested = True

    def close(self, final_metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        关闭 recorder，保存最终 metrics。
        """
        final_metrics = dict(final_metrics or {})
        final_metrics["total_elapsed_sec"] = time.time() - self.start_time
        final_metrics["closed_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

        metrics_path = self.metrics_dir / "episode_result.json"
        metrics_path.write_text(
            json.dumps(self._json_safe(final_metrics), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if self.live:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass


def run_demo(result_dir: Path, steps: int = 30, live: bool = True) -> None:
    """
    独立 demo，用假数据测试显示和保存逻辑。
    """
    recorder = LiveVLFMRecorder(
        result_dir=result_dir,
        live=live,
        save_every=5,
        window_name="test_vlfm_1_recorder_demo",
    )

    h, w = 480, 640

    for step in range(steps):
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        rgb[:, :, 0] = 120
        rgb[:, :, 1] = 80
        rgb[:, :, 2] = 40
        cv2.circle(rgb, (80 + step * 10 % 500, 240), 40, (255, 80, 80), -1)

        depth = np.tile(np.linspace(0, 1, w, dtype=np.float32), (h, 1))

        fake_top_down = {
            "map": np.random.randint(0, 5, size=(300, 300), dtype=np.uint8),
            "fog_of_war_mask": np.ones((300, 300), dtype=np.uint8),
            "agent_map_coord": np.array([150, 40 + step * 5 % 220]),
            "agent_angle": step * 0.1,
        }

        fake_value_1 = np.zeros((260, 260, 3), dtype=np.uint8)
        fake_value_1[:, :, 0] = np.linspace(0, 255, 260, dtype=np.uint8)[None, :]
        fake_value_1[:, :, 1] = 120
        fake_value_1[:, :, 2] = 255 - fake_value_1[:, :, 0]

        fake_value_2 = np.ones((260, 260, 3), dtype=np.uint8) * 230
        cv2.putText(
            fake_value_2,
            "render_below",
            (25, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        policy_info = {
            "top_down_map": fake_top_down,
            "render_below_images": [fake_value_1, fake_value_2],
            "target_object": "chair",
            "target_detected": int(step > steps // 2),
            "stop_called": int(step == steps - 1),
        }

        recorder.update(
            step=step,
            observations={"rgb": rgb, "depth": depth},
            policy_info=policy_info,
            action=1,
            scene_id="Z6MFQCViBuw",
            episode_id="demo_episode",
        )

        time.sleep(0.05)

    recorder.close(
        final_metrics={
            "demo": True,
            "steps": steps,
            "success": None,
            "spl": None,
        }
    )

    print("===== demo finished =====")
    print(f"result_dir: {result_dir}")
    print(f"frames:     {result_dir / 'frames'}")
    print(f"snapshots:  {result_dir / 'snapshots'}")
    print(f"logs:       {result_dir / 'logs'}")
    print(f"metrics:    {result_dir / 'metrics'}")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="test_vlfm_1 实时可视化 recorder")

    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path.home() / "vln_edge_task/results/test_vlfm_1",
        help="test_vlfm_1 结果目录",
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行独立 demo",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="demo 步数",
    )

    parser.add_argument(
        "--no-live",
        action="store_true",
        help="不弹出窗口，只保存文件",
    )

    return parser.parse_args()


def main() -> None:
    """
    主函数。
    """
    args = parse_args()

    if args.demo:
        run_demo(
            result_dir=args.result_dir,
            steps=args.steps,
            live=not args.no_live,
        )
    else:
        print(
            "当前 live_vlfm_recorder.py 主要作为模块被导入使用。\n"
            "如需测试，请运行：\n"
            "python src/experiments/test_vlfm_1/live_vlfm_recorder.py --demo"
        )


if __name__ == "__main__":
    main()