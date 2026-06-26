# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from omegaconf import OmegaConf


def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)


@baseline_registry.register_trainer(name="vlfm")
class VLFMTrainer(PPOTrainer):
    envs: VectorEnv

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu", weights_only=False)
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}

        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(render_view.uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)

        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *self._agent.hidden_state_shape,
            ),
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames: List[List[np.ndarray]] = [[] for _ in range(self.config.habitat_baselines.num_environments)]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, dataset only has {{total_num_eps}}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert number_of_eval_episodes > 0, "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self._agent.eval()

        from vlfm.utils.habitat_visualizer import HabitatVis

        num_successes = 0
        num_total = 0
        hab_vis = HabitatVis()
         # 修改1.test_vlfm_1 实时可视化 recorder
        # 说明：
        # 1. 这个 recorder 只负责“显示”和“保存”，不参与导航决策；
        # 2. 默认不开启，只有设置 TEST_VLFM_1_ENABLE=1 时才启用；
        # 3. 这样不会影响官方 VLFM 的正常 evaluation；
        # 4. 结果会保存到 TEST_VLFM_1_DIR 指定的目录。
        test_vlfm_1_recorder = None
        test_vlfm_1_step = 0

        if os.environ.get("TEST_VLFM_1_ENABLE", "0") == "1":
            try:
                import sys

                # 保证可以从 /home/nd/vln_edge_task/src/... 导入我们自己的实验代码
                project_root = os.environ.get("TEST_VLFM_1_PROJECT_ROOT", "/home/nd/vln_edge_task")
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

                from src.experiments.test_vlfm_1.live_vlfm_recorder import (
                    LiveVLFMRecorder,
                )

                test_vlfm_1_result_dir = os.environ.get(
                    "TEST_VLFM_1_DIR",
                    "/home/nd/vln_edge_task/results/test_vlfm_1",
                )
                test_vlfm_1_live = os.environ.get("TEST_VLFM_1_LIVE", "1") != "0"
                test_vlfm_1_save_every = int(os.environ.get("TEST_VLFM_1_SAVE_EVERY", "5"))

                test_vlfm_1_recorder = LiveVLFMRecorder(
                    result_dir=test_vlfm_1_result_dir,
                    live=test_vlfm_1_live,
                    save_every=test_vlfm_1_save_every,
                    window_name="test_vlfm_1_live",
                )

                print("[test_vlfm_1] LiveVLFMRecorder enabled")
                print(f"[test_vlfm_1] result_dir = {test_vlfm_1_result_dir}")
                print(f"[test_vlfm_1] live       = {test_vlfm_1_live}")
                print(f"[test_vlfm_1] save_every = {test_vlfm_1_save_every}")

            except Exception as exc:
                print("[test_vlfm_1][WARN] LiveVLFMRecorder 初始化失败")
                print(repr(exc))
                test_vlfm_1_recorder = None
                ##################################################################################
        while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:
            current_episodes_info = self.envs.current_episodes()

            with inference_mode():
                action_data = self._agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                    action_id = action_data.actions.cpu()[0].item()
                    filepath = os.path.join(
                        os.environ["VLFM_RECORD_ACTIONS_DIR"],
                        "actions.txt",
                    )
                    # If the file doesn't exist, create it
                    if not os.path.exists(filepath):
                        open(filepath, "w").close()
                    with open(filepath, "a") as f:
                        f.write(f"{action_id}\n")

                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    for i, should_insert in enumerate(action_data.should_inserts):
                        if should_insert.item():
                            test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                elif int(next_episodes_info[i].episode_id) == 123123123:
                    envs_to_pause.append(i)

                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)
                 # ============================================================
                # test_vlfm_1 每一步实时可视化
                # ------------------------------------------------------------
                # 插入位置说明：
                # 1. observations / infos 已经是 env.step 后的新观测；
                # 2. infos 已经 update 过 policy_infos；
                # 3. current_episodes_info 保存当前 episode 的 scene_id / episode_id；
                # 4. step_data 是本步真正送入 env.step 的动作；
                # 5. 只显示和保存，不改变 action、不改变地图、不改变策略。
                # ============================================================
                if test_vlfm_1_recorder is not None and i == 0:
                    try:
                        current_episode = current_episodes_info[i]

                        scene_id = getattr(current_episode, "scene_id", "UNKNOWN_SCENE")
                        episode_id = getattr(current_episode, "episode_id", "UNKNOWN_EPISODE")

                        # ============================================================
                        # 1. 汇总本步所有可用的调试信息
                        # ------------------------------------------------------------
                        # action_data.policy_info：policy 原始额外信息
                        # infos[i]：Habitat 返回信息，前面已经 update 过 policy_infos[i]
                        #
                        # 当前我们已经确认 infos/policy 中存在：
                        #   top_down_map
                        #   render_below_images
                        #   target_object
                        #   target_detected
                        #   stop_called
                        #   success / spl / distance_to_goal
                        # ============================================================
                        policy_info_for_recorder = {}

                        if isinstance(action_data.policy_info, dict):
                            policy_info_for_recorder.update(action_data.policy_info)

                        if isinstance(infos[i], dict):
                            policy_info_for_recorder.update(infos[i])

                        # ============================================================
                        # 2. 强制写一个 trainer 侧 debug 文件
                        # ------------------------------------------------------------
                        # 这个文件不依赖 live_vlfm_recorder.py。
                        # 如果这个文件都没有，说明这个 if 块没有执行。
                        # ============================================================
                        debug_dir = os.path.join(
                            os.environ.get(
                                "TEST_VLFM_1_DIR",
                                "/home/nd/vln_edge_task/results/test_vlfm_1",
                            ),
                            "logs",
                        )
                        os.makedirs(debug_dir, exist_ok=True)

                        trainer_debug_path = os.path.join(
                            debug_dir,
                            "trainer_policy_info_debug_once.txt",
                        )

                        if not os.path.exists(trainer_debug_path):
                            with open(trainer_debug_path, "w", encoding="utf-8") as f:
                                f.write("===== trainer policy_info_for_recorder keys and types =====\n")
                                for key in sorted(policy_info_for_recorder.keys()):
                                    value = policy_info_for_recorder[key]
                                    f.write(f"{key}: type={type(value)}\n")

                                    if isinstance(value, dict):
                                        f.write(f"  dict_keys={list(value.keys())}\n")

                                    if isinstance(value, (list, tuple)):
                                        f.write(f"  len={len(value)}\n")
                                        if len(value) > 0:
                                            f.write(f"  first_type={type(value[0])}\n")
                                            if hasattr(value[0], "shape"):
                                                f.write(f"  first_shape={getattr(value[0], 'shape', None)}\n")

                                    if hasattr(value, "shape"):
                                        f.write(f"  shape={getattr(value, 'shape', None)}\n")
                                        f.write(f"  dtype={getattr(value, 'dtype', None)}\n")

                        # ============================================================
                        # 3. 目标类别 target
                        # ============================================================
                        target = policy_info_for_recorder.get("target_object", None)

                        if isinstance(target, (list, tuple)) and len(target) > 0:
                            target = target[0]

                        if target is None:
                            target = os.environ.get("TEST_VLFM_1_TARGET", None)

                        if target is None:
                            target = "UNKNOWN_TARGET"

                        target = str(target)

                        # ============================================================
                        # 4. 当前状态 mode
                        # ============================================================
                        mode = (
                            policy_info_for_recorder.get("mode")
                            or policy_info_for_recorder.get("nav_mode")
                            or policy_info_for_recorder.get("state")
                        )

                        if mode is None:
                            try:
                                stop_called = float(policy_info_for_recorder.get("stop_called", 0))
                            except Exception:
                                stop_called = 0.0

                            try:
                                target_detected = float(policy_info_for_recorder.get("target_detected", 0))
                            except Exception:
                                target_detected = 0.0

                            if stop_called > 0:
                                mode = "stop"
                            elif target_detected > 0:
                                mode = "target_detected"
                            else:
                                mode = "navigate"

                        mode = str(mode)

                        # ============================================================
                        # 5. 显式传入地图和值图
                        # ------------------------------------------------------------
                        # 左下角：
                        #   top_down_map
                        #
                        # 右下角：
                        #   render_below_images
                        #
                        # 注意：
                        #   这里直接传给 recorder 的 obstacle_map / value_map 参数，
                        #   不再依赖 recorder 自己从 policy_info 里猜字段。
                        # ============================================================
                        obstacle_map_for_recorder = policy_info_for_recorder.get("top_down_map", None)
                        value_map_for_recorder = policy_info_for_recorder.get("render_below_images", None)

                        # ============================================================
                        # 6. 动作
                        # ============================================================
                        action_for_recorder = step_data[i] if i < len(step_data) else "UNKNOWN_ACTION"

                        # ============================================================
                        # 7. 更新实时可视化 recorder
                        # ============================================================
                        test_vlfm_1_recorder.update(
                            step=test_vlfm_1_step,
                            observations=observations[i],
                            infos=infos[i],
                            policy_info=policy_info_for_recorder,
                            action=action_for_recorder,
                            mode=mode,
                            target=target,
                            scene_id=scene_id,
                            episode_id=episode_id,
                            obstacle_map=obstacle_map_for_recorder,
                            value_map=value_map_for_recorder,
                        )

                        test_vlfm_1_step += 1

                    except Exception as exc:
                        # 可视化不能影响主流程。
                        # 即使 recorder 出错，VLFM 也继续跑。
                        print("[test_vlfm_1][WARN] recorder.update failed")
                        print(repr(exc))
######################################################################################################################
                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_total += 1
                    print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")

                    from vlfm.utils.episode_stats_logger import (
                        log_episode_stats,
                    )

                    try:
                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
                            infos[i],
                        )
                    except Exception:
                        failure_cause = "Unknown"

                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        rgb_frames[i] = hab_vis.flush_frames(failure_cause)
                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()

        if "ZSOS_DONE_PATH" in os.environ:
            # Create an empty file at ZSOS_DONE_PATH to signal that the
            # evaluation is done
            done_path = os.environ["ZSOS_DONE_PATH"]
            with open(done_path, "w") as f:
                f.write("")

        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
         # ============================================================
        # test_vlfm_1 保存最终指标并关闭实时窗口
        # ============================================================
        if test_vlfm_1_recorder is not None:
            try:
                final_metrics = dict(metrics)
                final_metrics["average_reward"] = float(aggregated_stats.get("reward", 0.0))
                final_metrics["num_finished_episodes"] = len(stats_episodes)
                final_metrics["number_of_eval_episodes"] = number_of_eval_episodes
                final_metrics["evals_per_ep"] = evals_per_ep
                test_vlfm_1_recorder.close(final_metrics=final_metrics)
                print("[test_vlfm_1] recorder closed")
            except Exception as exc:
                print("[test_vlfm_1][WARN] recorder.close failed")
                print(repr(exc))
                ################################################################################
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
