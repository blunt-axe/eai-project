from typing import Any, Optional, Sequence, Tuple, Union
import grasp_cube.agents.robots.so101.so_101_ee

import numpy as np
import sapien
import torch
import torch.nn.functional as F
import gymnasium as gym

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.agents import register_agent
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array
from mani_skill.agents import MultiAgent
from mani_skill.agents.registration import REGISTERED_AGENTS

import sapien
import numpy as np
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent

from mani_skill.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_configs,
    update_camera_configs_from_dict,
)

from transforms3d.euler import euler2quat

from so101_lift_cube import PickCubeSO101Env


from utils.builder import (
    build_cube_w_friction,
    build_box_w_friction,
)

@register_env("SortCubeSO101-v0", max_episode_steps=300)
class SortCubeSO101Env(PickCubeSO101Env):
    def __init__(self, *args, robot_uids="so101", **kwargs):
        # 1. Define constants FIRST
        self.RED_SIZE = 0.03
        self.GREEN_SIZE = 0.03
        self.MIN_SEPARATION = 0.05
        self.CENTER_X, self.CENTER_Y = 0.29, 0.26
        self.RAND_RANGE = 0.06 

        # 2. Then call super().__init__
        self.robot_uids = (robot_uids, robot_uids) if isinstance(robot_uids, str) else robot_uids
        super().__init__(*args, robot_uids=self.robot_uids, **kwargs)
        
        if not hasattr(self, "left_arm_rest_qpos"):
            rest_np = self.agent.agents[0].keyframes["rest"].qpos
            self.left_arm_rest_qpos = torch.from_numpy(rest_np).float().to(self.device)
        
        if not hasattr(self, "right_arm_rest_qpos"):
            rest_np = self.agent.agents[1].keyframes["rest"].qpos
            self.right_arm_rest_qpos = torch.from_numpy(rest_np).float().to(self.device)

    def _load_agent(self, options: dict):
        super()._load_agent(options, [
            Pose.create_from_pq(p=[0.481, 0.025, 0], q=[np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2]),
            Pose.create_from_pq(p=[0.119, 0.025, 0], q=[np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2]),
        ])
        # agent_poses = [
        #     sapien.Pose(p=[0.481, 0.025, 0], q=[np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2]),
        #     sapien.Pose(p=[0.119, 0.025, 0], q=[np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2])
        # ]
        # agents = []
        # for i, uid in enumerate(self.robot_uids):
        #     if uid not in REGISTERED_AGENTS:
        #         raise RuntimeError(f"❌ 机器人 '{uid}' 未注册! 请检查 import 路径。")
                
        #     original_agent_cls = REGISTERED_AGENTS[uid].agent_cls
        #     unique_uid = f"{uid}_{i}"
        #     SafeAgentCls = type(f"Safe_{uid}_{i}", (original_agent_cls,), {"uid": unique_uid})
            
        #     agent = SafeAgentCls(
        #         scene=self.scene,
        #         control_freq=self.control_freq,
        #         control_mode="pd_joint_delta_pos",
        #         initial_pose=agent_poses[i]
        #     )
        #     agents.append(agent)
        
        # self.agent = MultiAgent(agents)

    def _load_scene(self, options: dict):
        # Build table and red cube from Lift class
        super()._load_scene(options)

        # Add the Green Cube
        self.green_half = self.GREEN_SIZE / 2
        self.green_cube = build_cube_w_friction(
            self.scene, 
            half_size=self.green_half, 
            color=[0, 1, 0, 1], 
            name="target_green_cube", 
            body_type="dynamic",
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.1
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        
        b = len(env_idx)
        with torch.device(self.device):
            green_xy = (torch.rand((b, 2)) * 2 - 1) * self.RAND_RANGE + torch.tensor([self.CENTER_X, self.CENTER_Y])
            
            self.green_cube.set_pose(Pose.create_from_pq(
                p=torch.cat([green_xy, torch.ones((b, 1)) * self.green_half], dim=-1), 
                q=torch.tensor([1, 0, 0, 0]).repeat(b, 1)
            ))

            # Ensure Red Cube is not overlapping Green Cube
            red_xyz = torch.zeros((b, 3))
            red_xyz[:, 2] = torch.ones((b,)) * self.RED_SIZE / 2
            for i in range(b):
                while True:
                    cand_xy = (torch.rand(2) * 2 - 1) * self.RAND_RANGE + torch.tensor([self.CENTER_X, self.CENTER_Y])
                    if torch.linalg.norm(cand_xy - green_xy[i]) > self.MIN_SEPARATION:
                        red_xyz[i, :2] = cand_xy
                        break

            yaw = (torch.rand(b) * 2 - 1) * torch.pi
            half_yaw = yaw * 0.5
            qs = torch.zeros((b, 4))
            qs[:, 0] = torch.cos(half_yaw)  # w
            qs[:, 3] = torch.sin(half_yaw)  # z
            
            self.red_cube.set_pose(Pose.create_from_pq(p=red_xyz, q=qs))

            if not hasattr(self, "green_cube_init_xy"):
                self.green_cube_init_xy = torch.zeros([self.num_envs, 2], dtype=torch.float32)

            if not hasattr(self, "red_cube_init_xy"):
                self.red_cube_init_xy = torch.zeros([self.num_envs, 2], dtype=torch.float32)
            
            self.green_cube_init_xy[env_idx] = green_xy
            self.red_cube_init_xy[env_idx] = red_xyz[..., :2]

    def _get_obs_extra(self, info: dict):
        """
        额外观测函数
        """
        # 基础可观测信息（非特权信息，常驻返回，无开关）
        tcp_left = self.agent.agents[0].tcp_pose.raw_pose
        tcp_right = self.agent.agents[1].tcp_pose.raw_pose
        tcp_pose = torch.cat([tcp_left, tcp_right], dim=-1)

        is_left_grasping = self.agent.agents[0].is_grasping(self.red_cube).float().unsqueeze(-1)
        is_right_grasping = self.agent.agents[1].is_grasping(self.green_cube).float().unsqueeze(-1)
        is_grasping = torch.cat([is_left_grasping, is_right_grasping], dim=-1)

        obs = dict(
            tcp_pose=tcp_pose,
        )

        # 特权信息（通过use_state开关控制，满足条件时追加到观测字典）
        if self.obs_mode_struct.use_state:
            # 获取方块绝对位姿（特权信息）
            red_cube_pose = self.red_cube.pose.raw_pose
            green_cube_pose = self.green_cube.pose.raw_pose
            # 计算所有机械臂-方块的相对位置（特权信息）
            rel_pos_left_red = red_cube_pose[:, :3] - tcp_left[:, :3]
            rel_pos_right_green = green_cube_pose[:, :3] - tcp_right[:, :3]
            rel_pos_left_green = green_cube_pose[:, :3] - tcp_left[:, :3]
            rel_pos_right_red = red_cube_pose[:, :3] - tcp_right[:, :3]
            
            # 追加所有特权信息到观测字典
            obs.update(
                is_grasping=is_grasping,
                obj_pose=torch.cat([red_cube_pose, green_cube_pose], dim=-1),
                rel_pose=torch.cat([
                    rel_pos_left_red,
                    rel_pos_right_green,
                    rel_pos_left_green,
                    rel_pos_right_red
                ], dim=-1)
            )

        return obs
    
    def step(self, action: Union[None, np.ndarray, torch.Tensor, dict]):

        # # --- 课程学习 ---
        # # 只训练 Red Cube任务
        if isinstance(action, dict):
            raw_uid = self.agent.agents[0].uid  # "so101"
            arm_key = f"{raw_uid}-0"       # "so101-0"
            
            # for key in action.keys():
            #     if key != arm_key:
            #         val = action[key]
            #         if isinstance(val, torch.Tensor):
            #             action[key] = torch.zeros_like(val)
            #         elif isinstance(val, np.ndarray):
            #             action[key] = np.zeros_like(val)
        
        # # --- 课程学习：阶段 2 ---
        # # 只训练 Green Cube任务
        # if isinstance(action, dict):
        #     raw_uid = self.agent.agents[0].uid  # "so101"
        #     arm_key = f"{raw_uid}-1"       # "so101-1"
            
        #     for key in action.keys():
        #         if key != arm_key:
        #             val = action[key]
        #             if isinstance(val, torch.Tensor):
        #                 action[key] = torch.zeros_like(val)
        #             elif isinstance(val, np.ndarray):
        #                 action[key] = np.zeros_like(val)

        action = self._step_action(action)
        self._elapsed_steps += 1
        info = self.get_info()
        obs = self.get_obs(info, unflattened=True)
        reward = self.get_reward(obs=obs, action=action, info=info)
        obs = self._flatten_raw_obs(obs)
        if "success" in info:
            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        self._last_obs = obs

        # # 2. 强制覆盖物理状态
        # with torch.device(self.device):
        #     # 直接操作右手的 robot 对象
        #     right_robot = self.agent.agents[1].robot
            
        #     # 构造 Batch 数据
        #     batch_rest_qpos = self.right_arm_rest_qpos.unsqueeze(0).repeat(self.num_envs, 1)
            
        #     # 强制设置位置和速度
        #     right_robot.set_qpos(batch_rest_qpos)
        #     right_robot.set_qvel(batch_rest_qpos * 0.0)

        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )
    
    def evaluate(self):
        red_x = self.red_cube.pose.p[:, 0]
        red_y = self.red_cube.pose.p[:, 1]
        green_x = self.green_cube.pose.p[:, 0]
        green_y = self.green_cube.pose.p[:, 1]

        # red_success = (red_x >= self.x3) & (red_x <= self.x4) & \
        #             (red_y >= self.y1) & (red_y <= self.y2)

        # green_success = (green_x >= self.x1) & (green_x <= self.x2) & \
        #                 (green_y >= self.y1) & (green_y <= self.y2)

        target_z = 0.05
        red_target_pos = torch.tensor([0.48, 0.26, target_z], device=self.device)
        green_target_pos = torch.tensor([0.11, 0.26, target_z], device=self.device)
        
        red_dist = torch.linalg.norm(self.red_cube.pose.p[:, :2] - red_target_pos[:2], dim=1)
        green_dist = torch.linalg.norm(self.green_cube.pose.p[:, :2] - green_target_pos[:2], dim=1)

        radius = 0.06
        red_success = red_dist <= radius
        green_success = green_dist <= radius

        return {
            "red_success": red_success,
            "green_success": green_success,
            "success": red_success & green_success
        }

    # def compute_dense_reward(self, obs: any, action: Array, info: dict):
    #     target_z = 0.05
    #     red_target_pos = torch.tensor([0.48, 0.26, target_z], device=self.device)
    #     green_target_pos = torch.tensor([0.11, 0.26, target_z], device=self.device)

    #     def compute_single_arm_reward(agent, cube, goal_pos):
    #         f1_pose = agent.finger1_tip.pose
    #         f2_pose = agent.finger2_tip.pose
    #         tcp_pose = agent.tcp_pose
    #         cube_pose = cube.pose

    #         is_grasping = agent.is_grasping(cube).float()
            
    #         vec_1 = f1_pose.p - cube_pose.p
    #         vec_2 = cube_pose.p - f2_pose.p
    #         cos_sim = F.cosine_similarity(vec_1, vec_2, dim=-1)
    #         norm_cos_sim = (cos_sim + 1) / 2
            
    #         qvel = agent.robot.get_qvel()[..., :-1]
    #         vel = torch.linalg.norm(qvel, axis=1)
            
    #         dist = torch.linalg.norm(tcp_pose.p - cube_pose.p, axis=1)
            
    #         grasp = is_grasping * (1 - torch.tanh((1.0 - norm_cos_sim).clamp(min=0.0)))
    #         slow = is_grasping * (1 - torch.tanh(vel))
    #         goal_dist = torch.linalg.norm(cube_pose.p - goal_pos, axis=1)
    #         lift = is_grasping * (1 - torch.tanh(5 * goal_dist)) * 2.0
    #         reaching = (1 - torch.tanh(5 * dist)) * (1 - is_grasping)
    #         gripper_width = torch.linalg.norm(f1_pose.p - f2_pose.p, axis=1)
    #         is_close = (dist < 0.03).float()
    #         caging = is_close * (1.0 - torch.tanh(20.0 * (gripper_width - 0.03).clamp(min=0.0))) * 0.5 * (1 - is_grasping)
            
    #         stay_dist = torch.linalg.norm(tcp_pose.p - goal_pos, axis=1)
    #         stay = 1 - torch.tanh(5 * stay_dist)

    #         return is_grasping, stay, reaching + grasp + lift

    #     left_is_grasping, _, left_other = compute_single_arm_reward(self.agent.agents[0], self.red_cube, red_target_pos)
    #     _, right_stay, right_other = compute_single_arm_reward(self.agent.agents[1], self.green_cube, green_target_pos)
        
    #     # right_coef = left_is_grasping * 0.8 + 0.2

    #     total_reward = left_other + left_is_grasping * right_other + (1 - left_is_grasping) * right_stay + left_is_grasping
        
    #     # print(f"left: {left_other.mean():.3f}, "
    #     #       f"left_ig: {left_is_grasping.mean():.3f}, "
    #     #       f"right(move): {(left_is_grasping * right_other).mean():.3f}, "
    #     #       f"right(stay): {((1 - left_is_grasping) * right_stay).mean():.3f}")

    #     if "sucess" in info:
    #         total_reward[info["success"]] = 3.0

    #     return total_reward
    
    def compute_dense_reward(self, obs: any, action: Array, info: dict):
        target_z = 0.05
        red_target_pos = torch.tensor([0.48, 0.26, target_z], device=self.device)
        green_target_pos = torch.tensor([0.11, 0.26, target_z], device=self.device)

        def compute_single_arm_reward(agent, cube, goal_pos):
            f1_pose = agent.finger1_tip.pose
            f2_pose = agent.finger2_tip.pose
            tcp_pose = agent.tcp_pose
            cube_pose = cube.pose

            is_grasping = agent.is_grasping(cube)
            
            vec_1 = f1_pose.p - cube_pose.p
            vec_2 = cube_pose.p - f2_pose.p
            cos_sim = F.cosine_similarity(vec_1, vec_2, dim=-1)
            norm_cos_sim = (cos_sim + 1) / 2

            robot_qpos = agent.robot.get_qpos()
            
            dist = torch.linalg.norm(tcp_pose.p - cube_pose.p, dim=1)
            goal_dist = torch.linalg.norm(cube_pose.p - goal_pos, dim=1)
            qpos_dist = torch.linalg.norm(robot_qpos[..., :-1] - self.left_arm_rest_qpos[..., :-1], dim=1)
            
            reaching = info["red_success"] + (~info["red_success"] & ~is_grasping) * (1 - torch.tanh(5 * dist)) 
            
            grasp = (~info["red_success"] & is_grasping) * (1 - torch.tanh((0.9 - norm_cos_sim).clamp(min=0.0)))

            restore = is_grasping * (1 - torch.tanh(5 * goal_dist)) \
            + (~is_grasping) * info["red_success"] * (1 - torch.tanh(0.5 * qpos_dist))

            release = info["red_success"] * (1 - torch.tanh(1.5 * (torch.abs(robot_qpos[..., -1] - 0.60) - 0.05).clamp(min=0.00)))

            default = 1 - torch.tanh(0.5 * qpos_dist)

            return reaching, grasp, 2.0 * restore, 4.0 * release, 0.1 * default

        l_reach, l_grasp, l_restore, l_release, l_default = compute_single_arm_reward(self.agent.agents[0], self.red_cube, red_target_pos)
        
        green_stay = 2.0 * (1 - torch.tanh(5 * torch.linalg.norm(self.green_cube_init_xy - self.green_cube.pose.p[..., :2], dim=1)))

        total_reward = l_reach + l_grasp + l_restore + l_release + l_default + green_stay

        #print(f"green={green_stay.mean():.4f}, reach={l_reach.mean():.4f}, grasp={l_grasp.mean():.4f} \
        #restore={l_restore.mean():.4f}, release={l_release.mean():.4f}, default={l_default.mean():.4f}")

        if "success" in info:
            total_reward[info["success"]] = 7.0

        return total_reward
        
    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 9.0















































































































































































































































































@register_env("SortCubeSO101-v1", max_episode_steps=200)
class SortCubeSO101EnvGreenCube(SortCubeSO101Env):
    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        
        b = len(env_idx)
        with torch.device(self.device):
            # Randomize other arm qpos
            self.agent.agents[0].robot.set_qpos((torch.rand((b, 6)) * 2 - 1) * 0.20 + torch.tensor(self.agent.agents[0].keyframes["rest"].qpos))

            # Randomize Green Cube
            green_xy = (torch.rand((b, 2)) * 2 - 1) * self.RAND_RANGE * 1.2 + torch.tensor([self.CENTER_X, self.CENTER_Y], device=self.device)
            red_xy = (torch.rand((b, 2)) * 2 - 1) * self.RAND_RANGE + torch.tensor([(self.x3 + self.x4) / 2, (self.y1 + self.y2) / 2], device=self.device)

            green_xyz = torch.cat([green_xy, torch.ones((b, 1)) * self.green_half], dim=-1)
            red_xyz = torch.cat([red_xy, torch.ones((b, 1)) * self.RED_SIZE / 2], dim=-1)

            def random_rotate_quat():
                yaw = (torch.rand(b) * 2 - 1) * torch.pi
                half_yaw = yaw * 0.5
                qs = torch.zeros((b, 4), device=self.device)
                qs[:, 0] = torch.cos(half_yaw)  # w
                qs[:, 3] = torch.sin(half_yaw)  # z
                return qs

            self.green_cube.set_pose(Pose.create_from_pq(
                p=green_xyz, 
                q=random_rotate_quat()
            ))
            
            self.red_cube.set_pose(Pose.create_from_pq(
                p=red_xyz, 
                q=random_rotate_quat()
            ))
    
    def step(self, action: Union[None, np.ndarray, torch.Tensor, dict]):
        """
        Phase 2
        """
        # # --- 课程学习：阶段 1 ---
        # # 只训练 Red Cube任务
        # if isinstance(action, dict):
        #     raw_uid = self.agent.agents[0].uid  # "so101"
        #     arm_key = f"{raw_uid}-0"       # "so101-0"
            
        #     for key in action.keys():
        #         if key != arm_key:
        #             val = action[key]
        #             if isinstance(val, torch.Tensor):
        #                 action[key] = torch.zeros_like(val)
        #             elif isinstance(val, np.ndarray):
        #                 action[key] = np.zeros_like(val)
        
        # --- 课程学习：阶段 2 ---
        # 只训练 Green Cube任务
        if isinstance(action, dict):
            raw_uid = self.agent.agents[0].uid  # "so101"
            arm_key = f"{raw_uid}-1"       # "so101-1"
            
            for key in action.keys():
                if key != arm_key:
                    val = action[key]
                    if isinstance(val, torch.Tensor):
                        action[key] = (torch.rand_like(val) * 2 - 1) * 0.20 # 添加抖动
                    elif isinstance(val, np.ndarray):
                        action[key] = (np.rand_like(val) * 2 - 1) * 0.20 # 添加抖动

        action = self._step_action(action)
        self._elapsed_steps += 1
        info = self.get_info()
        obs = self.get_obs(info, unflattened=True)
        reward = self.get_reward(obs=obs, action=action, info=info)
        obs = self._flatten_raw_obs(obs)
        if "success" in info:
            if "fail" in info:
                terminated = torch.logical_or(info["success"], info["fail"])
            else:
                terminated = info["success"].clone()
        else:
            if "fail" in info:
                terminated = info["fail"].clone()
            else:
                terminated = torch.zeros(self.num_envs, dtype=bool, device=self.device)
        self._last_obs = obs
        
        # # 强制覆盖物理状态
        # with torch.device(self.device):
        #     # 直接操作 robot 0
        #     robot = self.agent.agents[0].robot
            
        #     batch_rest_qpos = self.right_arm_rest_qpos.unsqueeze(0).repeat(self.num_envs, 1)
            
        #     robot.set_qpos(batch_rest_qpos)
        #     robot.set_qvel(batch_rest_qpos * 0.0)

        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )
        
    # def compute_dense_reward(self, obs: any, action: Array, info: dict):
    #     """
    #     Phase 2
    #     """
    #     target_z = 0.05
    #     red_target_pos = torch.tensor([0.48, 0.26, target_z], device=self.device)
    #     green_target_pos = torch.tensor([0.11, 0.26, target_z], device=self.device)

    #     def compute_single_arm_reward(agent, cube, goal_pos):
    #         f1_pose = agent.finger1_tip.pose
    #         f2_pose = agent.finger2_tip.pose
    #         tcp_pose = agent.tcp_pose
    #         cube_pose = cube.pose

    #         is_grasping = agent.is_grasping(cube).float()
            
    #         vec_1 = f1_pose.p - cube_pose.p
    #         vec_2 = cube_pose.p - f2_pose.p
    #         cos_sim = F.cosine_similarity(vec_1, vec_2, dim=-1)
    #         norm_cos_sim = (cos_sim + 1) / 2
            
    #         qvel = agent.robot.get_qvel()[..., :-1]
    #         vel = torch.linalg.norm(qvel, axis=1)
            
    #         dist = torch.linalg.norm(tcp_pose.p - cube_pose.p, axis=1)
            
    #         grasp = is_grasping * (1 - torch.tanh((1.0 - norm_cos_sim).clamp(min=0.0)))
    #         slow = is_grasping * (1 - torch.tanh(vel))
    #         goal_dist = torch.linalg.norm(cube_pose.p - goal_pos, axis=1)
    #         lift = is_grasping * (1 - torch.tanh(5 * goal_dist)) * 2.0
    #         reaching = (1 - torch.tanh(5 * dist)) * (1 - is_grasping)
    #         gripper_width = torch.linalg.norm(f1_pose.p - f2_pose.p, axis=1)
    #         is_close = (dist < 0.03).float()
    #         caging = is_close * (1.0 - torch.tanh(20.0 * (gripper_width - 0.03).clamp(min=0.0))) * 0.5 * (1 - is_grasping)
            
    #         stay_dist = torch.linalg.norm(tcp_pose.p - goal_pos, axis=1)
    #         stay = 1 - torch.tanh(5 * stay_dist)

    #         return is_grasping, stay, reaching + grasp + lift

    #     _, _, right_other = compute_single_arm_reward(self.agent.agents[1], self.green_cube, green_target_pos)
        
    #     total_reward = right_other

    #     print(f"right: {right_other.mean()}")

    #     if "success" in info:
    #         total_reward[info["success"]] = 2.0

    #     return total_reward

    def compute_dense_reward(self, obs: any, action: Array, info: dict):
        target_z = 0.05
        red_target_pos = torch.tensor([0.48, 0.26, target_z], device=self.device)
        green_target_pos = torch.tensor([0.11, 0.26, target_z], device=self.device)

        def compute_single_arm_reward(agent, cube, goal_pos):
            f1_pose = agent.finger1_tip.pose
            f2_pose = agent.finger2_tip.pose
            tcp_pose = agent.tcp_pose
            cube_pose = cube.pose

            is_grasping = agent.is_grasping(cube)
            
            vec_1 = f1_pose.p - cube_pose.p
            vec_2 = cube_pose.p - f2_pose.p
            cos_sim = F.cosine_similarity(vec_1, vec_2, dim=-1)
            norm_cos_sim = (cos_sim + 1) / 2

            robot_qpos = agent.robot.get_qpos()
            
            dist = torch.linalg.norm(tcp_pose.p - cube_pose.p, dim=1)
            goal_dist = torch.linalg.norm(cube_pose.p - goal_pos, dim=1)
            qpos_dist = torch.linalg.norm(robot_qpos[..., :-1] - self.left_arm_rest_qpos[..., :-1], dim=1)
            
            reaching = info["green_success"] + (~info["green_success"] & ~is_grasping) * (1 - torch.tanh(5 * dist)) 
            
            grasp = (~info["green_success"] & is_grasping) * (1 - torch.tanh((0.9 - norm_cos_sim).clamp(min=0.0)))

            restore = is_grasping * (1 - torch.tanh(5 * goal_dist)) \
            + (~is_grasping) * info["green_success"] * (1 - torch.tanh(0.5 * qpos_dist))

            release = info["green_success"] * (1 - torch.tanh(1.5 * (torch.abs(robot_qpos[..., -1] - 0.60) - 0.05).clamp(min=0.00)))

            default = 1 - torch.tanh(0.5 * qpos_dist)

            return reaching, grasp, 2.0 * restore, 4.0 * release, 0.1 * default

        r_reach, r_grasp, r_restore, r_release, r_default = compute_single_arm_reward(self.agent.agents[1], self.green_cube, green_target_pos)

        total_reward = r_reach + r_grasp + r_restore + r_release + r_default

        #print(f"reach={r_reach.mean():.4f}, grasp={r_grasp.mean():.4f}, restore={r_restore.mean():.4f}, release={r_release.mean():.4f}, default={r_default.mean():.4f}")

        if "success" in info:
            total_reward[info["success"]] = 6.0

        return total_reward
        
    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 7.0