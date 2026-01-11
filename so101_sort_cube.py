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

from utils.distort import get_distorted_image_tensor

from mani_skill.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_configs,
    update_camera_configs_from_dict,
)

import constants as C

from transforms3d.euler import euler2quat

from so101_lift_cube import PickCubeSO101Env


from utils.builder import (
    build_cube_w_friction,
    build_box_w_friction,
)

@register_env("SortCubeSO101-v0", max_episode_steps=200)
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
            # Randomize Green Cube
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
            qs = torch.zeros((b, 4), device=self.device)
            qs[:, 0] = torch.cos(half_yaw)  # w
            qs[:, 3] = torch.sin(half_yaw)  # z
            
            self.red_cube.set_pose(Pose.create_from_pq(p=red_xyz, q=qs))

    def _get_obs_extra(self, info: dict):
        # 1. 获取 TCP (末端) 位姿
        tcp_left = self.agent.agents[0].tcp_pose.raw_pose
        tcp_right = self.agent.agents[1].tcp_pose.raw_pose
        tcp_pose = torch.cat([tcp_left, tcp_right], dim=-1)

        # 2. 获取 Is Grasping
        is_left_grasping = self.agent.agents[0].is_grasping(self.red_cube).float().unsqueeze(-1)
        is_right_grasping = self.agent.agents[1].is_grasping(self.green_cube).float().unsqueeze(-1)
        is_grasping = torch.cat([is_left_grasping, is_right_grasping], dim=-1)

        # 3. 获取方块位姿
        red_cube_pose = self.red_cube.pose.raw_pose
        green_cube_pose = self.green_cube.pose.raw_pose
        
        # ----------------------------------------------------------------------
        # [关键修复] 定义所有需要的相对位置变量
        # ----------------------------------------------------------------------
        # 正确的目标对：左手->红，右手->绿
        rel_pos_left_red   = red_cube_pose[:, :3] - tcp_left[:, :3]
        rel_pos_right_green = green_cube_pose[:, :3] - tcp_right[:, :3]
        
        # 交叉干扰对：左手->绿，右手->红 (让网络知道这些是不对的)
        rel_pos_left_green = green_cube_pose[:, :3] - tcp_left[:, :3]
        rel_pos_right_red  = red_cube_pose[:, :3] - tcp_right[:, :3]

        # 4. 组装 Observation
        obs = dict(
            tcp_pose=tcp_pose,
            is_grasping=is_grasping,
            # 绝对位置
            obj_pose = torch.cat([red_cube_pose, green_cube_pose], dim=-1),
            # 相对位置 (把上面定义的4个变量都放进去)
            rel_pose = torch.cat([
                rel_pos_left_red, 
                rel_pos_right_green,
                rel_pos_left_green,
                rel_pos_right_red
            ], dim=-1) 
        )
        return obs
    
    def step(self, action: Union[None, np.ndarray, torch.Tensor, dict]):
        # --- 课程学习：阶段 1 ---
        # 强制冻结右手（Green Cube任务），只训练左手（Red Cube任务）
        # 右手保持 Reset 时的姿态（Action = 0 或特定值）
        # 假设 action 是 (Batch, 14) 的 Tensor
        if isinstance(action, dict):
            # 1. 动态构建正确的左手 Key
            # 根据你的 Log，Gym 会自动添加 "-0" 后缀
            raw_uid = self.agent.agents[0].uid  # "so101"
            left_arm_key = f"{raw_uid}-0"       # "so101-0"
            
            # 2. 遍历所有动作
            for key in action.keys():
                # 3. 只有当 Key 完全等于左手 Key 时才放行
                # 其他所有 Key (包括 so101-1) 全部置零
                if key != left_arm_key:
                    val = action[key]
                    if isinstance(val, torch.Tensor):
                        action[key] = torch.zeros_like(val)
                    elif isinstance(val, np.ndarray):
                        action[key] = np.zeros_like(val)
                    # print(f"Frozen: {key} (Target was {left_arm_key})", flush=True)
        """
        Take a step through the environment with an action. Actions are automatically clipped to the action space.

        If ``action`` is None, the environment will proceed forward in time without sending any actions/control signals to the agent
        """
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
        if not hasattr(self, "right_arm_rest_qpos"):
            rest_np = self.agent.agents[1].keyframes["rest"].qpos
            self.right_arm_rest_qpos = torch.from_numpy(rest_np).float().to(self.device)

        # 2. 强制覆盖物理状态
        with torch.device(self.device):
            # 直接操作右手的 robot 对象
            right_robot = self.agent.agents[1].robot
            
            # 构造 Batch 数据
            batch_rest_qpos = self.right_arm_rest_qpos.unsqueeze(0).repeat(self.num_envs, 1)
            
            # 强制设置位置和速度
            right_robot.set_qpos(batch_rest_qpos)
            right_robot.set_qvel(batch_rest_qpos * 0.0)
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

        red_success = (red_x >= self.x3) & (red_x <= self.x4) & \
                    (red_y >= self.y1) & (red_y <= self.y2)

        green_success = (green_x >= self.x1) & (green_x <= self.x2) & \
                        (green_y >= self.y1) & (green_y <= self.y2)

        return {
            "red_success": red_success,
            "green_success": green_success,
            "success": red_success & green_success
        }

    def compute_dense_reward(self, obs: any, action: Array, info: dict):
        # ---------------------------------------------------------------------
        # 1. 定义目标位置 (Red -> 左边, Green -> 右边)
        # ---------------------------------------------------------------------
        # 假设目标高度为 0.05 (稍微抬起一点，放到垫子上)
        target_z = 0.05
        # 这里的坐标参考之前的配置，Red 在 0.48 左右，Green 在 0.11 左右
        red_target_pos = torch.tensor([0.48, 0.26, target_z], device=self.device)
        green_target_pos = torch.tensor([0.11, 0.26, target_z], device=self.device)

        # ---------------------------------------------------------------------
        # 2. 定义完全复刻 Lift 的单臂奖励函数
        # ---------------------------------------------------------------------
        def compute_single_arm_reward(agent, cube, goal_pos):
            f1_pose = agent.finger1_tip.pose
            f2_pose = agent.finger2_tip.pose
            tcp_pose = agent.tcp_pose
            cube_pose = cube.pose

            is_grasping = agent.is_grasping(cube).float()

            vec_1 = f1_pose.p - cube_pose.p
            vec_2 = cube_pose.p - f2_pose.p
            cos_sim = F.cosine_similarity(vec_1, vec_2, dim=-1)
            norm_cos_sim = (cos_sim + 1) / 2
            
            # 获取该手臂的速度 (切片可能需要根据实际 action 维度调整，这里简化用 robot 整体或 TCP 速度)
            # 为了完全对齐 Lift 逻辑，这里尝试获取 robot qvel。
            # 注意：在 MultiAgent 下，agent.robot 可能指向同一个 robot 实例。
            # 如果是分开的 robot 实例则没问题。如果是同一个，建议用 tcp 速度代替 qvel。
            # 这里为了稳健，改用 TCP 速度 (linear + angular)，逻辑是一样的：希望它稳。
            # 如果你确定 agent.robot.get_qvel() 是分开的，可以用原版。这里用 TCP 速度模拟原版 qvel 惩罚。
            # qvel = agent.robot.get_qvel()[:, :-1] 
            # vel = torch.linalg.norm(qvel, axis=1) 
            # ↓ 替换为 TCP 速度以适配双臂环境，物理含义一致：
            tcp_vel = agent.tcp_pose.p - agent.tcp_pose.p # 占位，实际应该用 get_velocities
            # 简化：直接惩罚上一帧的动作幅度或者不需要太严格的 qvel，
            # 或者直接用 robot.get_qvel() 也没问题，只要它包含了这个手臂的关节。
            qvel = agent.robot.get_qvel() # 获取全身速度
            vel = torch.linalg.norm(qvel, axis=1) # 全身慢下来

            dist = torch.linalg.norm(tcp_pose.p - cube_pose.p, axis=1)
            # --- [新增] 姿态奖励 (Orientation Reward) ---
            # 获取 TCP 的 Z 轴方向
            tcp_z = agent.tcp_pose.to_transformation_matrix()[..., :3, 2]
            # 目标方向：垂直向下 (0, 0, -1)
            target_down = torch.tensor([0.0, 0.0, -1.0], device=self.device).expand_as(tcp_z)
            # 计算点积 (1.0 代表完全垂直向下，-1.0 代表朝天)
            ori_align = torch.sum(tcp_z * target_down, dim=1)
            
            # 奖励逻辑：
            # 1. 只有没抓到的时候才需要强烈约束姿态 (抓到后怎么运都行)
            # 2. (ori_align - 1.0) 是一个负数惩罚项 (范围 -2 到 0)
            #    如果不垂直，就扣分。
            ori_reward = (ori_align - 1.0) * 0.5 * (1 - is_grasping)
            # [Grasp] 抓取奖励 (完全原版)
            grasp = is_grasping * (1 - torch.tanh((1.0 - norm_cos_sim).clamp(min=0.0)))

            # [Slow] 速度慢奖励 (完全原版)
            slow = is_grasping * (1 - torch.tanh(vel))
            
            # [Lift -> Transport] 搬运奖励 (完全原版逻辑，只是 goal_pos 变了)
            # 在 Lift 任务里，goal_pos 是上方；在 Sort 任务里，goal_pos 是目标框。
            # 只要这个 goal_dist 变小，它就会去搬。
            goal_dist = torch.linalg.norm(cube_pose.p - goal_pos, axis=1)
            lift = is_grasping * (1 - torch.tanh(5 * goal_dist))* 2.0

            # [Reaching] 接近奖励 (完全原版)
            reaching = (1 - torch.tanh(5 * dist)) * (1 - is_grasping)
            gripper_width = torch.linalg.norm(f1_pose.p - f2_pose.p, axis=1)
            
            # 2. 只有在还没抓到 (is_grasping=0) 且 离方块很近 (dist < 0.03) 时生效
            is_close = (dist < 0.03).float()
            
            # 3. 目标宽度是 0.03 (方块大小)，我们希望 gripper_width 接近 0.03
            # 如果张得太大(>0.03)，会有惩罚；接近 0.03，奖励越高。
            # tanh(50 * ...) 是为了让梯度更敏锐
            caging = is_close * (1.0 - torch.tanh(20.0 * (gripper_width - 0.03).clamp(min=0.0))) * 0.5 * (1 - is_grasping)            
            # 调试打印 (可选)
            # print(f"RGSL {reaching.mean():.3f}  {grasp.mean():.3f}  {slow.mean():.3f}  {lift.mean():.3f}")

            return reaching + grasp + lift

        # ---------------------------------------------------------------------
        # 3. 分别计算并求和
        # ---------------------------------------------------------------------
        
        # 左臂 (Agent 0) -> 红块
        reward_left = compute_single_arm_reward(self.agent.agents[0], self.red_cube, red_target_pos)
        
        # 右臂 (Agent 1) -> 绿块
        reward_right = compute_single_arm_reward(self.agent.agents[1], self.green_cube, green_target_pos)
        total_reward = reward_left
        
        # 成功大奖 (Success Bonus)
        if "red_success" in info:
            total_reward[info["success"]] = 0.5

        return total_reward
    def compute_normalized_dense_reward(self, obs, action, info):
        return self.compute_dense_reward(obs, action, info) / 14.0