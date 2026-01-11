import grasp_cube.agents.robots.so101.so_101_ee

import numpy as np
import sapien
import torch
import torch.nn.functional as F
import gymnasium as gym

from typing import Any, Optional, Sequence, Tuple, Union
from transforms3d.euler import euler2quat

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
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import (
    Camera,
    CameraConfig,
    parse_camera_configs,
    update_camera_configs_from_dict,
)

# from utils.distort import get_distorted_image_tensor
from utils.builder import (
    build_cube_w_friction,
    build_box_w_friction,
)

import constants as C

@register_env("PickCubeSO101-v0", max_episode_steps=100)
class PickCubeSO101Env(BaseEnv):
    
    SUPPORTED_ROBOTS = ["so101"]
    agent: MultiAgent

    CAMERA_INTRINSIC = np.array([
        [570.21740069, 0., 327.45975405],
        [0., 570.1797441, 260.83642155],
        [0., 0., 1.]
    ], dtype=np.float64)

    CAMERA_DISTORTION = np.array([
        -0.735413911,
        0.949258417,
        0.000189059234,
        -0.00200351391,
        -0.864150312
    ], dtype=np.float64)

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                uid="main_camera",
                pose=sapien_utils.look_at([0.316 - 0.050, 0.154 + 0.225, 0.300], [0.316 + 0.000, 0.154, 0.000]),
                width=128, height=128,
                near=0.01, far=100,
                fov=np.deg2rad(87.5),
            ),
        ]

    def _get_obs_sensor_data(self, apply_texture_transforms: bool = True) -> dict:
        """
        Get data from all registered sensors. Auto hides any objects that are designated to be hidden

        Args:
            apply_texture_transforms (bool): Whether to apply texture transforms to the simulated sensor data to map to standard texture formats. Default is True.

        Returns:
            dict: A dictionary containing the sensor data mapping sensor name to its respective dictionary of data. The dictionary maps texture names to the data. For example the return could look like

            .. code-block:: python

                {
                    "sensor_1": {
                        "rgb": torch.Tensor,
                        "depth": torch.Tensor
                    },
                    "sensor_2": {
                        "rgb": torch.Tensor,
                        "depth": torch.Tensor
                    }
                }
        """
        for obj in self._hidden_objects:
            obj.hide_visual()
        self.scene.update_render(update_sensors=True, update_human_render_cameras=False)
        self.capture_sensor_data()
        sensor_obs = dict()
        for name, sensor in self.scene.sensors.items():
            if isinstance(sensor, Camera):
                if self.obs_mode in ["state", "state_dict"]:
                    # normally in non visual observation modes we do not render sensor observations. But some users may want to render sensor data for debugging or various algorithms
                    sensor_obs[name] = sensor.get_obs(position=False, segmentation=False, apply_texture_transforms=apply_texture_transforms)
                else:
                    sensor_obs[name] = sensor.get_obs(
                        rgb=self.obs_mode_struct.visual.rgb,
                        depth=self.obs_mode_struct.visual.depth,
                        position=self.obs_mode_struct.visual.position,
                        segmentation=self.obs_mode_struct.visual.segmentation,
                        normal=self.obs_mode_struct.visual.normal,
                        albedo=self.obs_mode_struct.visual.albedo,
                        apply_texture_transforms=apply_texture_transforms
                    )

        if self.backend.render_device.is_cuda():
            torch.cuda.synchronize()
        return sensor_obs
        
    @property
    def _default_human_render_camera_configs(self):
        # 渲染视频时用的高画质相机
        # pose = sapien_utils.look_at([0.0, 0.9, 0.6], [0.2, 0.2, 0.15])
        # return CameraConfig(uid="render_camera", pose=pose, fov=np.pi/3, width=480, height=360)
        
        # pose = sapien_utils.look_at([0.3, 0.7, 0.7], [0.2, 0.2, 0.1])
        # return CameraConfig(uid="render_camera", pose=pose, fov=np.pi/3, width=480, height=360)
        
        pose = sapien_utils.look_at([0.0, 0.25, 0.20], [0.2, 0.2, 0.1])
        return CameraConfig(uid="render_camera", pose=pose, fov=np.pi/3,
        near=0.01, far=100, width=256, height=256)

    def _load_scene(self, options: dict):
        self.goal_pos = torch.tensor([0.459, 0.260, 0.150], dtype=torch.float32).to(self.device)

        self.scene.set_ambient_light([0.1, 0.1, 0.1])
        self.scene.add_directional_light([-0.25, -0.5, -1], [3.5, 3.5, 3.5], shadow=True)

        t = 0.018
        frame_z = 1e-5
        table_len_x = 1.182
        table_len_y = 0.60
        table_len_z = 0.1
        table_z = -table_len_z / 2.0

        table_half_sizes = [table_len_x / 2.0, table_len_y / 2.0, table_len_z / 2.0]
        table_pose = sapien.Pose([table_len_x / 2.0, table_len_y / 2.0, table_z])
        table_actor = build_box_w_friction(
            scene=self.scene,
            half_sizes=table_half_sizes,
            color=[202/256, 164/256, 114/256, 1.0],
            name="table_surface",
            body_type='static',
            add_collision=True,
            initial_pose=table_pose,
            static_friction=1.0,
            dynamic_friction=0.5,
            restitution=0.1,
        )
        
        #### ADD WALLS

        # 首先定义墙体的厚度和高度
        wall_thickness = 0.1      # 墙体厚度
        wall_height = 3.0         # 墙体高度（从地板到天花板）
        ceiling_thickness = 0.1   # 天花板厚度
        wall_distance = 2.0       # 墙离桌子的距离（原为1m，现改为2m）

        # 1. 添加地板 (z = -1m处)
        # 地板比桌子区域扩大2米（每边2米）
        floor_half_sizes = [
            (table_len_x + 2 * wall_distance) / 2.0,  # x方向半尺寸：桌子长度+两边各2米
            (table_len_y + 2 * wall_distance) / 2.0,  # y方向半尺寸：桌子宽度+两边各2米
            wall_thickness / 2.0                      # z方向半尺寸（厚度）
        ]
        floor_pose = sapien.Pose([
            table_len_x / 2.0,                       # 中心x与桌子对齐
            table_len_y / 2.0,                       # 中心y与桌子对齐
            -1.0 - wall_thickness / 2.0              # 地板下表面在z=-1m处
        ])
        floor_actor = actors.build_box(
            scene=self.scene,
            half_sizes=floor_half_sizes,
            color=[0.5, 0.5, 0.5, 1.0],  # 灰色地板
            name="floor",
            body_type='static',
            add_collision=True,
            initial_pose=floor_pose
        )

        # 2. 添加后墙 (x = -2m处) - 桌子左侧的墙
        back_wall_half_sizes = [
            wall_thickness / 2.0,                    # x方向半尺寸（厚度）
            (table_len_y + 2 * wall_distance) / 2.0, # y方向半尺寸：桌子宽度+两边各2米
            wall_height / 2.0                        # z方向半尺寸（高度）
        ]
        back_wall_pose = sapien.Pose([
            -wall_distance - wall_thickness / 2.0,   # 墙体内表面在x=-2m处
            table_len_y / 2.0,                       # 中心y与桌子对齐
            -1.0 + wall_height / 2.0                 # 中心z（从地板开始算起）
        ])
        back_wall_actor = actors.build_box(
            scene=self.scene,
            half_sizes=back_wall_half_sizes,
            color=[0.8, 0.8, 0.9, 1.0],  # 浅蓝色墙壁
            name="wall1",
            body_type='static',
            add_collision=False,
            initial_pose=back_wall_pose
        )

        # 3. 添加右墙 (x = table_len_x + 2m处) - 桌子右侧的墙
        right_wall_half_sizes = [
            wall_thickness / 2.0,                    # x方向半尺寸（厚度）
            (table_len_y + 2 * wall_distance) / 2.0, # y方向半尺寸：桌子宽度+两边各2米
            wall_height / 2.0                        # z方向半尺寸（高度）
        ]
        right_wall_pose = sapien.Pose([
            table_len_x + wall_distance + wall_thickness / 2.0,  # 墙体内表面在x=table_len_x+2m处
            table_len_y / 2.0,                                   # 中心y与桌子对齐
            -1.0 + wall_height / 2.0                             # 中心z
        ])
        right_wall_actor = actors.build_box(
            scene=self.scene,
            half_sizes=right_wall_half_sizes,
            color=[0.8, 0.8, 0.9, 1.0],
            name="wall2",
            body_type='static',
            add_collision=False,
            initial_pose=right_wall_pose
        )

        # 4. 添加左墙 (y = -2m处) - 桌子后侧的墙（如果以桌子正面为前）
        left_wall_half_sizes = [
            (table_len_x + 2 * wall_distance) / 2.0, # x方向半尺寸：桌子长度+两边各2米
            wall_thickness / 2.0,                    # y方向半尺寸（厚度）
            wall_height / 2.0                        # z方向半尺寸（高度）
        ]
        left_wall_pose = sapien.Pose([
            table_len_x / 2.0,                       # 中心x与桌子对齐
            -wall_distance - wall_thickness / 2.0,   # 墙体内表面在y=-2m处
            -1.0 + wall_height / 2.0                 # 中心z
        ])
        left_wall_actor = actors.build_box(
            scene=self.scene,
            half_sizes=left_wall_half_sizes,
            color=[0.8, 0.8, 0.9, 1.0],
            name="wall3",
            body_type='static',
            add_collision=False,
            initial_pose=left_wall_pose
        )

        # 5. 添加前墙 (y = table_len_y + 2m处) - 桌子前侧的墙
        front_wall_half_sizes = [
            (table_len_x + 2 * wall_distance) / 2.0, # x方向半尺寸：桌子长度+两边各2米
            wall_thickness / 2.0,                    # y方向半尺寸（厚度）
            wall_height / 2.0                        # z方向半尺寸（高度）
        ]
        front_wall_pose = sapien.Pose([
            table_len_x / 2.0,                       # 中心x与桌子对齐
            table_len_y + wall_distance + wall_thickness / 2.0,  # 墙体内表面在y=table_len_y+2m处
            -1.0 + wall_height / 2.0                 # 中心z
        ])
        front_wall_actor = actors.build_box(
            scene=self.scene,
            half_sizes=front_wall_half_sizes,
            color=[0.8, 0.8, 0.9, 1.0],
            name="wall4",
            body_type='static',
            add_collision=False,
            initial_pose=front_wall_pose
        )

        #### END ADD WALLS

        w1, w2, w3 = 0.166, 0.156, 0.166
        self.y1, self.y2 = 0.159, 0.341
        self.x1 = 0.029
        self.x2 = self.x1 + w1 + t
        self.x3 = self.x2 + w2 + t
        self.x4 = self.x3 + w3 + t
        x5 = 0.609
        x_centers = [self.x1, self.x2, self.x3, self.x4, x5]

        v_bounds = [(self.y1 + t/2, self.y2 - t/2), (0.0, self.y2 + t/2), (0.0, self.y2 + t/2), (self.y1 + t/2, self.y2 - t/2), (0.0, 0.60)]
        z_center = frame_z / 2.0

        for i, (xc, (yb, yt)) in enumerate(zip(x_centers, v_bounds)):
            length = yt - yb
            center_y = yb + length / 2.0
            half_sizes = [t/2.0, length/2.0, frame_z/2.0]
            pose = sapien.Pose([xc, center_y, z_center])
            v_edge_actor = actors.build_box(
                scene=self.scene,
                half_sizes=half_sizes,
                color=[0.0, 0.0, 0.0, 1.0],
                name=f"v_edge_{i+1}",
                body_type='static',
                add_collision=False,
                initial_pose=pose
            )

        h_len = (self.x4 - self.x1) / 2.0 + t / 2.0
        h_center_x = (self.x1 + self.x4) / 2.0
        for i, y_pos in enumerate([self.y1, self.y2]):
            half_sizes = [h_len, t/2.0, frame_z/2.0]
            pose = sapien.Pose([h_center_x, y_pos, z_center])
            h_edge_actor = actors.build_box(
                scene=self.scene,
                half_sizes=half_sizes,
                color=[0.0, 0.0, 0.0, 1.0],
                name=f"h_edge_{i+1}",
                body_type='static',
                add_collision=False,
                initial_pose=pose
            )

        h_top_cell = 0.164  

        # 创建 Cube
        self.red_cube_half_size = 0.015
        self.red_cube = build_cube_w_friction(
            self.scene, 
            half_size=self.red_cube_half_size, 
            color=[1, 0, 0, 1], 
            name="target_red_cube", 
            body_type="dynamic",
            static_friction=2.0,
            dynamic_friction=1.5,
            restitution=0.1
        )

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._initialize_agent(env_idx)
        
        with torch.device(self.device):
            b = len(env_idx) # batch size

            # 1. 初始化机器人姿态
            self.agent.reset()
            
            # 2. 随机化 Cube 位置
            xyz = torch.zeros((b, 3))
            xyz[:, 0] = torch.rand(b) * 0.12 - 0.06 + 0.472
            xyz[:, 1] = torch.rand(b) * 0.12 - 0.06 + 0.26
            xyz[:, 2] = self.red_cube_half_size
            
            yaw = (torch.rand(b) * 2 - 1) * torch.pi
            half_yaw = yaw * 0.5
            qs = torch.zeros((b, 4), device=self.device)
            qs[:, 0] = torch.cos(half_yaw)  # w
            qs[:, 3] = torch.sin(half_yaw)  # z

            # 设置所有并行环境的 Cube 位置
            self.red_cube.set_pose(Pose.create_from_pq(p=xyz, q=qs))
            
            if not hasattr(self, "red_cube_init_xy"):
                self.red_cube_init_xy = torch.zeros([len(env_idx), 2], dtype=torch.float32)

            self.red_cube_init_xy[env_idx] = xyz[:, :2]

            self.cur_action = torch.zeros(self.one_agent.action_space.shape)
            self.last_qvel = torch.zeros_like(self.one_agent.robot.get_qvel())[:, :-1]

    def _initialize_agent(self, env_idx: torch.Tensor):
        with torch.device(self.device):
            # b = len(env_idx)

            init_qpos = torch.from_numpy(self.one_agent.keyframes["rest"].qpos).unsqueeze(0)
            # init_qpos = init_qpos.repeat(b, 1)

            if isinstance(self.agent, MultiAgent):
                init_qpos_dict = {
                    robot_name: init_qpos 
                    for robot_name, _ in self.agent.agents_dict.items()
                }
                self.agent.reset(init_qpos_dict)
                # self.agent.reset({
                #     "so101-0": init_qpos,
                #     "so101-1": init_qpos,
                # })
                # print(f"Initialize multiagent of {env_idx}")
            else:
                self.agent.reset(init_qpos)

    def __init__(self, *args, robot_uids="so101", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
    
    def _load_agent(self, options: dict, agent_poses: Optional[Union[sapien.Pose, Pose]] = None):
        if agent_poses == None:
            agent_poses = Pose.create_from_pq(p=[0.481, 0.025, 0], q=[np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2])
        super()._load_agent(options, agent_poses)

        self.one_agent = self.agent.agents[0] if isinstance(self.agent, MultiAgent) else self.agent
    
    def step(self, action: Union[None, np.ndarray, torch.Tensor, dict]):
        """
        Take a step through the environment with an action. Actions are automatically clipped to the action space.

        If ``action`` is None, the environment will proceed forward in time without sending any actions/control signals to the agent
        """
        action = self._step_action(action)
        with torch.no_grad():
            self.cur_action = action.detach().clone()
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
        return (
            obs,
            reward,
            terminated,
            torch.zeros(self.num_envs, dtype=bool, device=self.device),
            info,
        )
    
    def evaluate(self):
        """
        判断任务是否成功 / 失败
        """
        red_cube_x = self.red_cube.pose.p[:, 0]
        red_cube_y = self.red_cube.pose.p[:, 1]
        red_cube_z = self.red_cube.pose.p[:, 2]

        qvel = self.agent.robot.get_qvel()
        qvel = qvel[..., :-1]
        vel = torch.linalg.norm(qvel, axis=1)

        # 是否已经跑满
        # is_late = self.elapsed_steps >= max_episode_steps

        # (x,y)是否在方格内
        # is_inside = (torch.abs(red_cube_x - 0.472) <= 0.2) & (torch.abs(red_cube_y - 0.26) <= 0.2)

        # 是否举起
        is_lifted = red_cube_z > C.REQUIRED_HEIGHT

        is_inside = torch.linalg.norm(self.red_cube.pose.p - self.goal_pos, axis=1) < 0.050

        # 爪子是否关闭
        is_grasping = self.agent.is_grasping(self.red_cube)

        # 是否慢
        is_slow = vel < C.SPEED_LIMIT       

        return {
            "is_inside": is_inside,
            "is_grasping": is_grasping,
            "is_slow": is_slow,
            "success": is_lifted & is_grasping & is_slow,
        }

    def _get_obs_extra(self, info: dict):
        """
        额外观测函数
        """
        obs = dict(
            last_action=self.cur_action,
            tcp_pose=self.agent.tcp_pose.raw_pose,
        )

        if self.obs_mode_struct.use_state:
            obs.update(
                red_cube_pose=self.red_cube.pose.raw_pose,
                is_grasping=self.agent.is_grasping(self.red_cube)
            )

        return obs

    # def compute_dense_reward(self, obs: any, action: Array, info: dict):
    #   f1_pose = self.agent.finger1_tip.pose
    #   f2_pose = self.agent.finger2_tip.pose
    #   tcp_pose = self.agent.tcp_pose
    #   red_cube_pose = self.red_cube.pose

    #   is_grasping = self.agent.is_grasping(self.red_cube).float()

    #   vec_1 = f1_pose.p - red_cube_pose.p
    #   vec_2 = red_cube_pose.p - f2_pose.p
    #   cos_sim = F.cosine_similarity(vec_1, vec_2, dim=-1)
    #   norm_cos_sim = ((cos_sim + 1) / 2).clamp(max=0.9)

    #   qvel = self.agent.robot.get_qvel()[:, :-1]

    #   # 速度连续性
    #   # qvel_dist = torch.linalg.norm(self.last_qvel - qvel, axis=1)
    #   # consistency = 1 - torch.tanh(5 * qvel_dist)

    #   # 接近奖励
    #   dist = torch.linalg.norm(tcp_pose.p - red_cube_pose.p, axis=1)
    #   reaching = 1 - torch.tanh(5 * dist)

    #   # 抓取奖励
    #   grasp = is_grasping * (1 - torch.tanh(0.9 - norm_cos_sim))

    #   # [After Grasp] 速度慢奖励
    #   vel = torch.linalg.norm(qvel, axis=1)
    #   slow = is_grasping * (1 - torch.tanh(5 * vel))
        
    #   # [After Grasp] 和目标位置距离奖励
    #   goal_pos = torch.cat([
    #       self.red_cube_init_xy,
    #       torch.ones_like(self.red_cube_init_xy[:, :1]) * C.GOAL_HEIGHT,
    #   ], dim=-1)
    #   goal_dist = torch.linalg.norm(red_cube_pose.p - goal_pos, axis=1)
    #   lift = is_grasping * (1 - torch.tanh(5 * goal_dist))

    #   self.last_qvel = qvel
        
    #   total_reward = consistency + reaching + grasp + slow + lift
    #   total_reward[info["success"]] = 5.0

    #   return total_reward

    def compute_dense_reward(self, obs: any, action: Array, info: dict):
        f1_pose = self.agent.finger1_tip.pose
        f2_pose = self.agent.finger2_tip.pose
        tcp_pose = self.agent.tcp_pose
        red_cube_pose = self.red_cube.pose

        is_grasping = self.agent.is_grasping(self.red_cube).float()

        vec_1 = f1_pose.p - red_cube_pose.p
        vec_2 = red_cube_pose.p - f2_pose.p
        cos_sim = F.cosine_similarity(vec_1, vec_2, dim=-1)
        norm_cos_sim = (cos_sim + 1) / 2

        qvel = self.agent.robot.get_qvel()[:, :-1]

        dist = torch.linalg.norm(tcp_pose.p - red_cube_pose.p, axis=1)

        # 抓取奖励
        grasp = is_grasping * (1 - torch.tanh((1.0 - norm_cos_sim).clamp(min=0.0)))

        # [After Grasp] 速度慢奖励
        vel = torch.linalg.norm(qvel, axis=1)
        slow = is_grasping * (1 - torch.tanh(vel))
        
        # [After Grasp] 和目标位置距离奖励
        goal_dist = torch.linalg.norm(red_cube_pose.p - self.goal_pos, axis=1)
        lift = is_grasping * (1 - torch.tanh(5 * goal_dist))

        # [Before Grasp]
        reaching = (1 - torch.tanh(5 * dist)) * (1 - is_grasping)
        
        print(f"RGSL {reaching.mean()}  {grasp.mean()}  {slow.mean()}  {lift.mean()}")

        total_reward = reaching + grasp + slow + lift
        total_reward[info["success"]] = 3.0

        return total_reward
        
    def compute_normalized_dense_reward(self, obs, action, info):
        rew = self.compute_dense_reward(obs, action, info) / 3.0
        return rew
