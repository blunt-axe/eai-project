import numpy as np
import torch
import torch.nn.functional as F
from typing import Union

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from so101_lift_cube_v2 import PickCubeSO101Env
from utils.builder import build_cube_w_friction, build_visual_sphere, build_matte_box_area
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils


@register_env("ArrangeCubeSO101-v0", max_episode_steps=1500)
class ArrangeCubeSO101Env(PickCubeSO101Env):
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
			CameraConfig(
				uid="main_camera_2",
				pose=sapien_utils.look_at([0.316 + 0.050, 0.154 + 0.050, 0.450], [0.316, 0.154, 0.000]),
				width=128, height=128,
				near=0.01, far=100,
				fov=np.deg2rad(87.5),
			),
		]

	def __init__(self, *args, robot_uids="so101", **kwargs):
		self.CUBE_SIZE = 0.03
		self.CUBE_HALF_SIZE = self.CUBE_SIZE / 2  # 固定值 0.015

		self.IN_AIR_HEIGHT = 0.08
		self.IN_LOW_AIR_HEIGHT = 0.02
		self.DIST_MEAN = 0.02

		self.CENTER_POSITION_LEFT = np.array([0.121, 0.25, self.CUBE_HALF_SIZE])
		self.CENTER_POSITION = np.array([0.3, 0.25, self.CUBE_HALF_SIZE])
		self.CENTER_POSITION_RIGHT = np.array([0.3 * 2 - 0.121, 0.25, self.CUBE_HALF_SIZE])
		self.AREA_CENTERS = torch.tensor([
			self.CENTER_POSITION_RIGHT,  # 区域0 右
			self.CENTER_POSITION,        # 区域1 中
			self.CENTER_POSITION_LEFT    # 区域2 左
		], dtype=torch.float32)

		self.RAND_RANGE = torch.tensor([0.02, 0.02, 0.0], dtype=torch.float32)
		self.THRESHOLD = 0.015
		self.CENTER_BR_OFFSET = np.array([0.07, -0.07, 0.0])
		self.AREA1_BR_POS = torch.tensor(self.CENTER_POSITION + self.CENTER_BR_OFFSET, dtype=torch.float32)

		self.DEFAULT_TCP_CENTER_POS_0 = np.array([0.4828, 0.2630, 0.0676])
		self.DEFAULT_TCP_CENTER_POS_1 = np.array([0.1208, 0.2630, 0.0676])

		self.PERM_IDX_TO_ARR = torch.tensor([
			[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0],
		], dtype=torch.long)
		self.PERM_STR_LIST = ["012", "021", "102", "120", "201", "210"]

		# ========== 动作耗时参数 (保持你的原值不变) ==========
		self.MOVE_TO_CUBE_TIME = 1.0
		self.MOVE_WITH_CUBE_TIME = 1.0 
		self.CLAMP_TIME = 3.0          # 夹紧耗时
		self.RELEASE_TIME = 3.0        # 松开抬升耗时
		self.RESET_TIME = 4.0          # 复位耗时

		self.ADD_TIME_TCP2AREA0 = 1.0
		self.ADD_TIME_AREA02BR = 4.0
		self.ADD_TIME_BR2AREA1 = 2.0
		self.ADD_TIME_AREA12AREA0 = 6.0

		self.arm_step_actions = [
			"move2area0", "clamp", "move2br", "release_up", "move2area1", "clamp",
			"move2area0", "release_up", "move2br", "clamp", "move2area1",
			"release_up", "reset"
		]
		
		# 核心逻辑：无物移动=基础without cube耗时+附加；带物移动=基础with cube耗时+附加；同动作附加耗时完全一致，保证时间相同
		self.arm_step_durations = torch.tensor([
			self.MOVE_TO_CUBE_TIME + self.ADD_TIME_TCP2AREA0,
			self.CLAMP_TIME,
			self.MOVE_WITH_CUBE_TIME + self.ADD_TIME_AREA02BR,
			self.RELEASE_TIME,
			self.MOVE_TO_CUBE_TIME + self.ADD_TIME_BR2AREA1,
			self.CLAMP_TIME,
			self.MOVE_WITH_CUBE_TIME + self.ADD_TIME_AREA12AREA0,
			self.RELEASE_TIME,
			self.MOVE_TO_CUBE_TIME + self.ADD_TIME_AREA02BR,
			self.CLAMP_TIME,
			self.MOVE_WITH_CUBE_TIME + self.ADD_TIME_BR2AREA1,
			self.RELEASE_TIME,
			self.RESET_TIME
		], dtype=torch.float32)
		
		self.STEP_FREQ = 30  # 固定30Hz 操作频率
		self.arm_step_steps = torch.round(self.arm_step_durations * self.STEP_FREQ).long()
		self.arm_step_steps_cumsum = torch.cumsum(self.arm_step_steps, dim=0)
		self.total_steps = self.arm_step_steps_cumsum[-1].item()

		robot_uids = (robot_uids, robot_uids)
		super().__init__(*args, robot_uids=robot_uids, **kwargs)
		
	def _load_agent(self, options: dict):
		super()._load_agent(options, [
			Pose.create_from_pq(p=[0.481, 0.025, 0], q=[np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2]),
			Pose.create_from_pq(p=[0.119, 0.025, 0], q=[np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2]),
		])
		self.one_agent = self.agent.agents[0]

	def _load_scene(self, options):
		super()._load_scene(options)
		self.cube_A = self.red_cube
		self.cube_B = build_cube_w_friction(
			self.scene, self.CUBE_HALF_SIZE, [0, 1, 0, 1], "cube_B",
			initial_pose=Pose.create_from_pq(p=[0.0, 0.0, 2.0], q=[1.0, 0.0, 0.0, 0.0])
		)
		self.cube_C = build_cube_w_friction(
			self.scene, self.CUBE_HALF_SIZE, [0, 0, 1, 1], "cube_C",
			initial_pose=Pose.create_from_pq(p=[0.0, 0.0, 3.0], q=[1.0, 0.0, 0.0, 0.0])
		)
		self.cubes = [self.cube_A, self.cube_B, self.cube_C]
		self.cube_num = len(self.cubes)

		self.oracle_indicator = build_visual_sphere(
			self.scene, radius=0.01, color=[1, 1, 0, 1], name="oracle_indicator", body_type="kinematic",
			initial_pose=Pose.create_from_pq(p=[0.0, 0.0, 4.0], q=[1.0, 0.0, 0.0, 0.0])
		)

	def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
		self._initialize_agent(env_idx)

		b = len(env_idx)
		device = self.device
		
		self.AREA_CENTERS = self.AREA_CENTERS.to(device)
		self.RAND_RANGE = self.RAND_RANGE.to(device)
		self.PERM_IDX_TO_ARR = self.PERM_IDX_TO_ARR.to(device)
		self.arm_step_durations = self.arm_step_durations.to(device)
		self.arm_step_steps = self.arm_step_steps.to(device)
		self.arm_step_steps_cumsum = self.arm_step_steps_cumsum.to(device)
		self.AREA1_BR_POS = self.AREA1_BR_POS.to(device)
		
		with torch.device(self.device):
			# Randomize other arm qpos
			self.agent.agents[1].robot.set_qpos((torch.rand((b, 6)) * 2 - 1) * 0.20 + torch.tensor(self.agent.agents[0].keyframes["rest"].qpos))
			# self.agent.agents[0].robot.set_qpos(torch.tensor(self.agent.agents[0].keyframes["rest"].qpos))
			# self.agent.agents[1].robot.set_qpos(torch.tensor(self.agent.agents[0].keyframes["rest"].qpos))
			# print(f"{self.agent.agents[0].tcp_pose.p}")
			# print(f"{self.agent.agents[1].tcp_pose.p}")

		if not hasattr(self, "active_cube_idx"):
			self.active_cube_idx = torch.zeros(self.num_envs, dtype=torch.long, device=device)
			self._elapsed_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
			self.cube_success = torch.zeros((self.num_envs, self.cube_num), dtype=torch.bool, device=device)
			# self.cube_start_pos = torch.zeros((self.num_envs, self.cube_num, 3), dtype=torch.float32, device=device)
			# self.target_pos_global = torch.zeros((self.num_envs, self.cube_num, 3), dtype=torch.float32, device=device)
			self.cube_perm_idx = torch.zeros(self.num_envs, dtype=torch.long, device=device)
			self.cube_perm_str = [""] * self.num_envs
			self.gripper_state = torch.zeros(self.num_envs, dtype=torch.long, device=device) # 0松开 1夹紧
			self.arm_step = torch.zeros(self.num_envs, dtype=torch.long, device=device)     # 0-12 对应13步
			self.step_in_phase = torch.zeros(self.num_envs, dtype=torch.long, device=device) # 记录当前阶段内已执行的步数

		self.active_cube_idx[env_idx] = 0
		self._elapsed_steps[env_idx] = 0
		self.cube_success[env_idx] = False
		self.gripper_state[env_idx] = 0
		self.arm_step[env_idx] = 0
		self.step_in_phase[env_idx] = 0

		# self.target_pos_global[env_idx] = self.AREA_CENTERS[1].repeat(b, 1)
		# self.target_pos_global[env_idx, 2] = self.CUBE_HALF_SIZE

		self.cube_perm_idx[env_idx] = torch.randint(low=0, high=6, size=(b,), device=device)
		env_cube_perm = self.PERM_IDX_TO_ARR[self.cube_perm_idx[env_idx]]
		for i in range(b):
			self.cube_perm_str[env_idx[i]] = self.PERM_STR_LIST[self.cube_perm_idx[env_idx[i]]]

		cube_xyzs = torch.zeros((b, self.cube_num, 3), dtype=torch.float32, device=device)
		for area_idx in range(3):
			area_base_pos = self.AREA_CENTERS[area_idx]
			rand_offset = (torch.rand((b, 2), device=device) * 2 - 1) * self.RAND_RANGE[:2]
			area_rand_pos = torch.cat([area_base_pos[:2].repeat(b,1) + rand_offset, torch.full((b,1), self.CUBE_HALF_SIZE, device=device)], dim=-1)
			cube_ids_in_area = env_cube_perm[:, area_idx]
			cube_xyzs[torch.arange(b), cube_ids_in_area] = area_rand_pos

		def random_rotate_quat():
			yaw = (torch.rand(b) * 2 - 1) * torch.pi
			half_yaw = yaw * 0.5
			qs = torch.zeros((b, 4), device=self.device)
			qs[:, 0] = torch.cos(half_yaw)  # w
			qs[:, 3] = torch.sin(half_yaw)  # z
			return qs

		# self.cube_start_pos[env_idx] = cube_xyzs
		for cube_id in range(self.cube_num):
			self.cubes[cube_id].set_pose(Pose.create_from_pq(p=cube_xyzs[:, cube_id], q=random_rotate_quat()))

		self.oracle_indicator.set_pose(Pose.create_from_pq(p=self.DEFAULT_TCP_CENTER_POS_0))

	def step(self, action: Union[np.ndarray, torch.Tensor, dict]):
		# 强制添加 secondary arm 抖动
		if isinstance(action, dict):
			raw_uid = self.agent.agents[0].uid
			arm_key = f"{raw_uid}-0"
			
			for key in action.keys():
				if key != arm_key:
					val = action[key]
					if isinstance(val, torch.Tensor):
						action[key] = (torch.rand_like(val) * 2 - 1) * 0.20
					elif isinstance(val, np.ndarray):
						action[key] = (np.rand_like(val) * 2 - 1) * 0.20

		action = self._step_action(action)
		self._elapsed_steps += 1

		device = self.device
		idx = self.active_cube_idx
		arange = torch.arange(self.num_envs, device=device)

		# start_pos = self.cube_start_pos[arange, idx]
		# end_pos = self.target_pos_global
		cube_pos = torch.stack([c.pose.p for c in self.cubes])[idx, arange]
		new_pos = cube_pos.clone().float()
		
		def move_with_cube(start_p: torch.Tensor, end_p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
			# 核心加减速因子：S型曲线 0→1 平滑过渡，速度首尾为0，先加速后减速
			s = 0.5 - 0.5 * torch.cos(t * torch.pi)
			# XY轴：加减速平滑插值移动
			xy = (1 - s)[:, None] * start_p[:, :2] + s[:, None] * end_p[:, :2]
			# Z轴：保持峰值0.03，与加减速节奏匹配
			z = torch.full_like(t, self.IN_LOW_AIR_HEIGHT)
			return torch.cat([xy, z[:, None]], dim=-1).float()

		def release_and_lift(start_p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
			# 核心加减速因子：S型曲线 0→1 平滑过渡
			s = 0.5 - 0.5 * torch.cos(t * torch.pi)
			xy = start_p[:, :2]
			# Z轴：加减速平滑抬升，从0.015到0.08，无顿挫、首尾速度为0
			base_z = self.CUBE_HALF_SIZE
			lift_z = base_z + (self.IN_AIR_HEIGHT - base_z) * s
			return torch.cat([xy, lift_z[:, None]], dim=-1).float()

		def move_without_cube(start_p: torch.Tensor, end_p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
			s = 0.5 - 0.5 * torch.cos(t * torch.pi)
			xy = (1 - s)[:, None] * start_p[:, :2] + s[:, None] * end_p[:, :2]
			start_z = torch.maximum(start_p[:, 2], torch.full_like(start_p[:, 2], self.IN_AIR_HEIGHT))
			end_z = torch.maximum(end_p[:, 2], torch.full_like(end_p[:, 2], self.IN_AIR_HEIGHT))
			z = (1 - s) * start_z + s * end_z
			return torch.cat([xy, z[:, None]], dim=-1).float()

		def grasp_cube_drop(start_p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
			# 核心加减速因子：S型曲线 0→1 平滑过渡
			s = 0.5 - 0.5 * torch.cos(t * torch.pi)
			xy = start_p[:, :2]
			# Z轴：加减速平滑下降，从0.08到0.015，无冲击、首尾速度为0
			start_z = self.IN_AIR_HEIGHT
			end_z = self.IN_LOW_AIR_HEIGHT
			z = start_z - (start_z - end_z) * s
			return torch.cat([xy, z[:, None]], dim=-1).float()

		# ===================== 纯中心点坐标 (无任何cube pos依赖，全程用区域中心) =====================
		area0_pos = self.AREA_CENTERS[0].repeat(self.num_envs,1).float().to(device)
		area1_pos = self.AREA_CENTERS[1].repeat(self.num_envs,1).float().to(device)
		br_pos = self.AREA1_BR_POS.repeat(self.num_envs,1).float().to(device)
		reset_pos = torch.from_numpy(self.DEFAULT_TCP_CENTER_POS_0).float().to(device).repeat(self.num_envs,1)

		step_action_map = [
			("no_cube", reset_pos, area0_pos, self.MOVE_TO_CUBE_TIME, 0),  # 0 移到 area0 center
			("clamp", area0_pos, area0_pos, self.CLAMP_TIME, 1),          # 1 在 area0 夹紧
			("with_cube", area0_pos, br_pos, self.MOVE_WITH_CUBE_TIME, 1),# 2 area0 → BR
			("release_up", br_pos, br_pos, self.RELEASE_TIME, 0),         # 3 松开抬升

			("no_cube", br_pos, area1_pos, self.MOVE_TO_CUBE_TIME, 0),    # 4 移到 area1 center
			("clamp", area1_pos, area1_pos, self.CLAMP_TIME, 1),          # 5 在 area1 夹紧
			("with_cube", area1_pos, area0_pos, self.MOVE_WITH_CUBE_TIME, 1), # 6 area1 → area0
			("release_up", area0_pos, area0_pos, self.RELEASE_TIME, 0),   # 7 松开抬升

			("no_cube", area0_pos, br_pos, self.MOVE_TO_CUBE_TIME, 0),    # 8 移到 BR
			("clamp", br_pos, br_pos, self.CLAMP_TIME, 1),                # 9 在 BR 夹紧
			("with_cube", br_pos, area1_pos, self.MOVE_WITH_CUBE_TIME, 1), #10 BR → area1
			("release_up", area1_pos, area1_pos, self.RELEASE_TIME, 0),   #11 松开抬升

			("no_cube", area1_pos, reset_pos, self.RESET_TIME, 0)         #12 复位
		]

		current_step = self.arm_step[arange]
		# 当前阶段需要执行的总步数
		current_phase_total_steps = self.arm_step_steps[current_step]
		# 当前阶段内已执行的步数+1
		self.step_in_phase += 1

		# 计算当前阶段的归一化进度 t ∈ [0,1]  精准无误差
		t = (self.step_in_phase.float() / current_phase_total_steps.float()).clamp(0.0, 1.0)

		# 阶段切换判断：当 阶段内执行步数 >= 该阶段总步数 时，切换到下一阶段
		step_switch_mask = (self.step_in_phase >= current_phase_total_steps) & (current_step < 12)
		self.arm_step[step_switch_mask] = torch.clamp(self.arm_step[step_switch_mask] + 1, 0, 12)
		# 切换阶段后，重置该环境的「阶段内步数」为0
		self.step_in_phase[step_switch_mask] = 0
		# 更新夹爪状态
		self.gripper_state[arange] = torch.tensor([step_action_map[s][4] for s in current_step], device=device, dtype=torch.long)

		# ===================== oracle坐标计算+赋值 =====================
		oracle_p = torch.zeros_like(new_pos, dtype=torch.float32, device=device)
		for s in range(13):
			mask = current_step == s
			if not mask.any(): continue
			act_type, s_p, e_p, _, _ = step_action_map[s]
			t_mask = t[mask]
			if act_type == "with_cube":
				oracle_p[mask] = move_with_cube(s_p[mask], e_p[mask], t_mask)
			elif act_type == "release_up":
				oracle_p[mask] = release_and_lift(s_p[mask], t_mask)
			elif act_type == "no_cube":
				oracle_p[mask] = move_without_cube(s_p[mask], e_p[mask], t_mask)
			elif act_type == "clamp":
				oracle_p[mask] = grasp_cube_drop(s_p[mask], t_mask)
			else:
				oracle_p[mask] = s_p[mask]

		self.oracle_indicator.set_pose(Pose.create_from_pq(p=oracle_p.cpu().numpy()))

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
				terminated = torch.zeros(self.num_envs, dtype=bool, device=device)
		self._last_obs = obs
		if not hasattr(self, "right_arm_rest_qpos"):
			rest_np = self.agent.agents[1].keyframes["rest"].qpos
			self.right_arm_rest_qpos = torch.from_numpy(rest_np).float().to(device)

		return (
			obs,
			reward,
			terminated,
			torch.zeros(self.num_envs, dtype=bool, device=device),
			info,
		)

	def _get_obs_extra(self, info: dict):
		# 基础信息准备
		oracle_pos = self.oracle_indicator.pose.p
		device = self.device
		num_envs = self.num_envs
		
		obs = dict()

		# 1. 抓取状态检查 (保持你原有的逻辑)
		agent_left = self.agent.agents[0]
		agent_right = self.agent.agents[1]
		is_grasp_cubeA = agent_left.is_grasping(self.cube_A)
		is_grasp_cubeB = agent_left.is_grasping(self.cube_B)
		is_grasp_cubeC = agent_left.is_grasping(self.cube_C)
		
		# 2. TCP 位姿拼接
		tcp_left = agent_left.tcp_pose.raw_pose
		tcp_right = agent_right.tcp_pose.raw_pose
		tcp_pose = torch.cat([tcp_left, tcp_right], dim=-1)

		# 3. 添加 1-hot Phase 变量 (保持原有逻辑)
		phase_idx = self.arm_step
		phase_1hot = F.one_hot(phase_idx.to(torch.long), num_classes=13).float()
		phase_4_1hot = F.one_hot(phase_idx.to(torch.long) % 4, num_classes=4).float()

		tcp2target_cube_diff = torch.zeros((num_envs, 3), dtype=torch.float32, device=device)
		phase_mask = (phase_idx < 12)
		if phase_mask.any():
			tcp_pos = agent_left.tcp_pose.p[phase_mask]
			perm = self.PERM_IDX_TO_ARR[self.cube_perm_idx[phase_mask]]
			phase_sub = phase_idx[phase_mask] // 3  # phase//3 分类，0/1/2/3
			cube_ids = torch.where(phase_sub==1, perm[:,1], perm[:,0]) # =1用perm[1]，0/2/3都用perm[0]
			
			cubeA_pos = self.cube_A.pose.p[phase_mask]
			cubeB_pos = self.cube_B.pose.p[phase_mask]
			cubeC_pos = self.cube_C.pose.p[phase_mask]
			target_cube_pos = torch.zeros_like(tcp_pos)
			target_cube_pos[cube_ids==0] = cubeA_pos[cube_ids==0]
			target_cube_pos[cube_ids==1] = cubeB_pos[cube_ids==1]
			target_cube_pos[cube_ids==2] = cubeC_pos[cube_ids==2]
			
			tcp2target_cube_diff[phase_mask] = tcp_pos - target_cube_pos

		# 4. 填充 Obs 字典
		obs["is_grasping"] = torch.logical_or(torch.logical_or(is_grasp_cubeA, is_grasp_cubeB), is_grasp_cubeC)
		obs["tcp_pose"] = tcp_pose
		obs["diff_oracle_left"] = oracle_pos - tcp_left[..., :3]
		obs["phase_1hot"] = phase_1hot
		obs["phase_4_1hot"] = phase_4_1hot
		obs["tcp2target_cube_diff"] = tcp2target_cube_diff

		return obs

	def evaluate(self):
		device = self.device
		success_all = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
		results = {}
		results["success"] = success_all

		fail_all = torch.zeros(self.num_envs, dtype=torch.bool, device=device)
		agent_left = self.agent.agents[0]

		tcp_pos = agent_left.tcp_pose.p
		oracle_pos = self.oracle_indicator.pose.p
		dist_to_oracle = torch.linalg.norm(tcp_pos - oracle_pos, dim=1)
		
		tcp_quat = agent_left.tcp_pose.q
		target_vec = torch.tensor([0.0, 0.0, -1.0], device=device).repeat(self.num_envs, 1)
		qw, qx, qy, qz = tcp_quat[:,0], tcp_quat[:,1], tcp_quat[:,2], tcp_quat[:,3]
		x = 2*(qx*qz + qw*qy)
		y = 2*(qy*qz - qw*qx)
		z = 1 - 2*(qx**2 + qy**2)
		tcp_z_dir = torch.stack([x, y, z], dim=1)

		cos_sim = torch.nn.functional.cosine_similarity(tcp_z_dir, target_vec, dim=1)

		fail_all = (cos_sim < 0.9) | (dist_to_oracle > 0.1)
		results["fail"] = fail_all

		results["cos_sim_gravity"] = cos_sim

		return results

	def compute_dense_reward(self, obs, action, info: dict):
		device = self.device
		agent = self.agent.agents[0]

		def dist2smooth_penalty(dist, dist_mean=self.DIST_MEAN):
			norm_dist = dist / dist_mean
			cond = norm_dist >= 1.0
			return dist_mean / 2.0 * torch.where(cond, 2 * norm_dist - 1, norm_dist**2)

		diff_tcp_oracle = obs["extra"]["diff_oracle_left"]
		diff_tcp_oracle[..., 2] = 2.0 * diff_tcp_oracle[..., 2]
		dist_tcp2oracle = torch.linalg.norm(diff_tcp_oracle, dim=1)
		oracle_near_reward = 1.0 - torch.tanh(5.0 * dist2smooth_penalty(dist_tcp2oracle))
		
		grasp_correct = obs["extra"]["is_grasping"].long() == self.gripper_state

		total_reward = 3.0 * oracle_near_reward * (4 * grasp_correct + 1) / 5

		phase_mask = (self.arm_step % 4 == 1)
		if phase_mask.any():
			grasp_correct_mask = grasp_correct[phase_mask]
			
			tcp2cube_diff = obs["extra"]["tcp2target_cube_diff"][phase_mask]
			dist_tcp2cube = torch.linalg.norm(tcp2cube_diff, dim=1)
			reach = 1.0 - torch.tanh(5.0 * dist_tcp2cube)
			
			f1_pose = agent.finger1_tip.pose[phase_mask]
			f2_pose = agent.finger2_tip.pose[phase_mask]
			tcp_pose = agent.tcp_pose[phase_mask]

			cube_pos = tcp_pose.p - tcp2cube_diff
			vec_1 = f1_pose.p - cube_pos
			vec_2 = cube_pos - f2_pose.p
			cos_sim = F.cosine_similarity(vec_1, vec_2, dim=-1)
			norm_cos_sim = (cos_sim + 1) / 2

			gripper_width = torch.linalg.norm(f1_pose.p - f2_pose.p, dim=1)
			is_close = dist_tcp2cube < 0.010
			caging = is_close * (1.0 - torch.tanh(10.0 * (gripper_width - 0.045).clamp(min=0.0)))

			reach_and_grasp = (~grasp_correct_mask) * (reach + norm_cos_sim + caging) / 3 + grasp_correct_mask

			total_reward[phase_mask] = 2.9 * reach_and_grasp + 0.1 * oracle_near_reward[phase_mask]
			print(f"reach={reach.mean()}, norm_cos_sim={norm_cos_sim.mean()}, caging={caging.mean()}")
			print(f"reach_and_grasp={reach_and_grasp.mean()}  oracle_near_reward={oracle_near_reward.mean()}")
		
		else:
			print(f"mult={total_reward.mean()}")

		total_reward += 0.1 * info["cos_sim_gravity"]

		total_reward[info["fail"]] = -1.0
		
		return total_reward
	
	def compute_normalized_dense_reward(self, obs, action, info):
		return self.compute_dense_reward(obs, action, info) / 3.0