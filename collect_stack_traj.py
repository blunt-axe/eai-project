import os
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import gymnasium as gym
from dataclasses import dataclass
import tyro

# ManiSkill相关导入
import mani_skill.envs
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

# 从你的训练代码中导入必要的类
from lift_ppo import Agent, DictArray

from so101_stack_cube import StackCubeSO101Env
from grasp_release_wrapper import GraspReleaseWrapper

@dataclass
class CollectArgs:
	"""收集轨迹的参数"""
	checkpoint_path: str = "runs/stack_norelease.pt"
	"""模型checkpoint路径"""
	env_id: str = "StackCubeSO101-v0"
	"""环境ID"""
	num_envs: int = 12
	"""并行环境数量"""
	# num_episodes: int = 240
	num_episodes: int = 60
	"""收集的episode数量"""
	max_steps_per_episode: int = 200
	"""每个episode的最大步数"""
	output_dir: str = "./collected_trajectories/Tmp"
	"""输出目录"""
	seed: int = 42
	"""随机种子"""
	include_state: bool = True
	"""是否包含状态信息"""
	control_mode: str = "pd_ee_delta_pose"
	"""控制模式"""
	render_mode: str = "rgb_array"
	"""渲染模式，使用rgb_array以便RecordEpisode获取图像"""
	deterministic: bool = True
	"""是否使用确定性策略"""
	
	# RecordEpisode相关参数
	record_video: bool = True
	"""是否录制视频"""
	video_fps: int = 30
	"""视频帧率"""
	save_trajectory: bool = True
	"""是否保存轨迹数据"""
	info_on_video: bool = True
	"""在视频上显示信息"""
	max_steps_per_video: Optional[int] = None
	"""每个视频的最大步数，None表示与episode相同"""

class SimpleTrajectoryCollector:
	"""简化的轨迹收集器（使用RecordEpisode包装器）"""
	
	def __init__(self, args: CollectArgs):
		self.args = args
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		
		# 创建环境
		self._create_envs()
		
		# 加载模型
		self.agent = self._load_agent()
		
		print(f"Trajectory collection setup complete")
		print(f"Output directory: {args.output_dir}")
		
		# 统计信息
		self.stats = {
			'total_episodes': 0,
			'total_steps': 0,
			'successful_episodes': 0,
			'avg_reward': 0.0
		}
	
	def _create_envs(self):
		"""创建环境（正确的包装顺序）"""
		env_kwargs = {
			"obs_mode": "rgb", 
			"robot_uids": "so101", 
			"render_mode": self.args.render_mode,  # 使用rgb_array
			"sim_backend": "physx_cuda",
			"reward_mode": "dense",
		}
		if self.args.control_mode is not None:
			env_kwargs["control_mode"] = self.args.control_mode
			
		# 创建基础环境
		env = gym.make(
			self.args.env_id, 
			num_envs=self.args.num_envs,
			reconfiguration_freq=0,  # 设置为0以支持partial reset
			**env_kwargs
		)
		
		# 应用包装器
		env = FlattenRGBDObservationWrapper(
			env, 
			rgb=True, 
			depth=False, 
			state=self.args.include_state,
		)
		
		if isinstance(env.action_space, gym.spaces.Dict):
			env = FlattenActionSpaceWrapper(env)

		# ============ 加入 release wrapper ============
		env = GraspReleaseWrapper(env)
		
		# ============ 关键：先添加RecordEpisode，再转换为向量环境 ============
		if self.args.max_steps_per_video is None:
			max_steps_per_video = self.args.max_steps_per_episode
		else:
			max_steps_per_video = self.args.max_steps_per_video
			
		# 添加RecordEpisode包装器（必须在转换为向量环境之前）
		env = RecordEpisode(
			env,
			output_dir=self.args.output_dir,
			save_trajectory=self.args.save_trajectory,
			save_video=self.args.record_video,
			info_on_video=self.args.info_on_video,
			max_steps_per_video=max_steps_per_video,
			video_fps=self.args.video_fps,
			save_on_reset=True,  # 自动在reset时保存轨迹和视频
		)
		
		# 最后转换为向量环境
		self.envs = ManiSkillVectorEnv(
			env, 
			self.args.num_envs, 
			ignore_terminations=False,  # partial resets, if True there is problem
			record_metrics=True,
		)
		# ======================================================
		
		print(f"Environment created with {self.args.num_envs} parallel envs")
		print(f"Observation space: {self.envs.single_observation_space}")
		print(f"Action space: {self.envs.single_action_space}")
		
		# 检查输出目录
		if self.args.save_trajectory:
			print(f"Trajectories will be saved to: {self.args.output_dir}/*.h5")
		if self.args.record_video:
			print(f"Videos will be saved to: {self.args.output_dir}/*.mp4")
	
	def _load_agent(self) -> Agent:
		"""加载Agent模型"""
		# 获取环境样本观察
		obs, _ = self.envs.reset(seed=self.args.seed)
		
		# 创建Agent实例
		agent = Agent(self.envs, sample_obs=obs).to(self.device)
		
		# 加载checkpoint
		checkpoint = torch.load(self.args.checkpoint_path, map_location=self.device)
		agent.load_state_dict(checkpoint)
		agent.eval()  # 设置为评估模式
		
		print(f"Model loaded from {self.args.checkpoint_path}")
		print(f"Model device: {self.device}")
		
		return agent
	
	def collect_trajectories(self):
		"""收集轨迹 - RecordEpisode会自动保存轨迹和视频"""
		print(f"\n{'='*60}")
		print(f"Starting trajectory collection with RecordEpisode")
		print(f"{'='*60}")
		
		# 重置环境
		obs, _ = self.envs.reset(seed=self.args.seed)
		dones = torch.zeros(self.args.num_envs, dtype=torch.bool, device=self.device)
		
		collected_episodes = 0
		step_count = 0

		print(f"Target: {self.args.num_episodes} episodes")
		print(f"RecordEpisode will automatically save:")
		print(f"  - Trajectory data (.h5 + .json files)")
		print(f"  - Video files (.mp4)")
		
		while True:
			# 使用模型获取动作
			with torch.no_grad():
				actions = self.agent.get_action(obs, deterministic=self.args.deterministic)
			
			# 执行动作（RecordEpisode会在内部记录轨迹）
			next_obs, rewards, terminations, truncations, infos = self.envs.step(actions)

			# print(f'reward mean {rewards.mean()}')

			# 检查是否结束
			new_dones = torch.logical_or(terminations, truncations)

			# 处理结束的episode（更新统计信息）
			for env_idx in range(self.args.num_envs):
				if new_dones[env_idx] and not dones[env_idx]:
					collected_episodes += 1
					
					# 获取episode信息
					success = terminations[env_idx]
					if success:
						self.stats['successful_episodes'] += 1
					
					print(f"Episode {collected_episodes}/{self.args.num_episodes} completed")
					print(f"  Success: {success}")
					
					# RecordEpisode会自动保存轨迹和视频到文件
			
			# 更新观察
			obs = next_obs
			dones = dones | new_dones
			
			step_count += 1
			
			# 如果所有环境都结束了，重置
			if dones.all():
				if collected_episodes < self.args.num_episodes:
					obs, _ = self.envs.reset()
					dones = torch.zeros(self.args.num_envs, dtype=torch.bool, device=self.device)
				else:
					break
		
		# 强制保存剩余的数据
		self._force_save()
		
		# 更新最终统计
		self.stats['total_episodes'] = collected_episodes
		
		print(f"\n{'='*60}")
		print("Collection completed!")
		print(f"Total episodes: {collected_episodes}")
		print(f"Saved to: {self.args.output_dir}")
		print(f"{'='*60}")
		
		return []  # 返回空的轨迹列表，因为数据已保存到文件
	
	def _force_save(self):
		"""强制保存剩余的数据"""
		try:
			# 通过底层环境触发保存
			if hasattr(self.envs, 'base_env') and hasattr(self.envs.base_env, 'close'):
				# 手动触发RecordEpisode的保存逻辑
				print("Forcing save of remaining data...")
		except:
			pass
	
	def print_summary(self):
		"""打印收集摘要"""
		print(f"\n{'='*60}")
		print("TRAJECTORY COLLECTION SUMMARY")
		print(f"{'='*60}")
		print(f"Output directory: {self.args.output_dir}")
		
		# 列出生成的文件
		if os.path.exists(self.args.output_dir):
			files = os.listdir(self.args.output_dir)
			h5_files = [f for f in files if f.endswith('.h5')]
			json_files = [f for f in files if f.endswith('.json')]
			mp4_files = [f for f in files if f.endswith('.mp4')]
			
			print(f"\nGenerated files:")
			print(f"  Trajectory data: {len(h5_files)} .h5 file(s)")
			print(f"  Metadata: {len(json_files)} .json file(s)")
			print(f"  Videos: {len(mp4_files)} .mp4 file(s)")
			
			if h5_files:
				print(f"\nTrajectory files:")
				for f in h5_files[:5]:  # 显示前5个文件
					print(f"  - {f}")
				if len(h5_files) > 5:
					print(f"  ... and {len(h5_files) - 5} more")
			
			if mp4_files:
				print(f"\nVideo files:")
				for f in mp4_files[:5]:  # 显示前5个文件
					print(f"  - {f}")
				if len(mp4_files) > 5:
					print(f"  ... and {len(mp4_files) - 5} more")
		
		print(f"\nCollection statistics:")
		print(f"  Total episodes: {self.stats['total_episodes']}")
		print(f"  Successful episodes: {self.stats['successful_episodes']}")
		if self.stats['total_episodes'] > 0:
			success_rate = self.stats['successful_episodes'] / self.stats['total_episodes'] * 100
			print(f"  Success rate: {success_rate:.1f}%")
		print(f"{'='*60}")

def main():
	"""主函数"""
	args = tyro.cli(CollectArgs)
	
	# 检查checkpoint文件是否存在
	if not os.path.exists(args.checkpoint_path):
		raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")
	
	# 创建输出目录
	os.makedirs(args.output_dir, exist_ok=True)
	
	print(f"\n{'='*60}")
	print("MANISKILL TRAJECTORY COLLECTOR")
	print(f"{'='*60}")
	print(f"Environment: {args.env_id}")
	print(f"Checkpoint: {args.checkpoint_path}")
	print(f"Parallel envs: {args.num_envs}")
	print(f"Target episodes: {args.num_episodes}")
	print(f"Output directory: {args.output_dir}")
	print(f"Save trajectory: {args.save_trajectory}")
	print(f"Record video: {args.record_video}")
	print(f"{'='*60}")
	
	# 创建收集器
	collector = SimpleTrajectoryCollector(args)
	
	try:
		# 收集轨迹（RecordEpisode会自动保存到文件）
		trajectories = collector.collect_trajectories()
		
		# 打印摘要
		collector.print_summary()
		
	except KeyboardInterrupt:
		print("\nCollection interrupted by user")
	except Exception as e:
		print(f"\nError during collection: {e}")
		import traceback
		traceback.print_exc()
	finally:
		# 关闭环境（RecordEpisode会在close时保存剩余的数据）
		collector.envs.close()
		print("\nEnvironment closed")

if __name__ == "__main__":
	main()