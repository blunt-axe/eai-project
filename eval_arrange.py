import os
import torch
import gymnasium as gym
from dataclasses import dataclass
import mani_skill.envs

from mani_skill.utils.wrappers.flatten import (
	FlattenRGBDObservationWrapper,
	FlattenActionSpaceWrapper,
)
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode

from arrange_ppo import Agent

import so101_arrange_eval


# =========================
# Args
# =========================

@dataclass
class EvalArgs:
	left_checkpoint: str = "runs/ArrangeCubeOk4/ckpt_301.pt"
	right_checkpoint: str = "runs/ArrangeCubeSecondary/ckpt_801.pt"

	env_id: str = "ArrangeCubeSO101Eval-v0"
	num_envs: int = 8
	seed: int = 0

	control_mode: str = "pd_ee_delta_pose"
	obs_mode: str = "rgb+state"
	render_mode: str = "rgb_array"

	max_steps: int = 4500
	deterministic: bool = True

	output_dir: str = "./arrange_eval"


# =========================
# Utils
# =========================

def zero_action_for_arm(action, arm_key):
	"""
	将某只机械臂 action 清零
	兼容两种情况：
	1. Dict action（未 Flatten）
	2. Tensor action（FlattenActionSpaceWrapper 之后）
	"""
	# ------------------------------------------------------------
	# 情况 1：Dict action（原始 ManiSkill action space）
	# ------------------------------------------------------------
	if isinstance(action, dict):
		a = {}
		for k, v in action.items():
			if k == arm_key:
				a[k] = torch.zeros_like(v)
			else:
				a[k] = v
		return a

	# ------------------------------------------------------------
	# 情况 2：Flatten 后的 Tensor action
	# ------------------------------------------------------------
	if isinstance(action, torch.Tensor):
		a = action.clone()

		# 约定：so101-0 在前 7 维，so101-1 在后 7 维
		if arm_key == "so101-0":
			a[:, :7] = 0.0
		elif arm_key == "so101-1":
			a[:, 7:14] = 0.0
		else:
			raise ValueError(f"Unknown arm_key: {arm_key}")

		return a

	# ------------------------------------------------------------
	# 其他非法情况
	# ------------------------------------------------------------
	raise RuntimeError(f"Unsupported action type: {type(action)}")

# =========================
# Sequential Evaluator
# =========================

class SequentialEvaluator:
	def __init__(self, args: EvalArgs):
		self.args = args
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self._make_env()
		self._load_agents()

	def _make_env(self):
		env = gym.make(
			self.args.env_id,
			num_envs=self.args.num_envs,
			obs_mode=self.args.obs_mode,
			control_mode=self.args.control_mode,
			render_mode=self.args.render_mode,
			reward_mode="dense",
			reconfiguration_freq=0,
		)

		env = FlattenRGBDObservationWrapper(
			env, rgb=True, depth=False, state=True
		)

		if isinstance(env.action_space, gym.spaces.Dict):
			env = FlattenActionSpaceWrapper(env)

		env = RecordEpisode(
			env,
			output_dir=self.args.output_dir,
			save_video=True,
			save_trajectory=False,
			info_on_video=True,
			save_on_reset=True,
			max_steps_per_video=self.args.max_steps,
		)

		self.envs = ManiSkillVectorEnv(
			env, self.args.num_envs, ignore_terminations=True
		)

	def _load_agents(self):
		obs, _ = self.envs.reset(seed=self.args.seed)

		self.left_agent = Agent(self.envs, sample_obs=obs).to(self.device)
		self.right_agent = Agent(self.envs, sample_obs=obs).to(self.device)

		self.left_agent.load_state_dict(
			torch.load(self.args.left_checkpoint, map_location=self.device)
		)
		self.right_agent.load_state_dict(
			torch.load(self.args.right_checkpoint, map_location=self.device)
		)

		self.left_agent.eval()
		self.right_agent.eval()

		print("Red & Green agents loaded.")

	# =========================
	# Main Eval Loop
	# =========================

	def run(self):
		
		target_s=input("Input target permutation: ").upper()
		#target_s="RBG"

		now_s="RGB"

		swap_arm=[]

		p=[0,0,0]

		for i in range(0,3):
			for j in range(0,3):
				if(target_s[j]==now_s[i]):
					p[i]=j
		
		if(p[0]>p[1]):
			swap_arm.append(0)
			p[0],p[1]=p[1],p[0]
		if(p[1]>p[2]):
			swap_arm.append(1)
			p[1],p[2]=p[2],p[1]
		if(p[0]>p[1]):
			swap_arm.append(0)
			p[0],p[1]=p[1],p[0]
		print(p)
		print(swap_arm)

		obs, _ = self.envs.reset(seed=self.args.seed)

		phase = torch.zeros(self.args.num_envs, dtype=torch.long, device=self.device)
		# 0 = RED, 1 = GREEN

		done = torch.zeros(self.args.num_envs, dtype=torch.bool, device=self.device)
		base_env = self.envs._env
		while hasattr(base_env, "env"):
			base_env = base_env.env
		for step in range(1500*len(swap_arm)):
			print(step)
			if step%1500==0:
				base_env.set_active_arm(swap_arm[step//1500])
			with torch.no_grad():

				a0 = self.left_agent.get_action(
					obs, deterministic=self.args.deterministic
				)
				a0 = zero_action_for_arm(a0, "so101-1")  # 冻右侧
				a1 = self.right_agent.get_action(
					obs, deterministic=self.args.deterministic
				)
				a1 = zero_action_for_arm(a1, "so101-0")  # 冻左侧

			if swap_arm[step//1500]==0:
				action = a0
			else:
				action = a1


			obs, rew, terminated, truncated, infos = self.envs.step(action)


			if done.all():
				print(f"Episode success at step {step}")
				break

		self.envs.close()


# =========================
# Main
# =========================

def main():
	args = EvalArgs()
	evaluator = SequentialEvaluator(args)
	evaluator.run()


if __name__ == "__main__":
	main()