import torch
import gymnasium as gym
from dataclasses import dataclass
import mani_skill.envs
from mani_skill.utils.wrappers.record import RecordEpisode

from mani_skill.utils.wrappers.flatten import (
	FlattenRGBDObservationWrapper,
	FlattenActionSpaceWrapper,
)
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

from sort_ppo import Agent

import so101_sort_cube


@dataclass
class EvalArgs:
	red_checkpoint: str = "runs/SortCubePhase1Cam/ckpt_1201.pt"
	green_checkpoint: str = "runs/SortCubePhase2Cam/ckpt_901.pt"

	env_id: str = "SortCubeSO101-v0"
	num_envs: int = 4
	seed: int = 0

	control_mode: str = "pd_ee_delta_pose"
	obs_mode: str = "rgb"
	render_mode: str = "rgb_array"

	deterministic: bool = True
	max_steps: int = 300
	num_episodes: int = 100


def zero_action_for_arm(action, arm_key):
	a = action.clone()
	if arm_key == "so101-0":
		a[:, :7] = 0.0
	elif arm_key == "so101-1":
		a[:, 7:14] = 0.0
	return a


class SuccessRateEvaluator:
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

		env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)

		if isinstance(env.action_space, gym.spaces.Dict):
			env = FlattenActionSpaceWrapper(env)

		env = RecordEpisode(
			env,
			output_dir="./eval_videos",
			save_video=True,
			save_trajectory=True,
			info_on_video=True,
			save_on_reset=True,
			max_steps_per_video=self.args.max_steps,
		)

		self.envs = ManiSkillVectorEnv(
			env, self.args.num_envs, ignore_terminations=True
		)

	def _load_agents(self):
		obs, _ = self.envs.reset(seed=self.args.seed)

		self.red_agent = Agent(self.envs, sample_obs=obs).to(self.device)
		self.green_agent = Agent(self.envs, sample_obs=obs).to(self.device)

		self.red_agent.load_state_dict(
			torch.load(self.args.red_checkpoint, map_location=self.device)
		)
		self.green_agent.load_state_dict(
			torch.load(self.args.green_checkpoint, map_location=self.device)
		)

		self.red_agent.eval()
		self.green_agent.eval()

		print("✅ Red & Green agents loaded")

	def run(self):
		obs, _ = self.envs.reset(seed=self.args.seed)

		step = torch.zeros(self.args.num_envs, dtype=torch.long, device=self.device)

		collected_episodes = 0
		successful_episodes = 0

		while collected_episodes < self.args.num_episodes:

			# ===== phase based on *current* episode step =====
			phase = step >= 150

			with torch.no_grad():
				a_red = self.red_agent.get_action(obs, self.args.deterministic)
				a_red = zero_action_for_arm(a_red, "so101-1")

				a_green = self.green_agent.get_action(obs, self.args.deterministic)
				a_green = zero_action_for_arm(a_green, "so101-0")

				action = torch.where(phase[:, None], a_green, a_red)

			obs, _, terminated, truncated, infos = self.envs.step(action)
			terminated = terminated.to(self.device)
			truncated = truncated.to(self.device)
			done = terminated | truncated


			for i in range(self.args.num_envs):
				if done[i]:
					collected_episodes += 1
					success = infos["final_info"]["success"][i].item()
					successful_episodes += int(success)

					print(
						f"Episode {collected_episodes}/{self.args.num_episodes} "
						f"| Env {i} | Success: {success}"
					)

					step[i] = 0  # IMPORTANT: reset step counter
				else:
					step[i] += 1

			# ===== DO NOT touch obs / action for done envs here =====
			# ManiSkillVectorEnv has already reset them internally

			if collected_episodes > 0 and collected_episodes % 20 == 0:
				rate = successful_episodes / collected_episodes
				print(
					f"[Eval] Episodes: {collected_episodes} "
					f"| Success rate: {rate:.3f}"
				)

		print("\n==============================")
		print(f"✅ Final Success Rate: {successful_episodes / collected_episodes:.4f}")
		print("==============================")

		self.envs.close()


def main():
	evaluator = SuccessRateEvaluator(EvalArgs())
	evaluator.run()


if __name__ == "__main__":
	main()