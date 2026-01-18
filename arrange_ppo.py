from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
# 只保留需要的 wrapper
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper, FlattenRGBDObservationWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

import so101_arrange

# =============================================================================
# 🔧 最终修复：State 包装器 V3
# =============================================================================
class FlattenStateDictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.base_env = env.unwrapped
        
        # 1. 获取真实观测样本
        try:
            obs, _ = self.env.reset()
        except:
            obs = self.env.observation_space.sample()

        self.is_dict_obs = isinstance(obs, dict)

        if self.is_dict_obs:
            self.keys = sorted(obs.keys())
            # 核心修复：直接累加最后一个维度 (Feature Dim)
            # 无论是 (Batch, Dim) 还是 (Dim,)，shape[-1] 都是 Dim
            flat_dim = sum([obs[k].shape[-1] for k in self.keys])
        else:
            self.keys = []
            flat_dim = obs.shape[-1]

        # 2. 定义 single_observation_space (供 PPO 初始化网络使用)
        # 必须是 (38,) 这种纯 Feature 维度
        self.single_observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-np.inf, np.inf, shape=(int(flat_dim),), dtype=np.float32)
        })

        # 3. 定义 observation_space (供 Gym 检查使用)
        # 如果样本也是 batch 的，那空间也设为 batch
        if self.is_dict_obs:
            sample_shape = list(obs[self.keys[0]].shape) # e.g. [16, 7]
        else:
            sample_shape = list(obs.shape)
            
        # 替换最后一维为 flat_dim
        sample_shape[-1] = int(flat_dim)
        
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(-np.inf, np.inf, shape=tuple(sample_shape), dtype=np.float32)
        })

    def observation(self, observation):
        if self.is_dict_obs and isinstance(observation, dict):
            tensors = []
            for k in self.keys:
                t = observation[k]
                # 确保展平除 Batch (dim 0) 外的所有维度 (针对 state 通常不需要，但以防万一)
                if t.ndim > 2: 
                    t = t.reshape(t.shape[0], -1)
                tensors.append(t)
            flat_state = torch.cat(tensors, dim=-1)
            return {"state": flat_state}
        else:
            # 已经是 Tensor 或非 Dict
            if isinstance(observation, dict):
                 return {"state": list(observation.values())[0]} 
            return {"state": observation}
# =============================================================================
# 参数与网络定义
# =============================================================================
@dataclass
class Args:
    exp_name: Optional[str] = 'ArrangeCube'
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: str = "PPO"
    """the group of the run for wandb"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    # checkpoint: Optional[str] = None
    checkpoint: Optional[str] = "runs/ArrangeCubeOk4/ckpt_501.pt"
    # checkpoint: Optional[str] = "runs/ArrangeCubeOld10/ckpt_26.pt"
    # checkpoint: Optional[str] = "runs/ArrangeCubeOld5/ckpt_51.pt"
    """path to a pretrained checkpoint file to start evaluation/training from"""
    render_mode: str = "sensors"
    """the environment rendering mode"""

    # Algorithm specific arguments
    env_id: str = "ArrangeCubeSO101-v0" # modify
    """the id of the environment"""
    include_state: bool = True
    """whether to include state information in observations"""
    obs_mode: str = 'rgb+state'
    """whether to include privileged states, if yes use 'rgb+state'"""
    total_timesteps: int = 30000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 200
    """the number of parallel environments"""
    num_eval_envs: int = 1
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 80
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 1500
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_ee_delta_pose"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.995
    """the discount factor gamma"""
    gae_lambda: float = 0.97
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 5
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.005
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 50
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class DictArray(object):
    def __init__(self, buffer_shape, element_space, data_dict=None, device=None):
        self.buffer_shape = buffer_shape
        if data_dict:
            self.data = data_dict
        else:
            assert isinstance(element_space, gym.spaces.dict.Dict)
            self.data = {}
            for k, v in element_space.items():
                if isinstance(v, gym.spaces.dict.Dict):
                    self.data[k] = DictArray(buffer_shape, v, device=device)
                else:
                    dtype = (torch.float32 if v.dtype in (np.float32, np.float64) else
                            torch.uint8 if v.dtype == np.uint8 else
                            torch.int16 if v.dtype == np.int16 else
                            torch.int32 if v.dtype == np.int32 else
                            v.dtype)
                    self.data[k] = torch.zeros(buffer_shape + v.shape, dtype=dtype, device=device)

    def keys(self):
        return self.data.keys()

    def __getitem__(self, index):
        if isinstance(index, str):
            return self.data[index]
        return {k: v[index] for k, v in self.data.items()}

    def __setitem__(self, index, value):
        if isinstance(index, str):
            self.data[index] = value
        for k, v in value.items():
            self.data[k][index] = v

    @property
    def shape(self):
        return self.buffer_shape

    def reshape(self, shape):
        t = len(self.buffer_shape)
        new_dict = {}
        for k,v in self.data.items():
            if isinstance(v, DictArray):
                new_dict[k] = v.reshape(shape)
            else:
                new_dict[k] = v.reshape(shape + v.shape[t:])
        new_buffer_shape = next(iter(new_dict.values())).shape[:len(shape)]
        return DictArray(new_buffer_shape, None, data_dict=new_dict)

class NatureCNN(nn.Module):
    def __init__(self, sample_obs, feature_size=256, epsilon=1e-8, clip_obs=5.0):
        super().__init__()
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        self.out_features = 0

        if "state" in sample_obs:
            state_dim = sample_obs["state"].shape[-1]

            print(f'State Dimension: {state_dim}')

            self.register_buffer("running_mean", torch.zeros(state_dim))
            self.register_buffer("running_var", torch.ones(state_dim))
            self.register_buffer("count", torch.tensor(1e-4))

            self.state_net = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU()
            )
            self.out_features += feature_size
    
        if "rgb" in sample_obs:
            img_h = sample_obs["rgb"].shape[1]
            img_w = sample_obs["rgb"].shape[2]
            
            def make_cnn():
                return nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=8, stride=4),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1),
                    nn.ReLU(),
                    nn.Flatten(),
                )

            self.cnn_v1 = make_cnn()
            self.cnn_v2 = make_cnn()
            self.cnn_v3 = make_cnn()
            self.cnn_v4 = make_cnn()
            
            with torch.no_grad():
                dummy_img = torch.zeros(1, 3, img_h, img_w)
                n_flatten = self.cnn_v1(dummy_img).shape[1]
            
            # half_feature = feature_size // 2
            
            def make_visual_head():
                return nn.Sequential(
                    nn.Linear(n_flatten, feature_size),
                    nn.LayerNorm(feature_size),
                    nn.ReLU()
                )

            self.rgb_final1 = make_visual_head()
            self.rgb_final2 = make_visual_head()
            self.rgb_final3 = make_visual_head()
            self.rgb_final4 = make_visual_head()
            
            self.out_features += feature_size * 4

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []
        
        if "state" in observations:
            state = observations["state"].float().to(self.running_mean.device)
            
            if self.training:
                with torch.no_grad():
                    batch_mean = state.mean(dim=0)
                    batch_var = state.var(dim=0, unbiased=False)
                    batch_count = state.shape[0]
                    
                    # Welford 算法更新 Running Mean/Var
                    delta = batch_mean - self.running_mean
                    tot_count = self.count + batch_count
                    
                    new_mean = self.running_mean + delta * batch_count / tot_count
                    m_a = self.running_var * self.count
                    m_b = batch_var * batch_count
                    M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
                    
                    self.running_mean.copy_(new_mean)
                    self.running_var.copy_(M2 / tot_count)
                    self.count.copy_(tot_count)

            # 执行标准化和 Clipping
            state = (state - self.running_mean) / torch.sqrt(self.running_var + self.epsilon)
            state = torch.clamp(state, -self.clip_obs, self.clip_obs)
            encoded_tensor_list.append(self.state_net(state))
            
        if "rgb" in observations:
            obs_rgb = observations["rgb"].float().permute(0, 3, 1, 2) / 255.0

            rgb_v1 = obs_rgb[:, 0:3, :, :]
            rgb_v2 = obs_rgb[:, 3:6, :, :]
            rgb_v3 = obs_rgb[:, 6:9, :, :]
            rgb_v4 = obs_rgb[:, 9:12, :, :]
            
            feat_v1 = self.rgb_final1(self.cnn_v1(rgb_v1))
            feat_v2 = self.rgb_final2(self.cnn_v2(rgb_v2))
            feat_v3 = self.rgb_final2(self.cnn_v3(rgb_v3))
            feat_v4 = self.rgb_final2(self.cnn_v4(rgb_v4))
            
            rgb_features = torch.cat([feat_v1, feat_v2, feat_v3, feat_v4], dim=1)
            encoded_tensor_list.append(rgb_features)
            
        return torch.cat(encoded_tensor_list, dim=1)

class Agent(nn.Module):
    def __init__(self, envs, sample_obs, **kwargs):
        super().__init__()
        self.feature_net = NatureCNN(sample_obs=sample_obs)
        latent_size = self.feature_net.out_features
        
        action_dim = np.prod(envs.single_action_space.shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, 1)),
        )
        
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(latent_size, 512)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(512, action_dim), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

        assert isinstance(envs.single_action_space, gym.spaces.Box)
        action_low = envs.single_action_space.low
        action_high = envs.single_action_space.high
        self.register_buffer("action_low", torch.from_numpy(action_low).float())
        self.register_buffer("action_high", torch.from_numpy(action_high).float())

    def get_features(self, x):
        return self.feature_net(x)

    def get_value(self, x):
        features = self.feature_net(x)
        return self.critic(features)

    def denormalize_action(self, a):
        return (a + 1) / 2 * (self.action_high - self.action_low) + self.action_low

    def _squashed_normal(self, mean):
        logstd = self.actor_logstd.expand_as(mean)
        std = torch.exp(logstd)
        base_dist = Normal(mean, std)
        raw = base_dist.rsample()
        squashed = torch.tanh(raw)
        logprob = base_dist.log_prob(raw)
        logprob -= torch.log(1 - squashed.pow(2) + 1e-6)
        logprob = logprob.sum(1)
        entropy = base_dist.entropy().sum(1)
        return squashed, logprob, entropy

    def _logprob_squashed(self, mean: torch.Tensor, action_squashed: torch.Tensor):
        atanh = 0.5 * (torch.log1p(action_squashed + 1e-6) - torch.log1p(-action_squashed + 1e-6))
        std = torch.exp(self.actor_logstd.expand_as(mean))
        base_dist = Normal(mean, std)
        logprob = base_dist.log_prob(atanh) - torch.log(1 - action_squashed.pow(2) + 1e-6)
        return logprob.sum(1)

    def get_action(self, x, deterministic=False):
        features = self.feature_net(x)
        action_mean = self.actor_mean(features)
        if deterministic:
            squashed = torch.tanh(action_mean)
        else:
            squashed, _, _ = self._squashed_normal(action_mean)
        action = self.denormalize_action(squashed)
        return action
    
    def get_action_and_value(self, x, action=None):
        x = self.feature_net(x)
        action_mean = self.actor_mean(x)
        if action is None:
            squashed, logprob, entropy = self._squashed_normal(action_mean)
        else:
            squashed = action
            logprob = self._logprob_squashed(action_mean, squashed)
            entropy = Normal(action_mean, torch.exp(self.actor_logstd.expand_as(action_mean))).entropy().sum(1)
        action_real = self.denormalize_action(squashed)
        return squashed, logprob, entropy, self.critic(x), action_real

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    env_kwargs = dict(
        obs_mode=args.obs_mode, 
        robot_uids="so101", 
        render_mode=args.render_mode, 
        sim_backend="physx_cuda"
    )
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
        
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)

    # Wrapper 应用
    use_rgb = "rgb" in args.obs_mode
    if use_rgb:
        envs = FlattenRGBDObservationWrapper(envs, rgb=True, depth=False, state=args.include_state)
        eval_envs = FlattenRGBDObservationWrapper(eval_envs, rgb=True, depth=False, state=args.include_state)
    else:
        # 使用修复版 State Wrapper
        envs = FlattenStateDictWrapper(envs)
        eval_envs = FlattenStateDictWrapper(eval_envs)

    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
        
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=2000, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=2000, video_fps=30)
    
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group=args.wandb_group,
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    if use_rgb:
        obs = DictArray((args.num_steps, args.num_envs), envs.single_observation_space, device='cpu')
    else:
        # 🔧 FIX: 现在 single_observation_space 一定是 Dict {"state": Box}
        obs_dim = envs.single_observation_space["state"].shape[0]
        # 创建 Obs Buffer: (NumSteps, NumEnvs, ObsDim)
        obs = torch.zeros((args.num_steps, args.num_envs, obs_dim), device='cpu')

    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    
    agent = Agent(envs, sample_obs=next_obs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    cumulative_times = defaultdict(float)

    for iteration in range(1, args.num_iterations + 1):
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            stime = time.perf_counter()
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            t = eval_envs.render()
            print(t.shape)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if logger is not None:
                eval_time = time.perf_counter() - stime
                cumulative_times["eval_time"] += eval_time
                logger.add_scalar("time/eval_time", eval_time, global_step)
            if args.evaluate:
                break
        
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
            
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
            
        rollout_time = time.perf_counter()
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            if use_rgb:
                obs[step] = next_obs
            else:
                # 🔧 FIX: 如果是 dict 包装的 state tensor
                obs[step] = next_obs["state"]

            dones[step] = next_done
            
            with torch.no_grad():
                squashed, logprob, _, value, action_real = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = squashed
            logprobs[step] = logprob
            if torch.isnan(action_real).any() or torch.isinf(action_real).any():
                print("!!! [CRITICAL] 检测到 NaN/Inf 动作，停止运行以保护物理引擎 !!!")
                print("Logprobs:", logprobs[step])
                print("Values:", values[step])
                print("Action Real:", action_real)
                exit()
            next_obs, reward, terminations, truncations, infos = envs.step(action_real)
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)

                # 🔧 FIX: 处理 final_observation
                if use_rgb:
                    for k in infos["final_observation"]:
                        infos["final_observation"][k] = infos["final_observation"][k][done_mask]
                    final_obs_for_val = infos["final_observation"]
                else:
                    # 获取 Done 环境的 State，并确保 shape 匹配
                    f_obs = infos["final_observation"]["state"][done_mask]
                    final_obs_for_val = {"state": f_obs}

                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(final_obs_for_val).view(-1)

        rollout_time = time.perf_counter() - rollout_time
        cumulative_times["rollout_time"] += rollout_time
        
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t]
                delta = rewards[t] + args.gamma * real_next_values - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam
            returns = advantages + values

        if use_rgb:
            b_obs = obs.reshape((-1,))
        else:
            # obs: (steps, envs, dim) -> (steps*envs, dim)
            b_obs = obs.reshape((-1, obs.shape[-1]))
            # 包装成 dict 以喂给 agent
            b_obs = {"state": b_obs}

        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.perf_counter()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                if use_rgb:
                    obs_batch = b_obs[mb_inds]
                    obs_batch_gpu = {}
                    for k, v in obs_batch.items():
                        if isinstance(v, torch.Tensor):
                            obs_batch_gpu[k] = v.to(device)
                        else:
                            obs_batch_gpu[k] = v
                else:
                    # 🔧 FIX: State 模式数据搬运
                    obs_batch_gpu = {"state": b_obs["state"][mb_inds].to(device)}

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    obs_batch_gpu,
                    b_actions[mb_inds]
                )

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        update_time = time.perf_counter() - update_time
        cumulative_times["update_time"] += update_time
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
        for k, v in cumulative_times.items():
            logger.add_scalar(f"time/total_{k}", v, global_step)
        logger.add_scalar("time/total_rollout+update_time", cumulative_times["rollout_time"] + cumulative_times["update_time"], global_step)
    
    if args.save_model and not args.evaluate:
        model_path = f"runs/{run_name}/final_ckpt.pt"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    if logger is not None: logger.close()