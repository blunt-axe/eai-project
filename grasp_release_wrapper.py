import torch
import numpy as np
import gymnasium as gym
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from so101_stack_cube import StackCubeSO101Env

class GraspReleaseWrapper(gym.Wrapper):
    """
    一个包装器，用于在特定步数后强制执行松爪动作。
    支持单环境和向量化环境。
    """
    def __init__(self, env, release_step=150):
        super().__init__(env)
        
        # 尝试多种方式获取并行环境数量
        if hasattr(env, "num_envs"):
            self.num_envs = env.num_envs
        elif hasattr(env, "get_wrapper_attr"):
            try:
                self.num_envs = env.get_wrapper_attr("num_envs")
            except:
                exit()
        else:
            self.num_envs = getattr(env.unwrapped, "num_envs", 1)
            
        print(f"Wrapper initialized with num_envs: {self.num_envs}")
        
        self.current_step = torch.zeros(self.num_envs, dtype=torch.long)
        self.release_step = release_step * torch.ones_like(self.current_step)

    def reset(self, **kwargs):
        if kwargs["options"] is not None:
            env_idx = kwargs["options"]["env_idx"]
            self.current_step[env_idx] = 0
        else:
            self.current_step = torch.zeros(self.num_envs, dtype=torch.long)
            
        return self.env.reset(**kwargs)

    def step(self, action):
        self.current_step = 1 + self.current_step
        
        release_mask = self.current_step >= self.release_step
        action[release_mask, -1] = 1.0
        action[release_mask, :-1] = 0.0
                
        return self.env.step(action)