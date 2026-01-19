# EAI Project

## Install

1. Execute `pip install -e ./ManiSkill`
2. Install Vulkan, refer to https://autodl.com/docs/vulkan/ if using AutoDL.

## Experiments

Below are the success rates of each benchmark task in **simulation**:

| Task  | Success Rate |
| ----- | ------------ |
| Lift  | $81.7\%$     |
| Stack | $66.7\%$     |
| Sort  | $79\%$       |

The trajectories are stored in [THU cloud disk](https://cloud.tsinghua.edu.cn/f/34bfd5308045438094a0/). All trajectories are collected **without privileged information**, which is implemented by setting `obs_mode=rgb` so that `obs_mode_struct.use_state=False` in the ManiSkill environments.

## Videos

Some testing videos of evaluation are in the `videos` folder.


## Code Structures

### Environments

+ Environments of benchmark tasks are in `so101_lift_cube.py`, `so101_stack_cube.py` and `so101_sort_cube.py`.

+ The file `so101_lift_cube_v2.py` is only for the `so101_arrange` family to inhere from.

+ `so101_arrange.py`, `so101_arrange_secondary.py` are for training, while `so101_arrange_eval.py` and `so101_arrange_color.py` are for evaluation.

### Training

You can use `*_ppo.py` for training. The code is modified from `ppo_rgb.py` in examples of ManiSkill, adding a bag of tricks including tanh squashing, state running average and std normalization, and LayerNorm before rgb last feature output.

### Control

Inside `grasp_cube/agents/robots/so101` there are `so_101_ee.py` and `so_101_ee_new_rest_qpos.py` for end-effector control and different rest positions.

## Evaluating for task Arrange

For the default environment `eval_arrange.py`, initialized cube colors are `R G B` in order, and your input is the desired final order.

--------------

```python
self.cube_perm_idx[env_idx] = torch.randint(low=0, high=1, size=(b,), device=device) # Change to 6 if random initial color permutation
```

Modify line 200 in `so101_arrange_eval.py` with `high=6` to test random initial state. (Note that for easier controlling parallel environments, the sequence of swaps are still designed for `RGB->Input`, same for the following)

------------

Modify `import so101_arrange_eval` to `import so101_arrange_color` and `env_id: str = "ArrangeCubeSO101Eval-v0"` to `env_id: str = "ArrangeCubeSO101Color-v0"` in `eval_arrange.py`, to test the generalization of the task with different colors.

### Task and Methodology Highlights

The self-defined task, Arrange, introduces several sources of difficulty:

+ Long horizon: The task contains multiple substeps of lifting and placing cubes. Our methodrequires 75 seconds in average if the given permutation is random, with a maximum of 150seconds to finish the task in simulation time.

+ Dual arm:  Both arms occupy overlapping spatial regions, and can not reach the regionfurthest from them. (For the example in the figure, the left arm can not reach the blue cube.)So two-arm coordination is required for this task.

+ Instruction-related: The task requires an input as the instruction, which is the desired finalconfiguration for the robot to execute.

+ Generalization: Our method can be generalized to the task with different color of cubes. (See `arrange_different_color.mp4` in `videos`)

