# eai project

## Install

1. Execute `pip install -e ./ManiSkill`
2. Install Vulkan, refer to https://autodl.com/docs/vulkan/

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

## Arrange Evaluating

For the default environment `eval_arrange.py`, initialized cube colors are `R G B` in order, and your input is the desired final order.

--------------

```python
self.cube_perm_idx[env_idx] = torch.randint(low=0, high=1, size=(b,), device=device) # Change to 6 if random initial color permutation
```

Modify line 200 in `so101_arrange_eval.py` with `high=6` to test random initial state. (Note that for easier controlling parallel environments, the sequence of swaps are still designed for `RGB->Input`, same for the following)

------------

Modify `import so101_arrange_eval` to `import so101_arrange_color` and `env_id: str = "ArrangeCubeSO101Eval-v0"` to `env_id: str = "ArrangeCubeSO101Color-v0"` in `eval_arrange.py`, to test the generalization of the task with different colors.