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