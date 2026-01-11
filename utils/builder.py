from typing import Optional, Union

import numpy as np
import sapien
import sapien.render

from mani_skill.envs.scene import ManiSkillScene
from mani_skill.utils.building.actor_builder import ActorBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import Array

from mani_skill.utils.building.actors.common import _build_by_type

def build_cube_w_friction(
    scene: ManiSkillScene,
    half_size: float,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
    # 添加物理材质参数
    static_friction: float = 1.0,  # 静摩擦系数
    dynamic_friction: float = 1.0,  # 动摩擦系数
    restitution: float = 0.0,      # 恢复系数（弹性）
):
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=[half_size] * 3,
            # 添加物理材质
            material=sapien.physx.PhysxMaterial(
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
            ),
        )
    builder.add_box_visual(
        half_size=[half_size] * 3,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)

def build_box_w_friction(
    scene: ManiSkillScene,
    half_sizes,
    color,
    name: str,
    body_type: str = "dynamic",
    add_collision: bool = True,
    scene_idxs: Optional[Array] = None,
    initial_pose: Optional[Union[Pose, sapien.Pose]] = None,
    # 新增物理材质参数
    static_friction: float = 1.0,
    dynamic_friction: float = 1.0,
    restitution: float = 0.0,
):  
    builder = scene.create_actor_builder()
    if add_collision:
        builder.add_box_collision(
            half_size=half_sizes,
            material=sapien.physx.PhysxMaterial(
                static_friction=static_friction,
                dynamic_friction=dynamic_friction,
                restitution=restitution,
            ),
        )
    
    builder.add_box_visual(
        half_size=half_sizes,
        material=sapien.render.RenderMaterial(
            base_color=color,
        ),
    )
    
    return _build_by_type(builder, name, body_type, scene_idxs, initial_pose)