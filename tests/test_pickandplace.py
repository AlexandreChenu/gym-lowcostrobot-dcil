import gymnasium as gym
import pytest
from gymnasium.utils.env_checker import check_env

import mujoco
import gym_lowcostrobot  # noqa

from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np

import time

def read_dataset():

    REPO_ID = "lerobot/koch_pick_place_1_lego"
    FILENAME = "data.csv"

    dataset = pd.read_csv(
        hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
    )

    print(dataset)

def do_env_sim():
    env = gym.make("PickPlaceCube-v0", render_mode="human")#, observation_mode="state", render_mode="human", action_mode="ee")
    env.reset()

    max_step = 10000
    for _ in range(max_step):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        # print("Observation:", observation)
        # print("Reward:", reward)

        env.render()
        if terminated:
            if not truncated:
                print(f"Cube reached the target position at step: {env.current_step} with reward {reward}")
            else:
                print(
                    f"Cube didn't reached the target position at step: {env.current_step} with reward {reward} but was truncated"
                )
            env.reset()

def test_demo():

    df_demo = pd.read_parquet('/Users/achenu/Documents/Research/robotics/github_repos/gym-lowcostrobot/demos/train-00000-of-00001_simu.parquet', engine='pyarrow')
    
    list_states = list(df_demo["observation.state"])
    list_actions = list(df_demo["action"])

    env = gym.make("PickPlaceCube-v0", observation_mode="state", render_mode="human", action_mode="joint")#, observation_mode="state", render_mode="human", action_mode="ee")
    env.reset()

    cube_pos = env.np_random.uniform(env.cube_low, env.cube_high)
    cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
    robot_qpos = list_states[0]
    env.data.qpos[env.arm_dof_id:env.arm_dof_id+env.nb_dof] = robot_qpos
    env.data.qpos[env.cube_dof_id:env.cube_dof_id + 7] = np.concatenate([cube_pos, cube_rot])

    # Sample the target position
    env.target_pos = env.np_random.uniform(env.target_low, env.target_high).astype(np.float32)

    # update visualization
    env.model.geom('target_region').pos = env.target_pos[:]

    mujoco.mj_forward(env.model, env.data)

    # for action in list_actions: 
    #     observation, reward, terminated, truncated, info = env.step(action)

    #     # print("Observation:", observation)
    #     # print("Reward:", reward)

    #     env.render()
    #     if terminated:
    #         if not truncated:
    #             print(f"Cube reached the target position at step: {env.current_step} with reward {reward}")
    #         else:
    #             print(
    #                 f"Cube didn't reached the target position at step: {env.current_step} with reward {reward} but was truncated"
    #             )
    #         env.reset()

    for state in list_states: 
        time.sleep(0.02)
        env.data.qpos[env.arm_dof_id:env.arm_dof_id+env.nb_dof] = state
        env.data.qpos[env.cube_dof_id:env.cube_dof_id + 7] = np.concatenate([cube_pos, cube_rot])

        mujoco.mj_forward(env.model, env.data)

        # print("Observation:", observation)
        # print("Reward:", reward)

        env.render()

def test_IK():

    env = gym.make("PickPlaceCube-v0", observation_mode="state", render_mode="human", action_mode="joint")#, observation_mode="state", render_mode="human", action_mode="ee")
    env.reset()

    ee_id = env.model.body("link_6").id
    ee_pos = env.data.xpos[ee_id]
    # desired_ee_pos = ee_pos.copy() + np.array([-0.1, 0.1, 0.])
    desired_ee_pos = env.data.xpos[env.model.body("cube").id].copy()
    desired_ee_pos[-1] = ee_pos[-1]
    # desired_ee_pos = env.np_random.uniform(env.cube_low, env.cube_high)
    env.inverse_kinematics_GD(desired_ee_pos)

    time.sleep(2)

def test_waypoints_follow():

    env = gym.make("PickPlaceCube-v0", observation_mode="state", render_mode="human", action_mode="joint")#, observation_mode="state", render_mode="human", action_mode="ee")
    env.reset()

    list_waypoints_approach = []
    list_waypoints_grasping = []
    list_waypoints_return = []
    
    ## Phase 1 -> approach
    
    # position above cube 
    ee_initial_pos = env.data.xpos[env.model.body("link_6").id].copy()
    # desired_ee_pos = env.data.xpos[env.model.body("cube").id].copy()
    # desired_ee_pos[-1] = ee_initial_pos[-1].copy()
    # list_waypoints_approach.append(desired_ee_pos.copy())

    # get closer to the cube 
    desired_ee_pos = env.data.xpos[env.model.body("cube").id].copy()
    desired_ee_pos[-1] += 0.15
    desired_ee_pos[0] += 0.01
    desired_ee_pos[1] += 0.02
    list_waypoints_approach.append(desired_ee_pos.copy())
    list_waypoints_return.append(desired_ee_pos.copy())



    # place wirst above the cube
    for waypoint in list_waypoints_approach: 
        # desired_ee_pos = env.np_random.uniform(env.cube_low, env.cube_high)
        env.inverse_kinematics_GD(waypoint, joint_name="link_4")

    ## Phase 2 -> grasping 

    

    # position gripper in "from above position"


    # open the gripper
    env.open_gripper(nb_steps_open=200)
    # time.sleep(1.)

    # position the gripper around the object
    desired_ee_pos = env.data.xpos[env.model.body("cube").id].copy()

    desired_ee_pos[0] += 0.005
    desired_ee_pos[1] += 0.02
    desired_ee_pos[-1] += 0.023
    
    list_waypoints_grasping.append(desired_ee_pos.copy())

    for waypoint in list_waypoints_grasping: 
        # desired_ee_pos = env.np_random.uniform(env.cube_low, env.cube_high)
        env.inverse_kinematics_GD(waypoint, joint_name="link_6")

    # close the gripper 
    env.close_gripper(nb_steps_close=200)
    # time.sleep(1.)

    # Phase 3 -> moving the cube back to initial position
    
    desired_ee_pos = ee_initial_pos
    # list_waypoints_return.append(desired_ee_pos.copy())

    for waypoint in list_waypoints_return: 
        # desired_ee_pos = env.np_random.uniform(env.cube_low, env.cube_high)
        env.inverse_kinematics_GD(waypoint, "link_6")    

    print("total steps = ", env.total_steps)

    time.sleep(2)


if __name__ == "__main__":
    # test_demo()
    # read_dataset()
    # do_env_sim()
    # test_IK()
    test_waypoints_follow()