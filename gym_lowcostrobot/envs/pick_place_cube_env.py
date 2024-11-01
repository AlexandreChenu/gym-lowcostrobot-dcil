import os, time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import Env, spaces

from gym_lowcostrobot import ASSETS_PATH, BASE_LINK_NAME

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PickPlaceCubeEnv(Env):
    """
    ## Description

    The robot has to pick and place a cube with its end-effector.

    ## Action space

    Two action modes are available: "joint" and "ee". In the "joint" mode, the action space is a 6-dimensional box
    representing the target joint angles.

    | Index | Action              | Type (unit) | Min  | Max |
    | ----- | ------------------- | ----------- | ---- | --- |
    | 0     | Shoulder pan joint  | Float (rad) | -1.0 | 1.0 |
    | 1     | Shoulder lift joint | Float (rad) | -1.0 | 1.0 |
    | 2     | Elbow flex joint    | Float (rad) | -1.0 | 1.0 |
    | 3     | Wrist flex joint    | Float (rad) | -1.0 | 1.0 |
    | 4     | Wrist roll joint    | Float (rad) | -1.0 | 1.0 |
    | 5     | Gripper joint       | Float (rad) | -1.0 | 1.0 |

    In the "ee" mode, the action space is a 4-dimensional box representing the target end-effector position and the
    gripper position.

    | Index | Action        | Type (unit) | Min  | Max |
    | ----- | ------------- | ----------- | ---- | --- |
    | 0     | X             | Float (m)   | -1.0 | 1.0 |
    | 1     | Y             | Float (m)   | -1.0 | 1.0 |
    | 2     | Z             | Float (m)   | -1.0 | 1.0 |
    | 5     | Gripper joint | Float (rad) | -1.0 | 1.0 |

    ## Observation space

    The observation space is a dictionary containing the following subspaces:

    - `"arm_qpos"`: the joint angles of the robot arm in radians, shape (6,)
    - `"arm_qvel"`: the joint velocities of the robot arm in radians per second, shape (6,)
    - `"target_pos"`: the position of the target, as (x, y, z)
    - `"image_front"`: the front image of the camera of size (240, 320, 3)
    - `"image_top"`: the top image of the camera of size (240, 320, 3)
    - `"cube_pos"`: the position of the cube, as (x, y, z)

    Three observation modes are available: "image" (default), "state", and "both".

    | Key             | `"image"` | `"state"` | `"both"` |
    | --------------- | --------- | --------- | -------- |
    | `"arm_qpos"`    | ✓         | ✓         | ✓        |
    | `"arm_qvel"`    | ✓         | ✓         | ✓        |
    | `"target_pos"`  | ✓         | ✓         | ✓        |
    | `"image_front"` | ✓         |           | ✓        |
    | `"image_top"`   | ✓         |           | ✓        |
    | `"cube_pos"`    |           | ✓         | ✓        |

    ## Reward

    The reward is the negative distance between the cube and the target position.

    ## Arguments

    - `observation_mode (str)`: the observation mode, can be "image", "state", or "both", default is "image", see
        section "Observation space".
    - `action_mode (str)`: the action mode, can be "joint" or "ee", default is "joint", see section "Action space".
    - `render_mode (str)`: the render mode, can be "human" or "rgb_array", default is None.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 200}

    def __init__(self, observation_mode="state", action_mode="joint", render_mode=None):
        # Load the MuJoCo model and data
        self.model = mujoco.MjModel.from_xml_path(os.path.join(ASSETS_PATH, "pick_place_cube.xml"), {})
        self.data = mujoco.MjData(self.model)

        # Set the action space
        self.action_mode = action_mode
        action_shape = {"joint": 6, "ee": 4}[action_mode]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_shape,), dtype=np.float32)

        self.nb_dof = 6

        # Set the observations space
        self.observation_mode = observation_mode
        observation_subspaces = {
            "arm_qpos": spaces.Box(low=-np.pi, high=np.pi, shape=(6,)),
            "arm_qvel": spaces.Box(low=-10.0, high=10.0, shape=(6,)),
            "target_pos": spaces.Box(low=-10.0, high=10.0, shape=(3,)),
        }
        if self.observation_mode in ["image", "both"]:
            observation_subspaces["image_front"] = spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
            observation_subspaces["image_top"] = spaces.Box(0, 255, shape=(240, 320, 3), dtype=np.uint8)
            self.renderer = mujoco.Renderer(self.model)
        if self.observation_mode in ["state", "both"]:
            observation_subspaces["cube_pos"] = spaces.Box(low=-10.0, high=10.0, shape=(3,))
            observation_subspaces["ee_pos"] = spaces.Box(low=-10.0, high=10.0, shape=(3,))
        self.observation_space = gym.spaces.Dict(observation_subspaces)

        # Set the render utilities
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if self.render_mode == "human":
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.azimuth = -75
            self.viewer.cam.distance = 1
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer = mujoco.Renderer(self.model, height=640, width=640)

        # Set additional utils
        self.threshold_height = 0.5
        self.cube_low = np.array([-0.15, 0.10, 0.015])
        self.cube_high = np.array([0.15, 0.25, 0.015])
        self.target_low = np.array([-0.15, 0.10, 0.005])
        self.target_high = np.array([0.15, 0.25, 0.005])

        # get dof addresses
        self.cube_dof_id = self.model.body("cube").dofadr[0]
        self.arm_dof_id = self.model.body(BASE_LINK_NAME).dofadr[0]
        self.arm_dof_vel_id = self.arm_dof_id
        # if the arm is not at address 0 then the cube will have 7 states in qpos and 6 in qvel
        if self.arm_dof_id != 0:
            self.arm_dof_id = self.arm_dof_vel_id + 1

        self.control_decimation = 10 # number of simulation steps per control step

        self.tolerance = 0.015

        self.total_steps = 0

    def inverse_kinematics(self, ee_target_pos, step=0.2, joint_name="link_6", nb_dof=6, regularization=1e-6):
        """
        Computes the inverse kinematics for a robotic arm to reach the target end effector position.

        :param ee_target_pos: numpy array of target end effector position [x, y, z]
        :param step: float, step size for the iteration
        :param joint_name: str, name of the end effector joint
        :param nb_dof: int, number of degrees of freedom
        :param regularization: float, regularization factor for the pseudoinverse computation
        :return: numpy array of target joint positions
        """
        try:
            # Get the joint ID from the name
            joint_id = self.model.body(joint_name).id
        except KeyError:
            raise ValueError(f"Body name '{joint_name}' not found in the model.")

        # Get the current end effector position
        # ee_pos = self.d.geom_xpos[joint_id]
        ee_id = self.model.body(joint_name).id
        ee_pos = self.data.geom_xpos[ee_id]

        # Compute the Jacobian
        jac = np.zeros((3, self.model.nv))
        mujoco.mj_jacBodyCom(self.model, self.data, jac, None, joint_id)

        # Compute the difference between target and current end effector positions
        delta_pos = ee_target_pos - ee_pos

        # Compute the pseudoinverse of the Jacobian with regularization
        jac_reg = jac[:, :nb_dof].T @ jac[:, :nb_dof] + regularization * np.eye(nb_dof)
        jac_pinv = np.linalg.inv(jac_reg) @ jac[:, :nb_dof].T

        # Compute target joint velocities
        qdot = jac_pinv @ delta_pos

        # Normalize joint velocities to avoid excessive movements
        qdot_norm = np.linalg.norm(qdot)
        if qdot_norm > 1.0:
            qdot /= qdot_norm

        # Read the current joint positions
        qpos = self.data.qpos[self.arm_dof_id:self.arm_dof_id+nb_dof]

        # Compute the new joint positions
        q_target_pos = qpos + qdot * step

        return q_target_pos


    def check_joint_limits(self, q):
        """Check if the joints is under or above its limits"""

        target_low = np.array([-3.14159, -1.5708, -1.48353, -1.91986, -2.96706, -1.74533])
        target_high = np.array([3.14159, 1.22173, 1.74533, 1.91986, 2.96706, 0.0523599])
        target_qpos = np.array(q).clip(target_low, target_high)

    def plot_ee_traj(self, list_ee_pos, ee_pos_start, ee_target_pos):
        # Separate the points into x, y, and z coordinates
        x = [p[0] for p in list_ee_pos]
        y = [p[1] for p in list_ee_pos]
        z = [p[2] for p in list_ee_pos]

        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z, c='b', marker='.')  # Customize color and marker as needed

        ax.scatter(ee_pos_start[0], ee_pos_start[1], ee_pos_start[2], c="r", marker="o")
        ax.scatter(ee_target_pos[0], ee_target_pos[1], ee_target_pos[2], c="orange", marker="o")

        # Label axes
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()

    def open_gripper(self, nb_steps_open=10):

        new_ctrl = np.zeros((6,))
        new_ctrl[-3] = -np.pi/2
        self.data.ctrl += new_ctrl

        list_observations = []

        for _ in range(nb_steps_open):
            # time.sleep(0.01)
            print("rotating...")

            list_observations.append(self.get_observation_().copy())
            
            for _ in range(self.control_decimation):
                mujoco.mj_step(self.model, self.data)
                
                if self.render_mode == "human":
                    self.viewer.sync()

            if self.render_mode == "human":
                self.viewer.sync()

        # self.data.ctrl[-3] = 0.

        new_ctrl = np.zeros((6,))
        new_ctrl[-1] = -np.pi/2
        self.data.ctrl += new_ctrl
        
        for _ in range(nb_steps_open):
            # time.sleep(0.01)
            print("openning...")

            list_observations.append(self.get_observation_().copy())

            for _ in range(self.control_decimation):
                mujoco.mj_step(self.model, self.data)
                
                if self.render_mode == "human":
                    self.viewer.sync()

            self.total_steps += 1

            if self.render_mode == "human":
                self.viewer.sync()

        return list_observations

    def close_gripper(self, nb_steps_close=10):
        

        # new_ctrl = np.zeros((6,))
        # new_ctrl[-1:] = 3*np.pi/6
        self.data.ctrl[-1] = np.pi/6

        list_observations = []

        for _ in range(nb_steps_close):
            # time.sleep(0.01)
            print("openning...")

            list_observations.append(self.get_observation_().copy())

            for _ in range(self.control_decimation):
                mujoco.mj_step(self.model, self.data)
                
                if self.render_mode == "human":
                    self.viewer.sync()

            self.total_steps += 1

            if self.render_mode == "human":
                self.viewer.sync()

        return list_observations 
    

    def inverse_kinematics_GD(self, ee_target_pos, step=0.2, joint_name="link_4", nb_dof=6, regularization=1e-6):
        """
        Computes the inverse kinematics for a robotic arm to reach the target end effector position.

        :param ee_target_pos: numpy array of target end effector position [x, y, z]
        :param step: float, step size for the iteration
        :param joint_name: str, name of the end effector joint
        :param nb_dof: int, number of degrees of freedom
        :param regularization: float, regularization factor for the pseudoinverse computation
        :return: numpy array of target joint positions
        """
        try:
            # Get the joint ID from the name
            joint_id = self.model.body(joint_name).id
        except KeyError:
            raise ValueError(f"Body name '{joint_name}' not found in the model.")

        # Get the current end effector position
        # ee_pos = self.d.geom_xpos[joint_id]
        ee_id = self.model.body(joint_name).id
        ee_pos = self.data.xpos[ee_id]

        ee_pos_start = ee_pos.copy()

        # Compute the difference between target and current end effector positions
        delta_pos = ee_target_pos - ee_pos
        print("Error = ", np.linalg.norm(delta_pos))

        self.alpha = 1.
        self.step_size = 1.

        list_observations = []

        list_ee_pos = []

        nb_steps = 0
        while(np.linalg.norm(delta_pos) >= self.tolerance and nb_steps <= 1000):

            list_observations.append(self.get_observation_().copy())

            # time.sleep(0.01)

            print("Error = ", np.linalg.norm(delta_pos))

            # Compute the Jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBodyCom(self.model, self.data, jacp, jacr, joint_id)

            grad = self.alpha * jacp.T @ delta_pos 

            self.data.ctrl += self.step_size*grad[:6]#np.concatenate((self.step_size * grad, np.array([0])))
            
            # self.data.ctrl[-2:] -= 0.01
            # self.check_joint_limits(self.data.qpos[6:12])

            # mujoco.mj_step(self.model, self.data)
            for _ in range(self.control_decimation):
                mujoco.mj_step(self.model, self.data)
                
                if self.render_mode == "human":
                    self.viewer.sync()
            self.total_steps += 1

            if self.render_mode == "human":
                self.viewer.sync()

            # Update error
            ee_id = self.model.body(joint_name).id
            ee_pos = self.data.xpos[ee_id]
            delta_pos = ee_target_pos - ee_pos

            list_ee_pos.append(ee_pos.copy())

            nb_steps += 1
            print(nb_steps)

        # diplay trajectories 
        if self.render_mode == None: 
            self.plot_ee_traj(list_ee_pos, ee_pos_start, ee_target_pos)

        return list_observations 

    def apply_action(self, action):
        """
        Step the simulation forward based on the action

        Action shape
        - EE mode: [dx, dy, dz, gripper]
        - Joint mode: [q1, q2, q3, q4, q5, q6, gripper]
        """
        if self.action_mode == "ee":
            # raise NotImplementedError("EE mode not implemented yet")
            ee_action, gripper_action = action[:3], action[-1]

            # Update the robot position based on the action
            ee_id = self.model.body("link_6").id
            ee_target_pos = self.data.xpos[ee_id] + ee_action
            # ee_target_pos =  ee_action

            # Use inverse kinematics to get the joint action wrt the end effector current position and displacement
            target_qpos = self.inverse_kinematics(ee_target_pos=ee_target_pos)
            target_qpos[-1:] = gripper_action

        elif self.action_mode == "joint":
            action, gripper_action = action[:6], action[-1]
            target_low = np.array([-3.14159, -1.5708, -1.48353, -1.91986, -2.96706, -1.74533])
            target_high = np.array([3.14159, 1.22173, 1.74533, 1.91986, 2.96706, 0.0523599])
            target_qpos = np.array(action).clip(target_low, target_high)
            # target_qpos = action.copy()
        else:
            raise ValueError("Invalid action mode, must be 'ee' or 'joint'")

        # Set the target position
        self.data.ctrl = target_qpos

        # Step the simulation forward
        for _ in range(self.control_decimation):
            mujoco.mj_step(self.model, self.data)
            if self.render_mode == "human":
                self.viewer.sync()

    def get_observation_(self):
        # qpos is [x, y, z, qw, qx, qy, qz, q1, q2, q3, q4, q5, q6, gripper]
        # qvel is [vx, vy, vz, wx, wy, wz, dq1, dq2, dq3, dq4, dq5, dq6, dgripper]
        observation = {
            "arm_qpos": self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof].astype(np.float32),
            "arm_qvel": self.data.qvel[self.arm_dof_vel_id:self.arm_dof_vel_id+self.nb_dof].astype(np.float32),
            "target_pos": self.target_pos,
        }
        if self.observation_mode in ["image", "both"]:
            self.renderer.update_scene(self.data, camera="camera_front")
            observation["image_front"] = self.renderer.render()
            self.renderer.update_scene(self.data, camera="camera_top")
            observation["image_top"] = self.renderer.render()
        if self.observation_mode in ["state", "both"]:
            observation["cube_pos"] = self.data.qpos[self.cube_dof_id:self.cube_dof_id+3].astype(np.float32)
            observation["ee_pos"] = self.data.xpos[self.model.body("link_6").id].astype(np.float32)

        return observation

    def reset_model(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed, options=options)

        # Reset the robot to the initial position and sample the cube position
        # cube_pos = self.np_random.uniform(self.cube_low, self.cube_high)
        cube_pos = np.array([0.0, 0.13, 0.015])
        cube_rot = np.array([1.0, 0.0, 0.0, 0.0])
        robot_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.data.qpos[self.arm_dof_id:self.arm_dof_id+self.nb_dof] = robot_qpos
        self.data.qpos[self.cube_dof_id:self.cube_dof_id + 7] = np.concatenate([cube_pos, cube_rot])

        # Sample the target position
        self.target_pos = self.np_random.uniform(self.target_low, self.target_high).astype(np.float32)

        # update visualization
        self.model.geom('target_region').pos = self.target_pos[:]

        # Step the simulation
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation_(), {}
    
    def reset(self, seed=None, options=None):
        return self.reset_model(seed=seed, options=options)

    def step(self, action):
        # Perform the action and step the simulation
        self.apply_action(action)

        # Get the new observation
        observation = self.get_observation_()

        # Get the position of the cube and the distance between the end effector and the cube
        cube_pos = self.data.qpos[self.cube_dof_id:self.cube_dof_id+3]
        cube_to_target = np.linalg.norm(cube_pos - self.target_pos)

        # Compute the reward
        reward = -cube_to_target
        return observation, reward, False, False, {}

    def render(self):
        if self.render_mode == "human":
            self.viewer.sync()
        elif self.render_mode == "rgb_array":
            self.rgb_array_renderer.update_scene(self.data, camera="camera_vizu")
            return self.rgb_array_renderer.render()

    def close(self):
        if self.render_mode == "human":
            self.viewer.close()
        if self.observation_mode in ["image", "both"]:
            self.renderer.close()
        if self.render_mode == "rgb_array":
            self.rgb_array_renderer.close()


if (__name__=='__main__'):

	env = PickPlaceCubeEnv()#render_mode = "human")

	obs, info = env.reset()
	# print("obs = ", obs)

	list_ee_pos = []

	for i in range(200):
		
		action = env.action_space.sample()
		# print("sim_state = ", env.sim.get_state())
		obs, _, _, _, _ = env.step(action)

		print("obs = ", obs["ee_pos"])

		list_ee_pos.append(obs["ee_pos"].copy())

		# env.render()

	env.plot_ee_traj(list_ee_pos, list_ee_pos[0], list_ee_pos[-1])
