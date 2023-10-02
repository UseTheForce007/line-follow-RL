"""Line Following Robot in Webots with Gym Interface."""
import sys
from controller import Supervisor

try:
    import gym
    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    sys.exit(
        'Please make sure you have all dependencies installed. '
        'Run: "pip3 install numpy gym==0.21 stable_baselines3"'
    )

from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.noise import NormalActionNoise
from gym import spaces


# Create a custom Gym environment for the line-following task
class LineFollowingEnv(Supervisor, gym.Env):
    def __init__(self, max_episode_steps=1e9):
        super().__init__()
        
        self.current_time_step = 0

        self.sensor_threshold = 600
        self.max_speed = 6.28

        # Define the action space (motor speeds)

        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # Define the observation space (IR sensor values)
        self.observation_space = spaces.Box(low=0, high=1024, shape=(3,), dtype=np.float32)
        self.state = None
        self.spec = gym.envs.registration.EnvSpec(id='WebotsEnv-v0', max_episode_steps=max_episode_steps)

        # Environment specific
        self.__timestep = int(self.getBasicTimeStep())
        self.__wheels = []
        self.__ground_sensors = []

        # Tools
        self.keyboard = self.getKeyboard()
        self.keyboard.enable(self.__timestep)
        
        self.prev_action = None


    def wait_keyboard(self):
        while self.keyboard.getKey() != ord('Y'):
            super().step(self.__timestep)

    def reset(self):
        # Reset the simulation
        self.simulationResetPhysics()
        self.simulationReset()
        super().step(self.__timestep)
        self.current_time_step = 0

        # Motors
        self.__wheels = []
        for name in ['left wheel motor', 'right wheel motor']:
            wheel = self.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
            self.__wheels.append(wheel)

        # Sensors

        self.__ground_sensors = []
        for name in ['gs0', 'gs1', 'gs2']:
            ground_sensor = self.getDevice(name)
            ground_sensor.enable(self.__timestep)
            self.__ground_sensors.append(ground_sensor)


        # Internals
        super().step(self.__timestep)

        # Open AI Gym generic
        return np.array([0, 0, 0]).astype(np.float32)
    
    def step(self, action):
        # Execute the action
        self.current_time_step += 1
        
        self.__wheels[0].setVelocity(action[0] * self.max_speed)
        self.__wheels[1].setVelocity(action[1] * self.max_speed)
        super().step(self.__timestep)
        robot = self.getSelf()
        self.state = np.array([self.__ground_sensors[0].getValue(), self.__ground_sensors[1].getValue(), self.__ground_sensors[2].getValue()])

        if self.state[2] > self.sensor_threshold:
            reward = -2
        else:
            reward = 0

        done = False
        
        time_penalty = 1
        reward -= time_penalty
                
        speed_penalty = 3 # Adjust the penalty magnitude as needed
        reward -= speed_penalty * (self.max_speed - min(self.__wheels[0].getVelocity(), self.__wheels[1].getVelocity()))
        
        # Calculate smoother motion penalty
        smoother_motion_penalty = 0.1 # Adjust the penalty magnitude as needed
        if self.prev_action is not None:
            action_difference = np.abs(action - self.prev_action).sum()
            reward -= smoother_motion_penalty * action_difference

        #if self.state[0] > 900 and self.state[2] > 900:
           # reward = 0
            #done = True
        
        if self.current_time_step >= 500:
            done = True
        self.prev_action = action

        return self.state.astype(np.float32), reward, done, {}
       
class ActionNoiseWrapper(gym.ActionWrapper):
    def __init__(self, env, action_noise):
        super().__init__(env)
        self.action_noise = action_noise

    def step(self, action):
        # Apply action noise to the action
        noisy_action = action + self.action_noise()
        # Clip the noisy action to the action space bounds
        noisy_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)
        # Call the original environment's step method with the noisy action
        next_state, reward, done, info = self.env.step(noisy_action)
        return next_state, reward, done, info
        
        
def main():
    # Initialize the environment
    env = LineFollowingEnv()
    check_env(env)


    # Define the exploration noise
    exploration_std = 0.01  # Adjust this value as needed
    action_noise = NormalActionNoise(mean=np.zeros(2), sigma=np.array([exploration_std, exploration_std]))

    # Wrap your environment with the action noise
    env = ActionNoiseWrapper(env, action_noise)
    # Train
    model = PPO('MlpPolicy', env, n_steps=2048, verbose=1)
    model.learn(total_timesteps=1e5)

    # Replay
    print('Training is finished, press `Y` for replay...')
    env.wait_keyboard()

    obs = env.reset()

    
    for _ in range(100000):
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            obs = env.reset()


if __name__ == '__main__':
    main()
