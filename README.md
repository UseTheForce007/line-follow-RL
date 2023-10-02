# line-follow-RL

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h1> Line Following Robot V-1 </h1>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3> Objective</h3>\n",
    "\n",
    "<p> The objective of this project is to build a line following robot. The robot should be able to follow a black line on a white background. It should be a pure Reinforcement Learning based approach. </p>"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3> Dependencies </h3>\n",
    "\n",
    "<p> The following dependencies are required to run the code. </p>\n",
    "\n",
    "<ul>\n",
    "    <li> python == 3.6 </li>\n",
    "    <li> gym == 0.21.0 </li>\n",
    "    <li> stable-baselines3 == 1.7.0 </li>\n",
    "    <li> numpy == 1.26.0 </li>\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3> Gym Environment </h3>\n",
    "\n",
    "<p> The \"LineFollowingEnv\" is a custom Gym environment designed for the task of training an autonomous robot to follow a line in a simulated environment using reinforcement learning techniques. This environment serves as the foundation for developing and evaluating intelligent control policies for the robot.</p>\n",
    "\n",
    "<h5>Key Features </h5>\n",
    "\n",
    "* Action Space: 2 continous outputs between 0 and 1. It is used to control the speed of the robot. It will act as a pecentage of the maximum speed of the robot. A seperate one for the left and right motor.\n",
    "\n",
    "* Observation Space: 3 continous outputs between 0 and 1024. These represent the readings from the IR sensors on the robot.\n",
    "\n",
    "* Reward Function: The primary objective is to follow the line while minimizing penalties. The reward function is designed to encourage the robot to stay close to the line, move at an appropriate speed, and make smooth, controlled actions.\n",
    "\n",
    "* Simulation: LineFollowingEnv is integrated with the Webots simulator, providing a realistic and dynamic environment for training. The simulation environment mimics real-world conditions, allowing agents to learn and adapt to various scenarios.\n",
    "\n",
    "* Customization: The environment is highly customizable. Users can adjust parameters such as the sensor thresholds, maximum speed of the robot, and penalty magnitudes to tailor the learning task to their specific needs.\n",
    "\n",
    "<h5>Training Objective</h5>\n",
    "\n",
    "The primary training objective in LineFollowingEnv is to develop a reinforcement learning agent capable of effectively navigating the robot along a predefined line path. The agent learns to interpret sensor data, make decisions based on that data, and control the robot's motors to stay on course.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h5> Reset() </h5>\n",
    "\n",
    "The reset() function is called at the beginning of each episode. It is used to initialize the environment and prepare it for training. The function returns the initial observation of the environment."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def reset(self):\n",
    "    # Reset the simulation\n",
    "    self.simulationResetPhysics()\n",
    "    self.simulationReset()\n",
    "    super().step(self.__timestep)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The robot has 2 motors and 3 IR sensors. The inital velocity of each motor is set to 0 and the sensors are enabled to provide output at every simulation time step."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Motors\n",
    "self.__wheels = []\n",
    "for name in ['left wheel motor', 'right wheel motor']:\n",
    "    wheel = self.getDevice(name)\n",
    "    wheel.setPosition(float('inf'))\n",
    "    wheel.setVelocity(0.0)\n",
    "    self.__wheels.append(wheel)\n",
    "\n",
    "# Sensors\n",
    "\n",
    "self.__ground_sensors = []\n",
    "for name in ['gs0', 'gs1', 'gs2']:\n",
    "    ground_sensor = self.getDevice(name)\n",
    "    ground_sensor.enable(self.__timestep)\n",
    "    self.__ground_sensors.append(ground_sensor)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h5> Step() </h5>\n",
    "\n",
    "The step() function is called at each time step of the simulation. It is used to update the environment based on the agent's actions and return the new observation, reward, and done flag."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The action sampled is used to set the velocity of the motors. The velocity of the motors is set to a percentage of the maximum velocity of the robot. The percentage is calculated by multiplying the action sampled by the maximum velocity of the robot.\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "self.__wheels[0].setVelocity(action[0] * self.max_speed)\n",
    "self.__wheels[1].setVelocity(action[1] * self.max_speed)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The observations from the IR sensors are then recieved and added to the state."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "self.state = np.array([self.__ground_sensors[0].getValue(), self.__ground_sensors[1].getValue(), self.__ground_sensors[2].getValue()])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The basic reward function is as follows:\n",
    "\n",
    "If the middle sensor of the robot is on the line , it gets a reward of 0 otherwise -2. The terminal state is reached when the robot is out of the line entirely."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "if self.state[2] > self.sensor_threshold:\n",
    "    reward = -2\n",
    "else:\n",
    "    reward = 0\n",
    "\n",
    "if self.state[0] > 900 and self.state[2] > 900:\n",
    "    reward = 0\n",
    "    done = True"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3> Training </h3>\n",
    "\n",
    "The training is done using the stable-baselines3 library. The PPO algorithm is used to train the agent. \n",
    "\n",
    "<h5> Proximal Policy Optimization </h5>\n",
    "\n",
    "Proximal Policy Optimization (PPO) is a cutting-edge reinforcement learning algorithm known for its robustness, stability, and efficient use of computational resources. PPO belongs to the class of actor-critic methods, which combine value estimation (the critic) and policy improvement (the actor) to train agents in environments where actions influence both immediate rewards and future states.\n",
    "\n",
    "<h5> Key Characteristics of PPO </h5>\n",
    "\n",
    "* Policy Optimization: PPO directly optimizes the policy function, which is a neural network that maps observations to actions. This policy network is responsible for determining the agent's behavior in the environment.\n",
    "\n",
    "* Stability and Trust Region: PPO is designed to ensure that policy updates are performed within a \"trust region\" to prevent overly large policy changes. This constraint helps maintain stability during training and prevents catastrophic policy collapses.\n",
    "\n",
    "* Proximal Objective: PPO uses a proximal objective function, which quantifies the change in the policy's behavior concerning the previous policy. This objective encourages small, controlled policy updates.\n",
    "\n",
    "* Clipped Surrogate Objective: To stay within the trust region, PPO employs a clipped surrogate objective. This objective is a combination of two terms: a surrogate policy improvement term and a clipped version of it. This mechanism enforces a limit on how much the policy can change, enhancing stability."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h5> How it works </h5>\n",
    "\n",
    "1. Data Collection: The agent interacts with the environment to collect a batch of experiences, consisting of observations, actions, rewards, and next-state observations.\n",
    "\n",
    "2. Policy Evaluation: The collected data is used to estimate the advantages of the actions taken. This involves computing the expected return (cumulative rewards) for each state-action pair and subtracting a value estimate of the state's worth (the critic's role).\n",
    "\n",
    "3. Policy Update: PPO calculates the surrogate objective, which measures how much the new policy improves compared to the old policy while staying within the trust region. This optimization step seeks to maximize this surrogate objective.\n",
    "\n",
    "4. Clipping: The surrogate objective is clipped to ensure that the policy update remains within a certain threshold. This step enforces a limit on how much the policy can change between iterations.\n",
    "\n",
    "5. Repeat: Steps 1-4 are repeated iteratively, with the policy continually improving over time. PPO iteratively collects new data, evaluates the policy, and updates it while ensuring stability through trust region constraints."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3> Results </h3>\n",
    "\n",
    "The agent manages to learn to follow the line, however it moves in a very jerky manner. This is because the reward function is not designed to encourage smooth actions. The agent is only rewarded for staying on the line and penalized for going off the line. This results in the agent making very jerky movements to stay on the line."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<h3> Improvements </h3>\n",
    "\n",
    "* I added a penalty for the change in velocity of the motors as well as a time penalty. However this turned out to be counter productive as the agent would refused to make the turn.\n",
    "\n",
    "* The terminal state was initially the point where the agent goes off the line. This turned out to be too sudden and did not allow the agent to explore the environment enough to get enough information to learn efficiently. I changed the terminal state to a maximum episode duration of 500 steps."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  },
  "orig_nbformat": 4
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
