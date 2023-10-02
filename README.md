<h1> Line Following Robot V-1 </h1>

<h3> Objective</h3>

<p> The objective of this project is to build a line following robot. The robot should be able to follow a black line on a white background. It should be a pure Reinforcement Learning based approach. </p>

<h3> Dependencies </h3>

<p> The following dependencies are required to run the code. </p>

<ul>
    <li> python == 3.6 </li>
    <li> gym == 0.21.0 </li>
    <li> stable-baselines3 == 1.7.0 </li>
    <li> numpy == 1.26.0 </li>


<h3> Gym Environment </h3>

<p> The "LineFollowingEnv" is a custom Gym environment designed for the task of training an autonomous robot to follow a line in a simulated environment using reinforcement learning techniques. This environment serves as the foundation for developing and evaluating intelligent control policies for the robot.</p>

<h5>Key Features </h5>

* Action Space: 2 continous outputs between 0 and 1. It is used to control the speed of the robot. It will act as a pecentage of the maximum speed of the robot. A seperate one for the left and right motor.

* Observation Space: 3 continous outputs between 0 and 1024. These represent the readings from the IR sensors on the robot.

* Reward Function: The primary objective is to follow the line while minimizing penalties. The reward function is designed to encourage the robot to stay close to the line, move at an appropriate speed, and make smooth, controlled actions.

* Simulation: LineFollowingEnv is integrated with the Webots simulator, providing a realistic and dynamic environment for training. The simulation environment mimics real-world conditions, allowing agents to learn and adapt to various scenarios.

* Customization: The environment is highly customizable. Users can adjust parameters such as the sensor thresholds, maximum speed of the robot, and penalty magnitudes to tailor the learning task to their specific needs.

<h5>Training Objective</h5>

The primary training objective in LineFollowingEnv is to develop a reinforcement learning agent capable of effectively navigating the robot along a predefined line path. The agent learns to interpret sensor data, make decisions based on that data, and control the robot's motors to stay on course.


<h5> Reset() </h5>

The reset() function is called at the beginning of each episode. It is used to initialize the environment and prepare it for training. The function returns the initial observation of the environment.


```python
def reset(self):
    # Reset the simulation
    self.simulationResetPhysics()
    self.simulationReset()
    super().step(self.__timestep)
```

The robot has 2 motors and 3 IR sensors. The inital velocity of each motor is set to 0 and the sensors are enabled to provide output at every simulation time step.


```python
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
```

<h5> Step() </h5>

The step() function is called at each time step of the simulation. It is used to update the environment based on the agent's actions and return the new observation, reward, and done flag.

The action sampled is used to set the velocity of the motors. The velocity of the motors is set to a percentage of the maximum velocity of the robot. The percentage is calculated by multiplying the action sampled by the maximum velocity of the robot.




```python
self.__wheels[0].setVelocity(action[0] * self.max_speed)
self.__wheels[1].setVelocity(action[1] * self.max_speed)
```

The observations from the IR sensors are then recieved and added to the state.


```python
self.state = np.array([self.__ground_sensors[0].getValue(), self.__ground_sensors[1].getValue(), self.__ground_sensors[2].getValue()])
```

The basic reward function is as follows:

If the middle sensor of the robot is on the line , it gets a reward of 0 otherwise -2. The terminal state is reached when the robot is out of the line entirely.


```python
if self.state[2] > self.sensor_threshold:
    reward = -2
else:
    reward = 0

if self.state[0] > 900 and self.state[2] > 900:
    reward = 0
    done = True
```

<h3> Training </h3>

The training is done using the stable-baselines3 library. The PPO algorithm is used to train the agent. 

<h5> Proximal Policy Optimization </h5>

Proximal Policy Optimization (PPO) is a cutting-edge reinforcement learning algorithm known for its robustness, stability, and efficient use of computational resources. PPO belongs to the class of actor-critic methods, which combine value estimation (the critic) and policy improvement (the actor) to train agents in environments where actions influence both immediate rewards and future states.

<h5> Key Characteristics of PPO </h5>

* Policy Optimization: PPO directly optimizes the policy function, which is a neural network that maps observations to actions. This policy network is responsible for determining the agent's behavior in the environment.

* Stability and Trust Region: PPO is designed to ensure that policy updates are performed within a "trust region" to prevent overly large policy changes. This constraint helps maintain stability during training and prevents catastrophic policy collapses.

* Proximal Objective: PPO uses a proximal objective function, which quantifies the change in the policy's behavior concerning the previous policy. This objective encourages small, controlled policy updates.

* Clipped Surrogate Objective: To stay within the trust region, PPO employs a clipped surrogate objective. This objective is a combination of two terms: a surrogate policy improvement term and a clipped version of it. This mechanism enforces a limit on how much the policy can change, enhancing stability.

<h5> How it works </h5>

1. Data Collection: The agent interacts with the environment to collect a batch of experiences, consisting of observations, actions, rewards, and next-state observations.

2. Policy Evaluation: The collected data is used to estimate the advantages of the actions taken. This involves computing the expected return (cumulative rewards) for each state-action pair and subtracting a value estimate of the state's worth (the critic's role).

3. Policy Update: PPO calculates the surrogate objective, which measures how much the new policy improves compared to the old policy while staying within the trust region. This optimization step seeks to maximize this surrogate objective.

4. Clipping: The surrogate objective is clipped to ensure that the policy update remains within a certain threshold. This step enforces a limit on how much the policy can change between iterations.

5. Repeat: Steps 1-4 are repeated iteratively, with the policy continually improving over time. PPO iteratively collects new data, evaluates the policy, and updates it while ensuring stability through trust region constraints.

<h3> Results </h3>

The agent manages to learn to follow the line, however it moves in a very jerky manner. This is because the reward function is not designed to encourage smooth actions. The agent is only rewarded for staying on the line and penalized for going off the line. This results in the agent making very jerky movements to stay on the line.

<h3> Improvements </h3>

* I added a penalty for the change in velocity of the motors as well as a time penalty. However this turned out to be counter productive as the agent would refused to make the turn.

* The terminal state was initially the point where the agent goes off the line. This turned out to be too sudden and did not allow the agent to explore the environment enough to get enough information to learn efficiently. I changed the terminal state to a maximum episode duration of 500 steps.


