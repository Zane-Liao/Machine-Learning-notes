# Introduction to Reinforcement Learning

##### Some Personal Ideas and an Introduction to Reinforcement Learning
- Reinforcement learning differs from other machine learning methods in that it relies on agents making decisions within an environment to learn.
- There are many concepts in reinforcement learning, so I'd like to try to organize them according to my own understanding. Of course, this is just an introduction.
- The core of reinforcement learning is the three major components: Value, Policy, Actor, and Critic. Almost all algorithms originate from these three components.
- Deep reinforcement learning simply adds deep neural networks as a foundation and reinforcement learning as a method for learning.
- There are also many derivatives of reinforcement learning, including control optimization and robotics.
- To understand reinforcement learning, code or theory alone are not enough. Finding a framework and building a practical model from scratch is more useful than running 10 exercises on Colab. Because reinforcement learning interacts with the environment, we generally need a simulated or real environment to practice. 2D grids, 3D simulations, or even a real robot are all acceptable, but we must always remember that reinforcement learning interacts with the environment.
- In reality, reinforcement learning is often combined with other machine learning methods. The entire LLM pipeline is a good example.
- The most important challenges with reinforcement learning currently are insufficient data, the requirement for 99% or higher accuracy in real-world environments, and interpretability and safety issues.
- Reinforcement learning relies on continuous interaction with the environment to learn, so, to some extent, relying on experience alone won't work for other machine learning methods.
- Reinforcement learning is used in game playing, control optimization, and robotics, and may be a key step toward achieving AGI. Currently, AI has developed to the point where we have models that appear reasonably good, but these models lack autonomous action, self-feedback, and iteration, which are crucial for achieving AGI. If our AI models can't generate or generate unprecedented knowledge and behaviors, then we may never achieve AGI. Current models also lack the interpretability to generate new behaviors, as we don't know whether some of their actions are correct or incorrect. However, imposing constraints on the models reduces their capabilities. Furthermore, if the reward function is not optimal, our models may exhibit inappropriate behavior, which is central to safety.
- Reinforcement learning suffers from high feedback costs. If we can use a component to accelerate reinforcement learning while maintaining its original effectiveness, this component could be a lifesaver for the current problems facing reinforcement learning.
- It's important to give the model a discriminator that's good enough to directly use as feedback for most new behaviors. However, setting limits for this discriminator is difficult, and we currently don't know.
- In the infinitely large real world, relying on just one learning algorithm is far from enough. In other words, we need a general-purpose learning algorithm.
- I believe the human brain has something called intuition, which may be more important than IQ or other assessments. To achieve machine intuition, we must first allow it to think without constraints. To achieve such unconstrained thinking, we must enable it to autonomously explore and act. This feedback from exploration is what we call the process of forming "thinking" (this "thinking" is both surprising and dangerous).
![[11111.png]]

[[#What is Reinforcement Learning]]
[[#Value-based]]
[[#Model-based]]
[[#Policy/Value-Iteration]]
[[#Policy-Gradient Methods]]
[[#Actor-Critic (from-Policy-Gradient)]]
[[#Inverse Reinforcement Learning]]
[[#Offline RL]]
[[#Online RL]]
[[#Imitation Learning]]
[[#Deep Reinforcement Learning]]
[[#Multi-Agent RL]]

##### What is Reinforcement Learning
- Reinforcement learning relies on the interaction between an agent and its environment. The basic framework of reinforcement learning originates from psychology.
[[Screenshot 2025-09-21 17.50.09.png]]
- For a detailed introduction to reinforcement learning concepts and Markov, please see my notes for Sections 2 and 4 of CS285.

##### Policy/Value-Iteration
- Value Iteration: Updates the state-value function $V(s)$ using the Bellman equation to obtain the optimal policy $\pi^*$.
$$V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]$$
- Policy Iteration: Alternates between Policy Evaluation and Policy Improvement until convergence.

##### Value-based
- Learns the state value $V(s)$ or action value $Q(s,a)$ and extracts the optimal policy from it.
- Q-Learning:
$$Q(s,a) \gets Q(s,a) + \alpha \big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big]$$
- SARSA: Within the policy, updates depend on the next action.

##### Model-based
- Learn the environment model $P(s'|s,a) and R(s,a)$, and use the model for planning (e.g., Value Iteration / Policy Iteration).
- **Disadvantages**: Modeling is difficult, and model bias can accumulate.

##### Policy-Gradient Methods
- Directly parameterize the policy $\pi_\theta(a|s)$, optimizing the expected cumulative reward:
$$J(\theta) = \mathbb{E}_{\pi_\theta}[ \sum_t \gamma^t r_t]$$
- For details on policy gradients, see [[Policy-Gradient]]

##### Actor-Critic (from-Policy-Gradient)
- **Actor**: Updates the policy $\pi_\theta$
- **Critic**: Estimates the value function $V(s)$ or $Q(s,a)$, reducing variance and improving convergence speed.
- For details, see Section 6 of cs285.

##### Inverse Reinforcement Learning
- Given the expert trajectory $\tau^E$, derive the hidden reward function $R(s,a)$
- Application: Automatically imitate expert behavior and discover potential goals

##### Offline RL
- Offline reinforcement learning: Trains the policy using only an existing dataset, without interacting with the environment.

Online RL
- As the name suggests, in contrast to offline RL, this method learns by continuously interacting with the environment to generate data.

##### Imitation Learning
- Imitation learning: Directly imitates the expert behavior $\pi(a|s) \approx \pi^E(a|s)$
- Common methods include Behavioral Cloning (BC) and DAgger. See Section 2 of cs285 for details.

##### Deep Reinforcement Learning
- Uses a Deep Neural Network to approximate the policy $\pi_\theta$ and the value function $Q_\theta(s,a)$
- Application: DQN, DDPG, TD3, SAC PPO / A3C

##### Multi-Agent RL
- Multi-agent reinforcement learning, where multiple agents learn simultaneously, and the environment dynamics include the behavior of other agents.
- Question: Instability and cooperative vs. adversarial