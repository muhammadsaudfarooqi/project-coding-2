# Importing classes
import numpy as np

from env import Environment
from agent_brain import SarsaTable

def calculate_metrics(steps, costs, total_episodes, exploration_count):
    # Calculate average episode length
    avg_episode_length = np.mean(steps)

    # Calculate average episode cost
    avg_episode_cost = np.mean(costs)

    # Calculate success rate (percentage of episodes that reached the goal)
    success_rate = (steps.count(10) / len(steps)) * 100

    # Calculate exploration rate (percentage of exploration actions)
    exploration_rate = (exploration_count / total_episodes) * 100

    # Print the calculated metrics
    print("Average Episode Length:", avg_episode_length)
    print("Average Episode Cost:", avg_episode_cost)
    print("Success Rate (%):", success_rate)
    print("Exploration Rate (%):", exploration_rate)
    print("Total Episodes:", total_episodes)


def train_agent():
    # List to store the number of steps per episode
    steps_per_episode = []

    # List to store the total cost per episode
    total_costs_per_episode = []

    # Train the agent for a fixed number of episodes
    for episode in range(1000):
        RL.update_episode_count()  # Increment episode count at the beginning of each episode
        # Reset the environment to its initial state
        observation = env.reset()

        # Initialize variables to track the number of steps and the total cost for this episode
        steps = 0
        total_cost = 0

        # Choose the first action based on the current observation
        action = RL.choose_action(str(observation))

        while True:
            # Render the environment to visualize the agent's actions
            env.render()

            # Take the chosen action and observe the next state and reward
            observation_, reward, done = env.step(action)

            # Choose the next action based on the next observation
            action_ = RL.choose_action(str(observation_))

            # Learn from the transition and calculate the cost
            total_cost += RL.learn(str(observation), action, reward, str(observation_), action_)

            # Update the current observation and action
            observation = observation_
            action = action_

            # Increment the step count for this episode
            steps += 1
            # Exit the loop when the episode is done (agent reached the goal or an obstacle)
            if done:
                steps_per_episode.append(steps)
                total_costs_per_episode.append(total_cost)
                RL.update_episode_count()  # Update the episode count
                break

    # Show the final route
    env.final()

    # Print the Q-table with values for each action
    RL.print_q_table()
    # Calculate and print metrics
    calculate_metrics(steps_per_episode, total_costs_per_episode,RL.total_episodes, RL.exploration_count)
    # Plot the results (number of steps and total cost per episode)
    RL.plot_results(steps_per_episode, total_costs_per_episode)

# Execute this part only when the script is run directly
if __name__ == "__main__":
    # Initialize the environment
    env = Environment()

    # Initialize the SARSA agent
    RL = SarsaTable(actions=list(range(env.n_actions)),
                    learning_rate=0.1,
                    reward_decay=0.9,
                    e_greedy=0.9)

    # Start training the agent by calling the train_agent function
    env.after(100, train_agent)  # Delayed start or simply call train_agent()
    env.mainloop()
