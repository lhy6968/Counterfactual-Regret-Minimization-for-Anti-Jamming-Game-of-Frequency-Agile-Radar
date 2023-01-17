"""Python Deep CFR example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

from open_spiel.python import policy
from open_spiel.python.algorithms import expected_game_score
import pyspiel
from open_spiel.python.pytorch import deep_cfr
from open_spiel.python.algorithms import exploitability
import radar_game
from rl_environement import rl_environment
from DQN_source import dqn
import matplotlib.pyplot as plt
import numpy as np

FLAGS = flags.FLAGS



def main(unused_argv):
    logging.info("Loading %s", "My_Radar_Game")
    # game = pyspiel.load_game("kuhn_poker")
    reward_li = []
    avg_reward = []
    env = rl_environment.Environment("My_Radar_Game")
    state_size = 288
    num_actions = env.action_spec()["num_actions"]

    agents = [
        dqn.DQN(  # pylint: disable=g-complex-comprehension
            player_id,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=[1024,512,512,256, 256,128,128,32],
            replay_buffer_capacity=10000,
            batch_size=5) for player_id in [0, 1]
    ]
    li_result = []
    for _ in range(1):
      for i in range(28326*14):
        time_step = env.reset()
        while not time_step.last():

            current_player = time_step.observations["current_player"]
            current_agent = agents[current_player]
            agent_output = current_agent.step(time_step)
            time_step = env.step([agent_output.action])
        for agent in agents:
          agent.step(time_step)
            

        if i/28326 in [k for k in range(1,15)]:
          print(i/28326)
          num_players = 2
          sum_episode_rewards = np.zeros(num_players)
          for player_pos in range(num_players):
            if player_pos == 1:
                break
            cur_agents = agents
            for _ in range(1000):
              time_step = env.reset()
              episode_rewards = 0
              while not time_step.last():
                player_id = time_step.observations["current_player"]
                agent_output = cur_agents[player_id].step(time_step, is_evaluation=True)
                action_list = [agent_output.action]
                time_step = env.step(action_list)
                episode_rewards += time_step.rewards[player_pos]
        #   print(episode_rewards)
              sum_episode_rewards[player_pos] += episode_rewards
          li_result.append(sum_episode_rewards / 1000)
          print("convergence point",sum_episode_rewards / 1000)
        else:
            continue
      
    print(li_result)
    plt.plot(li_result)
    plt.savefig("DQN_action4.jpg")

if __name__ == "__main__":
  app.run(main)
