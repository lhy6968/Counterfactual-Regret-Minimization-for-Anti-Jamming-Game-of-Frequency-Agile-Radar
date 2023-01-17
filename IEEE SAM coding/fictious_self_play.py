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
from nfsp_source import nfsp


FLAGS = flags.FLAGS

# flags.DEFINE_integer("num_iterations", 10, "Number of iterations")
# flags.DEFINE_integer("num_traversals", 20, "Number of traversals/games")
# flags.DEFINE_string("game_name", "My_Radar_Game", "Name of the game")


def main(unused_argv):
    logging.info("Loading %s", "My_Radar_Game")
    # game = pyspiel.load_game("kuhn_poker")
    reward_li = []
    avg_reward = []
    env = rl_environment.Environment("My_Radar_Game")
    state_size = 288
    num_actions = env.action_spec()["num_actions"]

    agents = [
        nfsp.NFSP(  # pylint: disable=g-complex-comprehension
            player_id,
            state_representation_size=state_size,
            num_actions=num_actions,
            hidden_layers_sizes=[1024,512,512,256, 256,128,128,32],
            reservoir_buffer_capacity=10000,
            anticipatory_param=0.1) for player_id in [0, 1]
    ]
    for i in range(40000):
        time_step = env.reset()
        while not time_step.last():
            current_player = time_step.observations["current_player"]
            current_agent = agents[current_player]
            agent_output = current_agent.step(time_step)
            time_step = env.step([agent_output.action])
        for agent in agents:
            agent.step(time_step)
        if i < 20000:
            print(time_step.rewards[0])
            continue
        reward_li.append(time_step.rewards[0])
        avg_reward.append(np.mean(reward_li))
        print(time_step.rewards[0])
    
    print("Mean",avg_reward[-1])
    plt.plot(avg_reward)
    plt.savefig("NFSP_action4.jpg")

if __name__ == "__main__":
  app.run(main)
