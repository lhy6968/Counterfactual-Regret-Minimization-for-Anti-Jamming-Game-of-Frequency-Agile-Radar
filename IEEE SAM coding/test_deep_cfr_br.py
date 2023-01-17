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
import deep_cfr_br_source 
from open_spiel.python.algorithms import exploitability
import Environment

FLAGS = flags.FLAGS


def main(unused_argv):
  logging.info("Loading %s", "My_Radar_Game")
  game = pyspiel.load_game("My_Radar_Game")
  for i in range(15):
    deep_cfr_solver = deep_cfr_br_source.DeepCFRSolver(
        game,
        policy_network_layers=(1024,512,512,256, 256,128,128),
        advantage_network_layers=(1024,512,256,256,128,128),
        network0="middle_test2/policy_network_deepcfr0"+str(2*i)+".pkl", # Read from the previous stored one.
        network1 = "middle_test2/policy_network_deepcfr1"+str(2*i)+".pkl",
        save_path= "exploitability_cfr2/policy_network_deepcfr1"+str(2*i)+".pkl",
        num_iterations=10,
        num_traversals=3,
        learning_rate=1e-4,
        batch_size_advantage=256,
        batch_size_strategy=512,
        memory_capacity=int(8e4))

    _, advantage_losses,policy_loss = deep_cfr_solver.solve()
    for player, losses in advantage_losses.items():
      logging.info("Advantage for player %d: %s", player,
                  losses[:2] + ["..."] + losses[-2:])
      logging.info("Advantage Buffer Size for player %s: '%s'", player,
                  len(deep_cfr_solver.advantage_buffers[player]))
    logging.info("Strategy Buffer Size: '%s'",
                len(deep_cfr_solver.strategy_buffer))
    logging.info("Final policy loss: '%s'", policy_loss)



if __name__ == "__main__":
  app.run(main)
