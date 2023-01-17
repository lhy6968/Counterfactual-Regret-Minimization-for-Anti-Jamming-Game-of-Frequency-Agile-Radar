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
from deep_cfr_algorithm import deep_cfr
from open_spiel.python.algorithms import exploitability
import Environment

FLAGS = flags.FLAGS


def main(unused_argv):
  logging.info("Loading %s", "My_Radar_Game") # this name is determined in Environment.
  game = pyspiel.load_game("My_Radar_Game")
  deep_cfr_solver = deep_cfr.DeepCFRSolver(
      game,
      policy_network_layers=(1024,512,512,256, 256,128,128),
      advantage_network_layers=(1024,512,256,256,128,128),
      num_iterations=50,
      num_traversals=5,
      learning_rate=1e-4,
      batch_size_advantage=256,
      batch_size_strategy=512,
      memory_capacity=int(10000))

  _, advantage_losses,policy_loss = deep_cfr_solver.solve()
  for player, losses in advantage_losses.items():
    logging.info("Advantage for player %d: %s", player,
                losses[:2] + ["..."] + losses[-2:])
    logging.info("Advantage Buffer Size for player %s: '%s'", player,
                len(deep_cfr_solver.advantage_buffers[player]))
  logging.info("Strategy Buffer Size: '%s'",
              len(deep_cfr_solver.strategy_buffer))
  logging.info("Final policy loss: '%s'", policy_loss)

  average_policy = policy.tabular_policy_from_callable(
      game, deep_cfr_solver.action_probabilities)
  pyspiel_policy = policy.python_policy_to_pyspiel_policy(average_policy)
  exploit = exploitability.exploitability(game,average_policy)
  print(exploit)


if __name__ == "__main__":
  app.run(main)
