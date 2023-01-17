"""Implements Deep CFR Algorithm.

See https://arxiv.org/abs/1811.00164.

The algorithm defines an `advantage` and `strategy` networks that compute
advantages used to do regret matching across information sets and to approximate
the strategy profiles of the game.  To train these networks a fixed ring buffer
(other data structures may be used) memory is used to accumulate samples to
train the networks.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import random
import numpy as np
import torch
import torch.nn as nn
import random


from open_spiel.python import policy
from Embedding import boardTensor # this module is used to embed the board to a numerical vector
import pyspiel


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

AdvantageMemory = collections.namedtuple(
    "AdvantageMemory", "info_state iteration advantage action")

StrategyMemory = collections.namedtuple(
    "StrategyMemory", "info_state iteration strategy_action_probs")


def reset_params(m):
  """Reinitializes the parameters of the given layer.

  Args:
    m: (nn.Module) layer object
  """
  if isinstance(m, nn.Linear):
    m.reset_parameters()


class MLP(nn.Module):
  """Basic MLP implementation using PyTorch."""

  def __init__(self, layers):
    """Initialize a MLP in PyTorch.

    Args:
      layers: (list[int]) Layer sizes.
    """
    super(MLP, self).__init__()
    layers = [
        nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers[:-1]))
    ]
    self._model = nn.Sequential(*layers)

  def forward(self, model_input):
    return self._model(model_input)

  def reset(self):
    """Reinitializes the network."""
    self._model.apply(reset_params)


class ReservoirBuffer(object):
  """Allows uniform sampling over a stream of data.

  This class supports the storage of arbitrary elements, such as observation
  tensors, integer actions, etc.
  See https://en.wikipedia.org/wiki/Reservoir_sampling for more details.
  """

  def __init__(self, reservoir_buffer_capacity):
    self._reservoir_buffer_capacity = reservoir_buffer_capacity
    self._data = []
    self._add_calls = 0

  def add(self, element):
    """Potentially adds `element` to the reservoir buffer.

    Args:
      element: data to be added to the reservoir buffer.
    """
    if len(self._data) < self._reservoir_buffer_capacity:
      self._data.append(element)
    else:
      idx = np.random.randint(0, self._add_calls + 1)
      if idx < self._reservoir_buffer_capacity:
        self._data[idx] = element
    self._add_calls += 1

  def sample(self, num_samples):
    """Returns `num_samples` uniformly sampled from the buffer.

    Args:
      num_samples: `int`, number of samples to draw.

    Returns:
      An iterable over `num_samples` random elements of the buffer.
    Raises:
      ValueError: If there are less than `num_samples` elements in the buffer
    """
    if len(self._data) < num_samples:
      raise ValueError("{} elements could not be sampled from size {}".format(
          num_samples, len(self._data)))
    return random.sample(self._data, num_samples)

  def clear(self):
    self._data = []
    self._add_calls = 0

  def __len__(self):
    return len(self._data)

  def __iter__(self):
    return iter(self._data)


class DeepCFRSolver(policy.Policy):
  """Implements a solver for the Deep CFR Algorithm with PyTorch.

  See https://arxiv.org/abs/1811.00164.

  Define all networks and sampling buffers/memories.  Derive losses & learning
  steps. Initialize the game state and algorithmic variables.

  Note: batch sizes default to `None` implying that training over the full
        dataset in memory is done by default.  To sample from the memories you
        may set these values to something less than the full capacity of the
        memory.
  """

  def __init__(self,
               game,
               policy_network_layers=(256, 256),
               advantage_network_layers=(128, 128),
               num_iterations: int = 100,
               num_traversals: int = 20,
               learning_rate: float = 1e-4,
               batch_size_advantage=None,
               batch_size_strategy=None,
               memory_capacity: int = int(100),
               policy_network_train_steps: int = 1000,
               advantage_network_train_steps: int = 200,
               reinitialize_advantage_networks: bool = True):
    """Initialize the Deep CFR algorithm.

    Args:
      game: Open Spiel game.
      policy_network_layers: (list[int]) Layer sizes of strategy net MLP.
      advantage_network_layers: (list[int]) Layer sizes of advantage net MLP.
      num_iterations: (int) Number of training iterations.
      num_traversals: (int) Number of traversals per iteration.
      learning_rate: (float) Learning rate.
      batch_size_advantage: (int or None) Batch size to sample from advantage
        memories.
      batch_size_strategy: (int or None) Batch size to sample from strategy
        memories.
      memory_capacity: Number af samples that can be stored in memory.
      policy_network_train_steps: Number of policy network training steps (per
        iteration).
      advantage_network_train_steps: Number of advantage network training steps
        (per iteration).
      reinitialize_advantage_networks: Whether to re-initialize the advantage
        network before training on each iteration.
    """
    all_players = list(range(game.num_players()))
    super(DeepCFRSolver, self).__init__(game, all_players)
    self._game = game
    if game.get_type().dynamics == pyspiel.GameType.Dynamics.SIMULTANEOUS:
      # `_traverse_game_tree` does not take into account this option.
      raise ValueError("Simulatenous games are not supported.")
    self._batch_size_advantage = batch_size_advantage
    self._batch_size_strategy = batch_size_strategy
    self._policy_network_train_steps = policy_network_train_steps
    self._advantage_network_train_steps = advantage_network_train_steps
    self._num_players = game.num_players()
    self._root_node = self._game.new_initial_state()
    self._embedding_size = len(boardTensor()._board_to_string(self._root_node.board))# this is where we use the embedding
    self._num_iterations = num_iterations
    self._num_traversals = num_traversals
    self._reinitialize_advantage_networks = reinitialize_advantage_networks
    self._num_actions = game.num_distinct_actions()
    self._iteration = 1

    # Define strategy network, loss & memory.
    self._strategy_memories = [
        ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
    ]
    self._policy_network =[
        MLP([self._embedding_size] + list(policy_network_layers) +
            [self._num_actions]) for _ in range(self._num_players)
    ]
    # Illegal actions are handled in the traversal code where expected payoff
    # and sampled regret is computed from the advantage networks.
    self._policy_sm = nn.Softmax(dim=-1)
    self._loss_policy = nn.MSELoss()
    self._optimizer_policy = []
    for i in range(self._num_players):
      self._optimizer_policy.append(
      torch.optim.Adam(
        self._policy_network[i].parameters(), lr=learning_rate))

    # Define advantage network, loss & memory. (One per player)
    self._advantage_memories = [
        ReservoirBuffer(memory_capacity) for _ in range(self._num_players)
    ]
    self._advantage_networks = [
        MLP([self._embedding_size] + list(advantage_network_layers) +
            [self._num_actions]) for _ in range(self._num_players)
    ]
    self._loss_advantages = nn.MSELoss(reduction="mean")
    self._optimizer_advantages = []
    for p in range(self._num_players):
      self._optimizer_advantages.append(
          torch.optim.Adam(
              self._advantage_networks[p].parameters(), lr=learning_rate))

  @property
  def advantage_buffers(self):
    return self._advantage_memories

  @property
  def strategy_buffer(self):
    return self._strategy_memories

  def clear_advantage_buffers(self):
    for p in range(self._num_players):
      self._advantage_memories[p].clear()

  def reinitialize_advantage_network(self, player):
    self._advantage_networks[player].reset()

  def reinitialize_advantage_networks(self):
    for p in range(self._num_players):
      self.reinitialize_advantage_network(p)

  def solve(self): # This is the key part
    """Solution logic for Deep CFR.

    Traverses the game tree, while storing the transitions for training
    advantage and policy networks.

    Returns:
      1. (nn.Module) Instance of the trained policy network for inference.
      2. (list of floats) Advantage network losses for
        each player during each iteration.
      3. (float) Policy loss.
    """
    advantage_losses = collections.defaultdict(list)
    for iteration in range(self._num_iterations):
      print(iteration)
      for p in range(self._num_players):
        for _ in range(self._num_traversals):
          self._traverse_game_tree(self._root_node, p)
        if self._reinitialize_advantage_networks:
          # Re-initialize advantage network for player and train from scratch.
          self.reinitialize_advantage_network(p)
        # Re-initialize advantage networks and train from scratch.
        advantage_losses[p].append(self._learn_advantage_network(p))
      self._iteration += 1
      # Train policy network.
      if iteration % 5 == 0: # to store the network for testing
        for pp in range(self._num_players):
          policy_loss = self._learn_strategy_network(pp)
          torch.save(self._policy_network[pp],"middle_test4/policy_network_deepcfr"+str(pp)+str(iteration)+".pkl")
    return self._policy_network, advantage_losses,policy_loss

  def _traverse_game_tree(self, state, player):
    """Performs a traversal of the game tree.

    Over a traversal the advantage and strategy memories are populated with
    computed advantage values and matched regrets respectively.

    Args:
      state: Current OpenSpiel game state.
      player: (int) Player index for this traversal.

    Returns:
      (float) Recursively returns expected payoffs for each action.
    """
    expected_payoff = collections.defaultdict(float)
    if state.is_terminal():
      # Terminal state get returns.
      return state.returns()[player]
    elif state.is_chance_node():
      # If this is a chance node, sample an action
      action = np.random.choice([i[0] for i in state.chance_outcomes()])
      return self._traverse_game_tree(state.child(action), player)
    elif state.current_player() == player:
      sampled_regret = collections.defaultdict(float)
      limitation = 2 # We don't want to search all the states such that we give a limitation on searching states.
      # Update the policy over the info set & actions via regret matching.
      for action in range(limitation):
        _, strategy = self._sample_action_from_advantage(state, player)
        probs = np.array(strategy)
        probs /= probs.sum()
        sampled_action = np.random.choice(range(self._num_actions), p=probs) # Based on probability to search Marks for Monte-Carlo methods
        expected_payoff[sampled_action] = self._traverse_game_tree(
            state.child(sampled_action), player)
      # to calculate the counterfactual value for regret
      cfv = 0 
      for a_ in state.legal_actions():
        cfv += strategy[a_] * expected_payoff[a_]
      for action in state.legal_actions():
        sampled_regret[action] = expected_payoff[action]
        sampled_regret[action] -= cfv
      sampled_regret_arr = [0] * self._num_actions
      for action in sampled_regret:
        sampled_regret_arr[action] = sampled_regret[action]
      self._advantage_memories[player].add(
          AdvantageMemory(boardTensor()._board_to_string(state.board), self._iteration,
                          sampled_regret_arr, action))
      return cfv
    else:
      other_player = state.current_player()
      _, strategy = self._sample_action_from_advantage(state, other_player)
      # Recompute distribution for numerical errors.
      probs = np.array(strategy)
      probs /= probs.sum()
      sampled_action = np.random.choice(range(self._num_actions), p=probs)
      self._strategy_memories[other_player].add(
          StrategyMemory(
              boardTensor()._board_to_string(state.board), self._iteration,
              probs))
      return self._traverse_game_tree(state.child(sampled_action), player)

  def _sample_action_from_advantage(self, state, player):
    """Returns an info state policy by applying regret-matching.

    Args:
      state: Current OpenSpiel game state.
      player: (int) Player index over which to compute regrets.

    Returns:
      1. (list) Advantage values for info state actions indexed by action.
      2. (list) Matched regrets, prob for actions indexed by action.
    """
    info_state = boardTensor()._board_to_string(state.board)
    legal_actions = state.legal_actions(player)
    with torch.no_grad():
      state_tensor = torch.FloatTensor(np.expand_dims(info_state, axis=0))
      raw_advantages = self._advantage_networks[player](state_tensor)[0].numpy()
    advantages = [max(0., advantage) for advantage in raw_advantages]
    cumulative_regret = np.sum([advantages[action] for action in legal_actions])
    matched_regrets = np.array([0.] * self._num_actions)
    if cumulative_regret > 0.:
      for action in legal_actions:
        matched_regrets[action] = advantages[action] / cumulative_regret
    else:
      matched_regrets[max(legal_actions, key=lambda a: raw_advantages[a])] = 1
    return advantages, matched_regrets

  def action_probabilities(self, state):
    """Computes action probabilities for the current player in state.

    Args:
      state: (pyspiel.State) The state to compute probabilities for.

    Returns:
      (dict) action probabilities for a single batch.
    """
    cur_player = state.current_player()
    legal_actions = state.legal_actions(cur_player)
    info_state_vector = np.array(boardTensor()._board_to_string(state.board))
    if len(info_state_vector.shape) == 1:
      info_state_vector = np.expand_dims(info_state_vector, axis=0)
    with torch.no_grad():
      logits = self._policy_network[cur_player](torch.FloatTensor(info_state_vector))
      probs = self._policy_sm(logits).numpy()
    return {action: probs[0][action] for action in legal_actions}

  def _learn_advantage_network(self, player):
    """Compute the loss on sampled transitions and perform a Q-network update.

    If there are not enough elements in the buffer, no loss is computed and
    `None` is returned instead.

    Args:
      player: (int) player index.

    Returns:
      (float) The average loss over the advantage network.
    """
    for _ in range(self._advantage_network_train_steps):

      if self._batch_size_advantage:
        if self._batch_size_advantage > len(self._advantage_memories[player]):
          ## Skip if there aren't enough samples
          return None
        samples = self._advantage_memories[player].sample(
            self._batch_size_advantage)
      else:
        samples = self._advantage_memories[player]
      info_states = []
      advantages = []
      iterations = []
      for s in samples:
        info_states.append(s.info_state.detach().numpy())
        advantages.append(s.advantage)
        iterations.append([s.iteration])
      # Ensure some samples have been gathered.
      if not info_states:
        return None
      self._optimizer_advantages[player].zero_grad()
      advantages = torch.FloatTensor(np.array(advantages))
      iters = torch.FloatTensor(np.sqrt(np.array(iterations)))
      outputs = self._advantage_networks[player](
          torch.FloatTensor(np.array(info_states)))
      loss_advantages = self._loss_advantages(iters * outputs,
                                              iters * advantages)
      loss_advantages.backward()
      self._optimizer_advantages[player].step()
      print(loss_advantages.detach().numpy())

    return loss_advantages.detach().numpy()

  def _learn_strategy_network(self,player):
    """Compute the loss over the strategy network.

    Returns:
      (float) The average loss obtained on this batch of transitions or `None`.
    """
    for _ in range(self._policy_network_train_steps):
      if self._batch_size_strategy:
        if self._batch_size_strategy > len(self._strategy_memories[player]):
          ## Skip if there aren't enough samples
          return None
        samples = self._strategy_memories[player].sample(self._batch_size_strategy)
      else:
        samples = self._strategy_memories[player]
      info_states = []
      action_probs = []
      iterations = []
      for s in samples:
        info_states.append(s.info_state.detach().numpy())
        action_probs.append(s.strategy_action_probs)
        iterations.append([s.iteration])
      self._optimizer_policy[player].zero_grad()
      iters = torch.FloatTensor(np.sqrt(np.array(iterations)))
      ac_probs = torch.FloatTensor(np.array(np.squeeze(action_probs)))
      logits = self._policy_network[player](torch.FloatTensor(np.array(info_states)))
      outputs = self._policy_sm(logits)
      loss_strategy = self._loss_policy(iters * outputs, iters * ac_probs)
      loss_strategy.backward()
      self._optimizer_policy[player].step()

    return loss_strategy.detach().numpy()
