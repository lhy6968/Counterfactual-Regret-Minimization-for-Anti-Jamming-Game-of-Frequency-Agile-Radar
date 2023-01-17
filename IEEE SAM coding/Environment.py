import numpy as np

#Observer for imperfect information obvservations of public-info games.
from open_spiel.python.observation import IIGObserverForPublicInfoGame
from chi2comb import chi2comb_cdf, ChiSquared
import enum
import pyspiel
import torch

radar_action_list = [i for i in range(0,27)]# There are 27 actions for Radar

jammer_action_list = [27,28,29,30,31]# Five actions for Jammer

# parameters
sigma = [22, 22, 22]
pf = 1e-4  # false alarm rate
gcoef = 2
freedom_value = 2
B = 2e6  # bandwidth for each subpulse. The radar does not transmit signals with only single frequency
T0 = 290
B0 = 500e6  # noise bandwidth
k = 1.38e-23
N = k * T0 * B  # noise power
P = 30e3  # the power of radar transmitter for each subpulse
G = 10 ** (30 / 10)  # the antenna gain
f = 3e9  # carrier frequency. Ignore the influence caused by the change of the carrier frequency
lam = 1.5e8 / f  # the wavelength
R = 5e4  # range between the radar and the jammer
P_j = 1000  # the power of the jammer transmitter
G_j = 10 ** (5 / 10)  # the antenna gain


"""This program provide an approachable environment for Radar anti-jamming problem.
It is based on Google open source platform open-spiel. 
Please implement this program in the open-spiel together.
"""



_NUM_PLAYERS = 2# Two players, namely, Radar and Jammer
_NUM_ROWS = 2 # Two rows one for Jammer, one for Radar
_NUM_COLS = 4   # Only consider four rounds
_GAME_TYPE = pyspiel.GameType(
    short_name="My_Radar_Game",
    long_name="Python_Radar_Game",
    dynamics=pyspiel.GameType.Dynamics.SEQUENTIAL, # We implement simutaneous moves by sequential movements with special token.
    chance_mode=pyspiel.GameType.ChanceMode.DETERMINISTIC,
    information=pyspiel.GameType.Information.IMPERFECT_INFORMATION,
    utility=pyspiel.GameType.Utility.ZERO_SUM, # formulate as a zero sum imperfect information game
    reward_model=pyspiel.GameType.RewardModel.TERMINAL,
    max_num_players=_NUM_PLAYERS,
    min_num_players=_NUM_PLAYERS,
    provides_information_state_string=True,
    provides_information_state_tensor=True,
    provides_observation_string=True,
    provides_observation_tensor=True,
    parameter_specification={})

_GAME_INFO = pyspiel.GameInfo(
    num_distinct_actions=27+5, #total number of all the actions
    max_chance_outcomes=0,
    num_players=2,
    min_utility=-1.0,
    max_utility=1.0,
    utility_sum=0.0,
    max_game_length=_NUM_COLS * 2)

class RadarGame(pyspiel.Game):
  """A Python version of the Radar game."""

  def __init__(self, params=None):
    super().__init__(_GAME_TYPE, _GAME_INFO, params or dict())

  def new_initial_state(self):
    """Returns a state corresponding to the start of a game."""
    return RadarState(self)


  #We will modify this function to fit the imprefect game
  def make_py_observer(self, iig_obs_type=None, params=None):
    """Returns an object used for observing game state."""
    if ((iig_obs_type is None) or
        (iig_obs_type.public_info and not iig_obs_type.perfect_recall)):
      return BoardObserver(params)
    else:
      return IIGObserverForPublicInfoGame(iig_obs_type, params)


class RadarState(pyspiel.State):
  """A python version of the radar state."""

  def __init__(self, game):
    """Constructor; should only be called by Game.new_initial_state."""
    super().__init__(game)
    self._cur_player = 0
    self._player0_score = 0.0
    self._is_terminal = False
    self.board = np.full((_NUM_ROWS,_NUM_COLS), 66)
    self.current_column = 0
    self.radar_action = 0
    # radar parameters
    self.pf = pf
    # calculate sinr
    # 未被干扰，三个频点回波信噪比
    self.sinr = P * G ** 2 * lam ** 2 * np.array(sigma) / ((4 * np.pi) ** 3 * R ** 4 * N)
    # 被瞄准式干扰干扰的情况下，三个频点分别回波信干噪比
    self.pj_receive_spot = P_j * G * G_j * lam ** 2 / ((4 * np.pi) ** 2 * R ** 2)
    # 被压制干扰干扰的情况下，三个频点回波信干噪比
    self.pj_receive_barrage = P_j / B0 * B * G * G_j * lam ** 2 / ((4 * np.pi) ** 2 * R ** 2)
    # Different sigma
    self.sigma_unjammed = 1
    self.sigma_spot = self.pj_receive_spot / N
    self.sigma_barrage = self.pj_receive_barrage / N / 5000

  # OpenSpiel (PySpiel) API functions are below. This is the standard set that
  # should be implemented by every imperfect-information sequential-move game.


  def current_player(self):
    """Returns id of the next player to move, or TERMINAL if game is over."""
    if self._is_terminal:
      return pyspiel.PlayerId.TERMINAL
    else:
      return self._cur_player

  def transfer_action(self,action):
    action_list = []
    for i in range(0,3):
      for j in range(0,3):
        for k in range(0,3):
          action_list.append([i,j,k])
    return action_list[action]

  def _legal_actions(self, player):
    """Returns a list of legal actions, sorted in ascending order."""
    if player == 0:
      return radar_action_list
    else:
      return jammer_action_list

  def _apply_action(self, action):
    """Applies the specified action to the state."""
    if self._cur_player == 0:
      self.radar_action = action
      self.board[0,self.current_column] = 99 # after one player takes actions, we cover it with 99 token
      self._cur_player = 1 - self._cur_player
    else:
      self.board[0,self.current_column] = self.radar_action
      if action == 30:
        if self.radar_action >= 0 and self.radar_action <= 8:
          jammer_action = 32
        elif self.radar_action >= 9 and self.radar_action <= 17:
          jammer_action = 33
        elif self.radar_action >= 18 and self.radar_action <= 26:
          jammer_action = 34
      else:
        jammer_action = action
      self.board[1,self.current_column] = jammer_action
      self.current_column += 1
      if self.current_column == _NUM_COLS:
        self._is_terminal = True
        his_RadarFreq = []
        his_sinr = []
        his_sigma = []
        for i in range(0,_NUM_COLS):
          radar_state = self.board[0,i]
          jammer_state = self.board[1,i]
          radar_frequency_list = self.transfer_action(radar_state)
          if jammer_state == 27:
            jammer_frequency_list = [0,0,0]
          elif jammer_state == 28:
            jammer_frequency_list = [1,1,1]
          elif jammer_state == 29:
            jammer_frequency_list = [2,2,2]
          elif jammer_state == 31:
            jammer_frequency_list = [3,3,3]
          elif jammer_state == 32:
            jammer_frequency_list = [4,0,0]
          elif jammer_state == 33:
            jammer_frequency_list = [4,1,1]
          elif jammer_state == 34:
            jammer_frequency_list = [4,2,2]
          for j in range(0,3):
            his_RadarFreq.append(radar_frequency_list[j])
            his_sinr.append(self.sinr[radar_frequency_list[j]])
            if jammer_frequency_list[j] == radar_frequency_list[j]:
              result = self.sigma_spot
            elif jammer_frequency_list[j] != radar_frequency_list[j] and jammer_frequency_list[j] == 3:
              result = self.sigma_barrage
            else:
              result = self.sigma_unjammed
            his_sigma.append(result)
        his_RadarFreq = np.array(his_RadarFreq)
        his_sinr = np.array(his_sinr)
        his_sigma = np.array(his_sigma)
        pd = self.pd_cal(his_RadarFreq,his_sinr,his_sigma)
        self._player0_score = pd
        self._cur_player = 1 - self._cur_player
      else:
        self._cur_player = 1 - self._cur_player


  def pd_cal(self, his_RadarFreq, his_sinr, his_sigma):
      # Remove None part (None means successfully spot jamming)
      his_RadarFreq = his_RadarFreq[his_sigma != self.sigma_spot]
      his_sinr = his_sinr[his_sigma != self.sigma_spot]
      his_sigma = his_sigma[his_sigma != self.sigma_spot]

        # get cumulated sinr, sigma
      sinr_, sigma_ = [], []
      for i in range(3):
          sinr_part = his_sinr[his_RadarFreq == i]
          sigma_part = his_sigma[his_RadarFreq == i]
          if len(sinr_part):
              sinr_.append(np.sum(np.sqrt(sinr_part)) ** 2)
              sigma_.append(np.sum(sigma_part))
      sinr_ = np.array(sinr_)
      sigma_ = np.array(sigma_)
      T, _ = self.pfcalculation(sinr_, self.pf, sigma_)
      pd = self.pdcalculation(sinr_, T, sigma_)
      return pd

  # SWD
  def pfcalculation(self, snr, pf, Sigma):
      """
      :param snr: the estimation snr vector, which needs to be a numpy array type
      :param pf: given the probability of false alarm
      :return: the threshold
      """
      step, iter = 0.1, int(500)
      coefs = (snr / (snr + Sigma))
      dofs = freedom_value * np.ones(snr.shape[0])
      ncents = np.zeros(snr.shape[0])
      chi2s = [ChiSquared(coefs[i], ncents[i], dofs[i]) for i in range(snr.shape[0])]
      p = []
      interval = 1
      for k in range(0, iter, interval):
          result, _, _ = chi2comb_cdf(k * step, chi2s, gcoef)
          p.append(1 - result)
      posi = np.where(np.array(p) <= pf / 10)
      T = posi[0][0] * step * interval
      return T, p

  def pdcalculation(self, snr, T, Sigma):
      """
      :param snr: the estimation snr vector, which needs to be a numpy array type
      :param T: the threshold
      :return: the detection probability
      """
      coefs = snr / Sigma
      dofs = freedom_value * np.ones(snr.shape[0])
      ncents = np.zeros(snr.shape[0])
      chi2s = [ChiSquared(coefs[i], ncents[i], dofs[i]) for i in range(snr.shape[0])]
      result, _, _ = chi2comb_cdf(T, chi2s, gcoef)
      pd = 1 - result
      return pd

  def _action_to_string(self, player, action):
    """Action -> string."""
    #this function is meaningless and I can also implement it if you need
    pass
    
  def is_terminal(self):
    """Returns True if the game is over."""
    return self._is_terminal

  def returns(self):
    """Total reward for each player over the course of the game so far."""
    #we can define the reward here based on different ways of rewards
    return [self._player0_score, -self._player0_score]

  def __str__(self):
    """String for debug purposes. No particular semantics are required."""
    return _board_to_string(self.board)


class BoardObserver:
  """Observer, conforming to the PyObserver interface (see observation.py)."""

  def __init__(self, params):
    """Initializes an empty observation tensor."""
    if params:
      raise ValueError(f"Observation parameters not supported; passed {params}")
    # The observation should contain a 1-D tensor in `self.tensor` and a
    # dictionary of views onto the tensor, which may be of any shape.
    # Here the observation is indexed `(cell state, row, column)`.

    #I cannot understand why the shape is like this
    #here must be modified to fit the new game:1 + _NUM_PLAYERS
    shape = (1 + _NUM_PLAYERS,_NUM_ROWS, _NUM_COLS)
    self.tensor = np.zeros(np.prod(shape), np.float32)
    self.dict = {"observation": np.reshape(self.tensor, shape)}

  def set_from(self, state, player):
    """Updates `tensor` and `dict` to reflect `state` from PoV of `player`."""
    del player
    # We update the observation via the shaped tensor since indexing is more
    # convenient than with the 1-D tensor. Both are views onto the same memory.
    obs = self.dict["observation"]
    obs = state.board
    return obs

  def string_from(self, state, player):
    """Observation of `state` from the PoV of `player`, as a string."""
    del player
    return _board_to_string(state.board)
  


def _coord(move,column):
  """Returns (row, col) from an action id."""
  row = move - 1
  return (row,column)


def _board_to_string(board):
  """Returns a string representation of the board."""
  total_list = _board_to_tensor(board)
  total_string = ""
  for i in range(len(total_list)):
    total_string += str(total_list[i])
  return total_string

def _board_to_tensor(board):
  radar_part = board[0]
  jammer_part = board[1]
  total_list = []
  for i in range(_NUM_COLS):
    radar_action = radar_part[i]
    jammer_action = jammer_part[i]
    radar_list = _number_to_list(radar_action)
    jammer_list = _number_to_list(jammer_action)
    total_list = total_list + radar_list + jammer_list
  total_tensor = torch.tensor(total_list,dtype=torch.long)
  return total_tensor
    

def _number_to_list(number):
  number_list = []
  for i in range(36):
    number_list.append(0)
  if number >= 0 and number <= 29:
    number_list[number] = 1
  elif number == 31:
    number_list[30] = 1
  elif number >= 32 and number <= 34:
    number_list[number-1] = 1
  elif number == 66:
    number_list[34] = 1
  elif number == 99:
    number_list[35] = 1
  return number_list

# Register the game with the OpenSpiel library

pyspiel.register_game(_GAME_TYPE, RadarGame)
