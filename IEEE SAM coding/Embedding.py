import torch


class boardTensor():
    def _board_to_string(self,board):
        """Returns a string representation of the board."""
        total_list = self._board_to_tensor(board)
        total_tensor = torch.tensor(total_list,dtype=torch.float32)
        return total_tensor

    def _board_to_tensor(self,board):
    #turn the board to a 2*_NUM_COLS*(27+7+2) tensor
        radar_part = board[0]
        jammer_part = board[1]
        total_list = []
        for i in range(2):
            radar_action = radar_part[i]
            jammer_action = jammer_part[i]
            radar_list = self._number_to_list(radar_action)
            jammer_list = self._number_to_list(jammer_action)
            total_list = total_list + radar_list + jammer_list
        return total_list
        

    def _number_to_list(self,number):
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