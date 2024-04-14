import numpy as np

HEIGHT = 17
WIDTH = 17
OUT = 4
EMPTY = 0
LEFT = 0
RIGHT = 1
LEFT_UP = 2
RIGHT_UP = 3
LEFT_DOWN = 4
RIGHT_DOWN = 5
MOVES = [[0, -1], [0, 1], [-1, 0], [-1, 1], [1, -1], [1, 0]]

GOLD = 3
SILVER = 1
BRONZE = 1

PRIZES = [GOLD, SILVER, BRONZE]

ACTION_SIZE_OFFSET = [81 * 6, 25 * 25, 24 * 24, 24 * 24, 16 * 16]
ACTION_SUB_SPACE = [6, 25, 24, 24, 16]

                       # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
EMPTY_BOARD = np.array([
                        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4],  # 0
                        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4],  # 1
                        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4],  # 2
                        [4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4],  # 3
                        [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                        [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],  # 5
                        [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4],  # 6
                        [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4],  # 7
                        [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],  # 8
                        [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],  # 9 Center row
                        [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],  # 10
                        [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],  # 11
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],  # 12
                        [4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 13
                        [4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 14
                        [4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 15
                        [4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4],  # 16
                        ]).astype('int8')#12.

                # 0  1  2  3  4  5  6  7  8  9 10 11 12
START = np.array([
                    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4],  # 0
                    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 4, 4, 4, 4],  # 1
                    [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4],  # 2
                    [4, 4, 4, 4, 4, 4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4],  # 3
                    [4, 4, 4, 4, 2, 2, 2, 2, 0, 0, 0, 0, 0, 3, 3, 3, 3],  # 4
                    [4, 4, 4, 4, 2, 2, 2, 0, 0, 0, 0, 0, 0, 3, 3, 3, 4],  # 5
                    [4, 4, 4, 4, 2, 2, 0, 0, 0, 0, 0, 0, 0, 3, 3, 4, 4],  # 6
                    [4, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 4, 4, 4],  # 7
                    [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],  # 8
                    [4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],  # 9 Center row
                    [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],  # 10
                    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],  # 11
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],  # 12
                    [4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 13
                    [4, 4, 4, 4, 1, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 14
                    [4, 4, 4, 4, 1, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 15
                    [4, 4, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 16
                  ]).astype('int8')#12.

                #0  1  2  3  4  5  6  7  8  9 10 11 12
END = np.array([
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 4, 4, 4, 4],  # 0
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 4, 4, 4, 4],  # 1
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 4, 4, 4, 4],  # 2
                [4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 4, 4, 4, 4],  # 3
                [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 4
                [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],  # 5
                [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4],  # 6
                [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4],  # 7
                [4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 4],  # 8
                [4, 4, 4, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 4, 4, 4, 4],  # 9 Center row
                [4, 4, 3, 3, 0, 0, 0, 0, 0, 0, 0, 2, 2, 4, 4, 4, 4],  # 10
                [4, 3, 3, 3, 0, 0, 0, 0, 0, 0, 2, 2, 2, 4, 4, 4, 4],  # 11
                [3, 3, 3, 3, 0, 0, 0, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4],  # 12
                [4, 4, 4, 4, 0, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 13
                [4, 4, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 14
                [4, 4, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],  # 15
                [4, 4, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4],  # 16
                ]).astype('int8')#12

                # 0  1  2  3  4  5  6  7  8  9 10 11 12
GRID = np.array([
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 0
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0, 0, 0, 0],  # 1
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 0, 0, 0, 0],  # 2
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 2, 3, 0, 0, 0, 0],  # 3
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 1, 4, 1, 0, 0, 0, 0],  # 4
                    [0, 0, 0, 0, 0, 0, 0, 2, 3, 2, 3, 2, 3, 0, 0, 0, 0],  # 5
                    [0, 0, 0, 0, 0, 0, 1, 4, 1, 4, 1, 4, 1, 0, 0, 0, 0],  # 6
                    [0, 0, 0, 0, 0, 2, 3, 2, 3, 2, 3, 2, 3, 0, 0, 0, 0],  # 7
                    [0, 0, 0, 0, 1, 4, 1, 4, 1, 4, 1, 4, 1, 0, 0, 0, 0],  # 8
                    [0, 0, 0, 0, 3, 2, 3, 2, 3, 2, 3, 2, 0, 0, 0, 0, 0],  # 9 Center row
                    [0, 0, 0, 0, 1, 4, 1, 4, 1, 4, 1, 0, 0, 0, 0, 0, 0],  # 10
                    [0, 0, 0, 0, 3, 2, 3, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0],  # 11
                    [0, 0, 0, 0, 1, 4, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 12
                    [0, 0, 0, 0, 3, 2, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 13
                    [0, 0, 0, 0, 1, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 14
                    [0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 15
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 16
                 ]).astype('int8') #12


                  # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
ROTATION_LEFT = np.array(
                   [4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,  84,   4,   4,   4,   4,   
                    4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,  83, 100,   4,   4,   4,   4,
                    4,   4,   4,   4,   4,   4,   4,   4,   4,   4,  82,  99, 116,   4,   4,   4,   4,   
                    4,   4,   4,   4,   4,   4,   4,   4,   4,  81,  98, 115, 132,   4,   4,   4,   4,   
                    4,   4,   4,   4,  12,  29,  46,  63,  80,  97, 114, 131, 148, 165, 182, 199, 216,   
                    4,   4,   4,   4,  28,  45,  62,  79,  96, 113, 130, 147, 164, 181, 198, 215,   4,   
                    4,   4,   4,   4,  44,  61,  78,  95, 112, 129, 146, 163, 180, 197, 214,   4,   4,   
                    4,   4,   4,   4,  60,  77,  94, 111, 128, 145, 162, 179, 196, 213,   4,   4,   4,   
                    4,   4,   4,   4,  76,  93, 110, 127, 144, 161, 178, 195, 212,   4,   4,   4,   4,   
                    4,   4,   4,  75,  92, 109, 126, 143, 160, 177, 194, 211, 228,   4,   4,   4,   4,   
                    4,   4,  74,  91, 108, 125, 142, 159, 176, 193, 210, 227, 244,   4,   4,   4,   4,   
                    4,  73,  90, 107, 124, 141, 158, 175, 192, 209, 226, 243, 260,   4,   4,   4,   4, 
                   72,  89, 106, 123, 140, 157, 174, 191, 208, 225, 242, 259, 276,   4,   4,   4,   4,   
                    4,   4,   4,   4, 156, 173, 190, 207,   4,   4,   4,   4,   4,   4,   4,   4,   4,   
                    4,   4,   4,   4, 172, 189, 206,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   
                    4,   4,   4,   4, 188, 205,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   
                    4,   4,   4,   4, 204,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,])

ROTATION_RIGHT = np.array(
                   [4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,  72,   4,   4,   4,   4,   
                    4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,  89,  73,   4,   4,   4,   4,
                    4,   4,   4,   4,   4,   4,   4,   4,   4,   4, 106,  90,  74,   4,   4,   4,   4,   
                    4,   4,   4,   4,   4,   4,   4,   4,   4, 123, 107,  91,  75,   4,   4,   4,   4,   
                    4,   4,   4,   4, 204, 188, 172, 156, 140, 124, 108,  92,  76,  60,  44,  28,  12,   
                    4,   4,   4,   4, 205, 189, 173, 157, 141, 125, 109,  93,  77,  61,  45,  29,   4,   
                    4,   4,   4,   4, 206, 190, 174, 158, 142, 126, 110,  94,  78,  62,  46,   4,   4,   
                    4,   4,   4,   4, 207, 191, 175, 159, 143, 127, 111,  95,  79,  63,   4,   4,   4,   
                    4,   4,   4,   4, 208, 192, 176, 160, 144, 128, 112,  96,  80,   4,   4,   4,   4,   
                    4,   4,   4, 225, 209, 193, 177, 161, 145, 129, 113,  97,  81,   4,   4,   4,   4,   
                    4,   4, 242, 226, 210, 194, 178, 162, 146, 130, 114,  98,  82,   4,   4,   4,   4,   
                    4, 259, 243, 227, 211, 195, 179, 163, 147, 131, 115,  99,  83,   4,   4,   4,   4, 
                  276, 260, 244, 228, 212, 196, 180, 164, 148, 132, 116, 100,  84,   4,   4,   4,   4,   
                    4,   4,   4,   4, 213, 197, 181, 165,   4,   4,   4,   4,   4,   4,   4,   4,   4,   
                    4,   4,   4,   4, 214, 198, 182,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   
                    4,   4,   4,   4, 215, 199,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   
                    4,   4,   4,   4, 216,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,])

test = np.array([  4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,  12,   4,   4,   4,   4,   
                   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,  28,  29,   4,   4,   4,   4,
                   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,  44,  45,  46,   4,   4,   4,   4,   
                   4,   4,   4,   4,   4,   4,   4,   4,   4,  60,  61,  62,  63,   4,   4,   4,   4,   
                   4,   4,   4,   4,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,   
                   4,   4,   4,   4,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  99, 100,   4,   
                   4,   4,   4,   4, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,   4,   4,   
                   4,   4,   4,   4, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132,   4,   4,   4,   
                   4,   4,   4,   4, 140, 141, 142, 143, 144, 145, 146, 147, 148,   4,   4,   4,   4,   
                   4,   4,   4, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165,   4,   4,   4,   4,   
                   4,   4, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,   4,   4,   4,   4,   
                   4, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199,   4,   4,   4,   4, 
                 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216,   4,   4,   4,   4,   
                   4,   4,   4,   4, 225, 226, 227, 228,   4,   4,   4,   4,   4,   4,   4,   4,   4,   
                   4,   4,   4,   4, 242, 243, 244,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   
                   4,   4,   4,   4, 259, 260,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   4,   
                   4,   4,   4,   4, 276,   4,   4,   4,   4, 281,   4,   4,   4,   4,   4,   4,   4,])

class Board():

    def __init__(self):
        self.scores = np.array([0, 0, 0])
        self.scores_temporary = np.array([0, 0, 0])

    def get_start(self):
        return np.copy(START)

    # 1=left, 2=right, 3=lu, 4=ru, 5=ld, 6=rd
    def get_neighbor(self, y, x, dir, board):
        if y == 0 and dir in [LEFT_UP, RIGHT_UP]:
            return 0, 0, OUT
        if y == HEIGHT - 1 and dir in [LEFT_DOWN, RIGHT_DOWN]:
            return 0, 0, OUT
        if x == 0 and dir in [LEFT, LEFT_DOWN]:
            return 0, 0, OUT
        if x == WIDTH - 1 and dir in [RIGHT, RIGHT_UP]:
            return 0, 0, OUT

        yN, xN = y + MOVES[dir][0], x + MOVES[dir][1]

        if GRID[yN, xN] == 0:
            return 0, 0, OUT

        return yN, xN, board[yN, xN]

    def step(self, y, x, direction, board, step_size):
        if y < step_size and direction in [LEFT_UP, RIGHT_UP]:
            return 0, 0, OUT
        if y > HEIGHT - 1 - step_size and direction in [LEFT_DOWN, RIGHT_DOWN]:
            return 0, 0, OUT
        if x < step_size and direction in [LEFT, LEFT_DOWN]:
            return 0, 0, OUT
        if x > WIDTH - 1 - step_size and direction in [RIGHT, RIGHT_UP]:
            return 0, 0, OUT

        yN, xN = y + step_size * MOVES[direction][0], x + step_size * MOVES[direction][1]

        if GRID[yN, xN] == 0:
            return 0, 0, OUT

        return yN, xN, board[yN, xN]

    def get_jumps(self, y, x, board):
        jumps = []
        for direction in range(6):
            y_nn, x_nn, field_nn = self.step(y, x, direction, board, 2)
            if field_nn == EMPTY and self.right_zone(y, x, y_nn, x_nn):
                _, _, field_n = self.step(y, x, direction, board, 1)
                if field_n != 0:
                    jumps.append((y_nn, x_nn))
        return jumps

    def get_neighbors(self, y, x, board):
        neighbors = []
        for dir in [LEFT, RIGHT, LEFT_UP, RIGHT_UP, LEFT_DOWN, RIGHT_DOWN]:
            (yN, xN, valN) = self.get_neighbor(y, x, dir, board)
            if valN != OUT:
                neighbors.append((yN, xN, valN, dir))
        return neighbors

    def get_reachables_direct(self, y, x, board):
        reachables = []
        neighbors = self.get_neighbors(y, x, board)
        for (yN, xN, valN, dirN) in neighbors:
            if valN == EMPTY and self.right_zone(y, x, yN, xN):
                reachables.append((yN, xN, dirN))

        return reachables

    def get_reachables_jump(self, y, x, board):
        reachables = [(y,x)]
        start_index = 0
        while True:
            end_index = len(reachables)
            for i in range(start_index, end_index):
                (y, x) = reachables[i]
                jump_list = self.get_jumps(y, x, board)
                for j in range(len(jump_list)):
                    (y_nn, x_nn) = jump_list[j]
                    if (y_nn, x_nn) not in reachables:
                        reachables.append((y_nn, x_nn))

            if start_index == end_index:
                del reachables[0]
                return reachables
            start_index = end_index

    def right_zone(self, y_start, x_start, y_end, x_end):
        # if START[y_start, x_start] != 1 and START[y_end, x_end] == 1:
        #     return False
        if END[y_start, x_start] == 1 and END[y_end, x_end] != 1:
            return False
        return True

    def move(self, y_start, x_start, y_end, x_end, board, player):
        board[y_start, x_start] = EMPTY
        board[y_end, x_end] = player
        return board

    def get_done(self, board, player, color_matters):
        if color_matters:
            return False not in (board[END == player] == player)
        else:
            return False not in (board[END == player] != EMPTY)

    def get_win_state(self, board, temporary):
        if temporary:
            scores = self.scores_temporary
            for player in [1, 2, 3]:
                if not self.get_done(board, player, False):
                    scores[player - 1] = 0
        else:
            scores = self.scores

        first_found = False

        still_playing = []
        for player in [1, 2, 3]:
            if scores[player - 1] == 0:
                still_playing.append(player)

        prize = PRIZES[-len(still_playing)]
        if len(still_playing) < 3:
            color_matters = False
        else:
            color_matters = True
        for player in still_playing:
            if self.get_done(board, player, color_matters):
                scores[player - 1] = prize
                still_playing.remove(player)
                first_found = True

        if not temporary:
            self.scores_temporary = np.copy(scores)

        if first_found:
            return self.get_win_state(board, temporary)

        return scores

    def get_next_player(self, player):
        return player % 3 + 1

    def get_previous_player(self, player):
        previous = player - 1
        if previous == 0:
            previous = 3
        return previous

    def get_legal_moves(self, board):
        legal_moves_direct = []
        legal_moves_jumping = []

        player_y_list, player_x_list = np.where(board == 1)
        for i in range(len(player_y_list)):
            y_start, x_start = (player_y_list[i], player_x_list[i])
            reachables_direct = self.get_reachables_direct(y_start, x_start, board)
            reachables_jumping = self.get_reachables_jump(y_start, x_start, board)
            for (_, _, dir) in reachables_direct:
                legal_moves_direct.append((y_start, x_start, dir))

            for (y_end, x_end) in reachables_jumping:
                legal_moves_jumping.append((y_start, x_start, y_end, x_end))

        return legal_moves_direct, legal_moves_jumping

    def encode_move_direct(self, y_start, x_start, direction):
        start = self.encode_coordinates(y_start, x_start)

        return start * ACTION_SUB_SPACE[0] + direction

    def encode_move_jumping(self, y_start, x_start, y_end, x_end):
        grid_no, start = self.encode_coordinates_grid(y_start, x_start)
        grid_nu, end = self.encode_coordinates_grid(y_end, x_end)

        # encoded = sum(ACTION_SIZE_OFFSET[0:grid_no]) - 1 + start * ACTION_SUB_SPACE[grid_no] + end
        return sum(ACTION_SIZE_OFFSET[0:grid_no]) + start * ACTION_SUB_SPACE[grid_no] + end

    def decode_move(self, move):
        grid = 0

        while move > ACTION_SIZE_OFFSET[grid]:
            move -= ACTION_SIZE_OFFSET[grid]
            grid += 1

        start_position, direction = divmod(move, ACTION_SUB_SPACE[grid])

        if grid == 0:
            y_start, x_start = self.decode_coordinates(start_position)
            y_end, x_end = y_start + MOVES[direction][0], x_start + MOVES[direction][1]
        else:
            y_start, x_start = self.decode_coordinates_grid(start_position, grid)
            y_end, x_end = self.decode_coordinates_grid(direction, grid)

        return y_start, x_start, y_end, x_end

    def decode_coordinates(self, encoded):
        y_coordinates, x_coordinates = np.where(GRID != 0)
        return y_coordinates[encoded], x_coordinates[encoded]

    def decode_coordinates_grid(self, encoded, grid_no):
        y_coordinates, x_coordinates = np.where(GRID == grid_no)
        return y_coordinates[encoded], x_coordinates[encoded]

    # def encode_coordinates(self, y, x):
    #     y_coordinates, x_coordinates = np.where(GRID != 0)
    #     y_fits = np.where(y_coordinates == y)
    #     x_fits = np.where(x_coordinates == x)
    #     index_list = np.intersect1d(y_fits, x_fits)
    #     if index_list[0] != self.alternative_encode_coordinates(y, x):
    #         print("ERROR!")
    #     return index_list[0]

    def encode_coordinates(self, y, x):
        if y < 9:
            g_base = (y**2 + y) / 2
            g_plus = y + x - 10
        else:
            g_base = 81 - (17 - y) * (18 - y ) / 2
            g_plus = x - 2

        return int(g_base + g_plus - 1)

    def encode_coordinates_grid(self, y, x):
        grid_no = GRID[y,x]

        if grid_no == 1:
            if y < 7:
                g_base = y * (y + 2 ) / 8
                g_plus = (x + y + 1) / 2 - 4
            else:
                g_base = 16 - (14 - y) * (16 - y) / 8
                g_plus = (x + 1) / 2 - 1
        elif grid_no == 2:
            if y < 7:
                g_base = (y - 1) * (y + 1) / 8
                g_plus = (x + y + 1) / 2 - 4
            else:
                g_base = 12 - (13 - y) * (15 - y) / 8
                g_plus = x / 2 - 1
        elif grid_no == 3:
            if y < 7:
                g_base = (y - 1) * (y + 1) / 8
                g_plus = (x + y) / 2 - 4
            else:
                g_base = 12 - (13 - y) * (15 - y) / 8
                g_plus = (x - 1) / 2
        else:
            if y < 7:
                g_base = (y - 2) * y / 8
                g_plus = (x + y) / 2 - 4
            else:
                g_base = 9 - (12 - y) * (14 - y) / 8
                g_plus = x / 2 - 1

        return grid_no, int(g_base + g_plus - 1)

    def rotate_board(self, rotation_board, start_player, end_player):
        rotation_board = rotation_board.reshape(17 * 17)

        if self.get_next_player(start_player) == end_player:
            right = False
        else:
            right = True

        rotation_board = self.rotate(rotation_board, right)

        return rotation_board.reshape(17, 17)
        # rotated_board = np.copy(board)
        # for i in range(121):
        #     rotated_board[PLAYER1Board == i+1] = board[reference_board == i+1]
        #
        # return rotated_board

    def rotate(self, rotation_board, right):
        if right:
            rotation_index_board = ROTATION_RIGHT
        else:
            rotation_index_board = ROTATION_LEFT
        rotated_board = np.copy(EMPTY_BOARD.reshape(17 * 17))
        for i in range(len(rotation_board)):
            if rotation_board[i] not in [OUT, EMPTY]:
                rotated_board[rotation_index_board[i]] = rotation_board[i]

        return rotated_board

    def get_possible_board(self, y_start, x_start, board):
        possible_board = np.zeros((17,17))
        reachables_direct = self.get_reachables_direct(y_start, x_start, board)
        reachables_jumping = self.get_reachables_jump(y_start, x_start, board)
        for (_, _, direction) in reachables_direct:
            y_end, x_end = y_start + MOVES[direction][0], x_start + MOVES[direction][1]
            possible_board[y_end, x_end] = 1

        for (y_end, x_end) in reachables_jumping:
            possible_board[y_end, x_end] = 1

        return possible_board

    def get_action_by_coordinates(self, y_start, x_start, y_end, x_end):
        if GRID[y_start, x_start] == GRID[y_end, x_end]:
            return self.encode_move_jumping(y_start, x_start, y_end, x_end)
        else:
            diff_y, diff_x = y_end - y_start, x_end - x_start
            for direction in range(6):
                if MOVES[direction][0] == diff_y and MOVES[direction][1] == diff_x:
                    return self.encode_move_direct(y_start, x_start, direction)
