import numpy as np  # 提供维度数组与矩阵运算
import copy  # 从copy模块导入深度拷贝方法
from board import Chessboard


# 基于棋盘类，设计搜索策略
class Game:
    Queen_setRow = []
    solves = []
    def __init__(self, show=False):
        """
        初始化游戏状态.
        """

        self.chessBoard = Chessboard(show)
        self.solves = []
        self.gameInit()

    # 重置游戏
    def gameInit(self, show=True):
        """
        重置棋盘.
        """
        self.chessBoard.boardInit(False)

    def res(self, rows):
        for i in range(rows):
            a = self.Queen_setRow.pop(-1)
        self.gameInit(False)
        for i,j in enumerate(self.Queen_setRow):
            self.chessBoard.setQueen(i,j, False)

    def run(self, row=0):
        for col in range(8):
            if self.chessBoard.setLegal(row, col):
                self.Queen_setRow.append(col)
                self.chessBoard.setQueen(row, col, False)
                if row != 7:
                    self.run(row + 1)
                    self.res(len(self.Queen_setRow) - row)
                else:
                    a = copy.deepcopy(self.Queen_setRow)
                    print(a)
                    self.solves.append(a)
                    return

    def showResults(self, result):
        """
        结果展示.
        """

        self.chessBoard.boardInit(False)
        for i, item in enumerate(result):
            if item >= 0:
                self.chessBoard.setQueen(i, item, False)

        self.chessBoard.printChessboard(False)

    def get_results(self):
        """
        输出结果(请勿修改此函数).
        return: 八皇后的序列解的list.
        """
        self.run()
        return self.solves
game = Game()
solutions = game.get_results()
print(solutions)
print('There are {} results.'.format(len(solutions)))
game.showResults(solutions[0])
