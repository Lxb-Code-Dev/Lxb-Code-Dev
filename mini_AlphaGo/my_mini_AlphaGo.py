import random
from math import sqrt,log
from reversi_game import Game
import copy
import datetime
from test import *
#构建优先表，用来表示黑白棋上固定位置的优先级
priority_table = [['A1', 'H1', 'A8', 'H8'],
                  ['C1', 'F1', 'A3', 'A6', 'H3', 'H6', 'C8', 'F8'],
                  ['C3', 'F3', 'C6', 'F6'],
                  ['A4', 'A5', 'D1', 'E1', 'D8', 'E8', 'H4', 'H5'],
                  ['C4', 'C5', 'D3', 'E3', 'F4', 'F5', 'D6', 'E6'],
                  ['D4', 'E5', 'E4', 'D5'], # 0
                  ['D2', 'E2', 'B4', 'B5', 'D7', 'E7', 'G4', 'G5'],
                  ['C2', 'F2', 'B3', 'B6', 'C7', 'F7', 'G3', 'G6'],
                  ['B1', 'G1', 'B8', 'G8', 'A2', 'A7', 'H2', 'H7'],
                  ['B2', 'G7', 'G2', 'B7']]
roxanne_table = [['A1', 'H1', 'A8', 'H8'],
['C3', 'D3', 'E3', 'F3', 'C4', 'D4', 'E4', 'F4', 'C5', 'D5', 'E5', 'F5', 'C6', 'D6', 'E6', 'F6'],
['A3', 'A4', 'A5', 'A6', 'H3', 'H4', 'H5', 'H6', 'C1', 'D1', 'E1', 'F1', 'C8', 'D8', 'E8', 'F8'],
['B3', 'B4', 'B5', 'B6', 'G3', 'G4', 'G5', 'G6', 'C2', 'D2', 'E2', 'F2', 'C7', 'D7', 'E7', 'F7'],
['B1', 'A2', 'B2', 'G2', 'G1', 'H2', 'B7', 'A7', 'B8', 'G7', 'H7', 'G8']
]
class node():
    def __init__(self,state, player ,pre_mov=None):
        self.player=player
        # 模拟胜场
        self.win_number=0
        #模拟总场次
        self.total_number=0

        #子节点
        self.children=[]
        # 保存棋盘状态 当前没落子的状态
        self.state = state
        #准备下棋位置
        self.pre_mov=pre_mov

        self.remain_valid_moves=list(self.state.get_legal_actions(self.player.color))
        # 父节点
        self.parent = None
    #加入节点
    def insert(self,nex_move):
        child=node(self.state,self.player,nex_move)
        self.children.append(child)
        self.remain_valid_moves.remove(nex_move)
        child.parent=self
    #判断是否完全展开
    def isfullexpend(self):
        return len(self.remain_valid_moves)==0
    def terminal(self):
        return len(self.children)==0 and len(self.remain_valid_moves)==0
    #选择具有最大uct的子节点
    def best_child(self,c):
        child_value = [child.win_number / (child.total_number+1) + c * sqrt(log(self.total_number) / (child.total_number+1)) for child in self.children]
        value = max(child_value)
        idx = child_value.index(value)
        return self.children[idx]
    # 模拟走快棋

    def back_up(self,result):
        #result为模拟结果，胜利则为1，失败则为-1，平局为0
        while self is not None:
            self.total_number += 1
            self.win_number +=result
            self = self.parent

    #选择需要模拟的节点
    def best_choice(self):
        if self.isfullexpend()==0:
            if len(self.remain_valid_moves)!=0:
                if self.remain_valid_moves[0] is not None:
                    self.insert(self.remain_valid_moves[0])
                    return self.children[-1]
        if self.isfullexpend()==1 and self.terminal()==0:
            self.best_child(sqrt(2)).best_choice()

class AIPlayer:
    """
    AI 玩家
    """
    def __init__(self, color):
        """
        玩家初始化
        :param color: 下棋方，'X' - 黑棋，'O' - 白棋
        """
        self.color = color
        self.root=None
    def isend(self,board,color_1,color_2):
        return len(list(board.get_legal_actions(color_1)))==0 and len(list(board.get_legal_actions(color_1)))==0
    def max_choice(self,board,actions,color):
        actions_priorty_1=[]
        actions_priorty_2 = []
        actions_priorty=[]
        for i in actions:
            for j,k in enumerate(priority_table):
                if i in k:
                    actions_priorty_1.append(j+1)
        for i in actions:
            for j,k in enumerate(roxanne_table):
                if i in k:
                    actions_priorty_2.append(j+1)
        for i in range(len(actions_priorty_1)):
            actions_priorty.append(0.25*actions_priorty_1[i]+0.75*actions_priorty_2[i])
        best_=min(actions_priorty)
        acts_=[i for i,j in enumerate(actions_priorty) if j==best_]
        return actions[random.choice(acts_)]

    def bat(self,Node):
        bat_state=copy.deepcopy(Node.state)
        bat_state._move(Node.pre_mov,Node.player.color)
        bpl1_color = Node.player.color
        bpl2_color = "O" if Node.player.color == "X" else "X"
        if len(list(bat_state.get_legal_actions(bpl2_color))) != 0:
            bat_state._move(self.max_choice(bat_state,list(bat_state.get_legal_actions(bpl2_color)),bpl2_color), bpl2_color)
        Node.state = bat_state
        while self.isend(bat_state,bpl1_color,bpl2_color)==0:
            if len(list(bat_state.get_legal_actions(bpl1_color))) != 0:
                bat_state._move( self.max_choice(bat_state,list(bat_state.get_legal_actions(bpl1_color)),bpl1_color), bpl1_color)
            if len(list(bat_state.get_legal_actions(bpl2_color)))!=0:
                bat_state._move( self.max_choice(bat_state,list(bat_state.get_legal_actions(bpl2_color)),bpl2_color), bpl2_color)
        a,b=bat_state.get_winner()
        if a == 0:
            # 说明黑棋获胜
            if Node.player.color == "X":
                # 说明模拟方获胜
                return 1
            else:
                return -1
        elif a == 1:
            # 说明白棋获胜
            if Node.player.color == "O":
                # 说明模拟方获胜
                return 1
            else:
                return -1
        else:
            return 0
    #蒙特卡洛树搜索获得策略
    def get_move(self, board):
        """
        根据当前棋盘状态获取最佳落子位置
        :param board: 棋盘
        :return: action 最佳落子位置, e.g. 'A1'
        """
        if self.color == 'X':
            player_name = '黑棋'
        else:
            player_name = '白棋'
        print("请等一会，对方 {}-{} 正在思考中...".format(player_name, self.color))
        start_time=datetime.datetime.now()
        self.root=node(board,self)
        while(datetime.datetime.now()-start_time).seconds<3:
            new_node=self.root.best_choice()
            if new_node!=None:
                a =self.bat(new_node)
                new_node.back_up(a)
        best_node=self.root.best_child(sqrt(2))
        action=best_node.pre_mov
        return action
# 人类玩家黑棋初始化
black_player = AI_1Player("X")
# AI 玩家 白棋初始化
white_player = AIPlayer("O")
# 游戏初始化，第一个玩家是黑棋，第二个玩家是白棋
game = Game(black_player, white_player)
# 开始下棋
game.run()