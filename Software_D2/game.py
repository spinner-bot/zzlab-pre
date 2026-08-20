import copy

class TicTacToe:
    """
    井字棋游戏核心类
    职责：维护棋盘状态、判断输赢、执行落子、重置游戏。
    不包含任何AI逻辑。
    """

    def __init__(self):
        """初始化游戏"""
        self.reset()

    def reset(self):
        """
        重置游戏到初始状态
        - board: 3x3 的二维列表，0代表空，1代表玩家X，-1代表玩家O
        - current_player: 当前轮到谁 (1 或 -1)
        - winner: 获胜者 (1, -1, 0代表平局, None代表未结束)
        - is_over: 游戏是否结束
        """
        self.board = [[0, 0, 0],
                      [0, 0, 0],
                      [0, 0, 0]]
        self.current_player = 1  # 默认 X 先手
        self.winner = None
        self.is_over = False

    def get_valid_actions(self):
        """
        【API】获取当前所有合法的落子位置
        返回: 一个列表，包含所有空位的坐标元组 [(row, col), ...]
        例如: [(0, 0), (0, 2), (1, 1)]
        """
        actions = []
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == 0:
                    actions.append((r, c))
        return actions

    def step(self, action):
        """
        【API】执行一步落子
        参数 action: 元组 (row, col)，表示落子坐标
        返回: (reward, is_over, winner)
              - reward: 当前动作带来的即时奖励 (1:赢, -1:输, 0:其他)
              - is_over: 游戏是否结束
              - winner: 获胜者
        """
        r, c = action

        # 1. 检查动作是否合法
        if self.board[r][c] != 0:
            print(f"警告: 非法移动 ({r}, {c})，该位置已有棋子！")
            return 0, self.is_over, self.winner

        # 2. 更新棋盘
        self.board[r][c] = self.current_player

        # 3. 检查游戏结果
        win_result = self.check_winner()

        reward = 0
        if win_result is not None:
            self.is_over = True
            self.winner = win_result
            # 如果当前玩家赢了，奖励为1；如果平局(0)，奖励为0
            # 注意：这里返回的是对"当前行动者"的奖励
            if win_result == self.current_player:
                reward = 1
            elif win_result == 0: # 平局
                reward = 0
            else: # 实际上这一步走完对手不可能立刻赢，除非逻辑有误，这里主要处理平局和自胜
                 # 在井字棋中，如果check_winner返回对手，说明上一步对手赢了，这步不可能发生
                 # 但如果是因为这步填满了棋盘导致平局，winner是0
                 pass
        else:
            # 4. 如果没结束，切换玩家
            self.current_player = -self.current_player

        return reward, self.is_over, self.winner

    def check_winner(self):
        """
        检查是否有获胜者
        返回: 1 (X赢), -1 (O赢), 0 (平局), None (未结束)
        """
        # 检查行
        for r in range(3):
            if self.board[r][0] != 0 and \
               self.board[r][0] == self.board[r][1] == self.board[r][2]:
                return self.board[r][0]

        # 检查列
        for c in range(3):
            if self.board[0][c] != 0 and \
               self.board[0][c] == self.board[1][c] == self.board[2][c]:
                return self.board[0][c]

        # 检查主对角线
        if self.board[0][0] != 0 and \
           self.board[0][0] == self.board[1][1] == self.board[2][2]:
            return self.board[0][0]

        # 检查副对角线
        if self.board[0][2] != 0 and \
           self.board[0][2] == self.board[1][1] == self.board[2][0]:
            return self.board[0][2]

        # 检查平局 (没有空位了)
        if len(self.get_valid_actions()) == 0:
            return 0

        # 游戏继续
        return None

    def get_state_key(self):
        """
        【API】获取当前棋盘的字符串表示（用于存入Q表字典的Key）
        例如: "10-1001-1" 代表第一行是1,0,-1...
        这样可以将复杂的二维列表转化为简单的字符串哈希。
        """
        rows = []
        for r in range(3):
            rows.append("".join(str(x) for x in self.board[r]))
        return "-".join(rows)

    def render(self):
        """
        【调试用】在控制台打印当前棋盘
        """
        print("\n-------------")
        symbols = {1: " X ", -1: " O ", 0: "   "}
        for r in range(3):
            row_str = "|"
            for c in range(3):
                row_str += symbols[self.board[r][c]] + "|"
            print(row_str)
            print("-------------")
        print(f"当前轮到: {'X' if self.current_player == 1 else 'O'}")

    def play_console(self):
        """
        【调试用】双人控制台对战模式
        用于测试 game.py 逻辑是否正常
        """
        print("=== 井字棋控制台调试模式 ===")
        print("输入格式: 行 列 (例如: 0 0)")
        self.reset()

        while not self.is_over:
            self.render()
            valid_actions = self.get_valid_actions()
            print(f"合法位置: {valid_actions}")

            try:
                move_input = input("请输入落子坐标: ").split()
                r, c = int(move_input[0]), int(move_input[1])

                if (r, c) not in valid_actions:
                    print("无效输入，请重新输入！")
                    continue

                reward, is_over, winner = self.step((r, c))

                if is_over:
                    self.render()
                    if winner == 1: print("恭喜！X 获胜！")
                    elif winner == -1: print("恭喜！O 获胜！")
                    else: print("平局！")
                    break

            except (ValueError, IndexError):
                print("输入格式错误，请输入两个数字，如 '1 2'")


# --- 测试代码 ---
if __name__ == "__main__":
    # 实例化并启动控制台模式
    game = TicTacToe()
    game.play_console()
