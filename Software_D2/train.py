import json
import os
import random
from game import TicTacToe

class QLearningAgent:
    """
    井字棋强化学习智能体 (Q-Learning)
    职责：学习策略、保存/加载模型、提供落子建议。
    """

    def __init__(self, name="Agent", lr=0.1, gamma=0.9, epsilon=0.2):
        self.name = name
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.q_table = {}
        self.history = []
        self.file_path = f"{name}_q_table.json"

        # 核心新增：冻结状态标志
        self.frozen = False
        self.load_model()

    def freeze(self):
        """【API】冻结策略，进入测试/部署模式（纯利用，不更新Q表）"""
        self.frozen = True
        self.epsilon = 0.0
        print(f"[{self.name}] 策略已冻结，进入评估模式。")

    def unfreeze(self):
        """【API】解冻策略，恢复训练模式"""
        self.frozen = False
        self.epsilon = self.initial_epsilon
        print(f"[{self.name}] 策略已解冻，恢复训练模式。")

    def get_state_key(self, board):
        return ",".join(str(cell) for row in board for cell in row)

    def choose_action(self, game):
        state_key = self.get_state_key(game.board)
        valid_actions = game.get_valid_actions()

        # 如果冻结，强制 epsilon 为 0（纯利用）；如果状态没见过，只能随机
        current_eps = 0.0 if self.frozen else self.epsilon

        if state_key not in self.q_table or random.random() < current_eps:
            return random.choice(valid_actions)

        q_values = self.q_table[state_key]
        valid_q = [(action, q_values[action]) for action in valid_actions]
        max_q = max(val for _, val in valid_q)
        best_actions = [act for act, val in valid_q if val == max_q]
        return random.choice(best_actions)

    def update_history(self, state_key, action):
        self.history.append((state_key, action))

    def learn(self, reward):
        # 【核心逻辑】如果处于冻结状态，不更新Q表，只清空历史
        if self.frozen:
            self.history = []
            return

        for i, (state_key, action) in enumerate(reversed(self.history)):
            if state_key not in self.q_table:
                self.q_table[state_key] = [0.0] * 9

            current_q = self.q_table[state_key][action]
            step_reward = reward * (self.gamma ** i)
            new_q = current_q + self.lr * (step_reward - current_q)
            self.q_table[state_key][action] = new_q
        self.history = []

    def save_model(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.q_table, f)
        print(f"[{self.name}] 模型已保存至 {self.file_path} (共 {len(self.q_table)} 种状态)")

    def load_model(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r') as f:
                self.q_table = json.load(f)
            print(f"[{self.name}] 成功加载旧模型，包含 {len(self.q_table)} 种状态")
        else:
            print(f"[{self.name}] 未找到模型文件，将从零开始学习")


def run_ai_vs_ai_training(episodes=5000, window_size=100):
    """
    执行 AI vs AI 训练，并记录统计信息用于绘制学习曲线
    """
    agent_x = QLearningAgent(name="Agent_X")
    agent_o = QLearningAgent(name="Agent_O")
    agent_x.unfreeze()
    agent_o.unfreeze()

    stats_history = []
    recent_results = []  # 记录最近 window_size 局的 X 视角结果 (1:赢, -1:输, 0:平)

    print(f"开始训练，共 {episodes} 局...")
    for i in range(1, episodes + 1):
        game = TicTacToe()
        while not game.is_over:
            current_agent = agent_x if game.current_player == 1 else agent_o
            state_key = current_agent.get_state_key(game.board)
            action = current_agent.choose_action(game)
            current_agent.update_history(state_key, action)
            game.step(action)

        # 结算与更新
        r_x, r_o = (1, -1) if game.winner == 1 else ((-1, 1) if game.winner == -1 else (0.1, 0.1))
        agent_x.learn(r_x)
        agent_o.learn(r_o)

        # 记录统计
        recent_results.append(1 if game.winner == 1 else (-1 if game.winner == -1 else 0))
        if len(recent_results) > window_size:
            recent_results.pop(0)

        if i % window_size == 0:
            x_wins = recent_results.count(1) / len(recent_results)
            o_wins = recent_results.count(-1) / len(recent_results)
            draws = recent_results.count(0) / len(recent_results)
            stats_history.append({"epoch": i, "x_win": x_wins, "o_win": o_wins, "draw": draws})

            # 衰减探索率
            agent_x.epsilon = max(0.01, agent_x.epsilon * 0.99)
            agent_o.epsilon = max(0.01, agent_o.epsilon * 0.99)

    agent_x.save_model()
    agent_o.save_model()

    # 保存统计数据供 main.py 画图
    with open("training_stats.json", 'w') as f:
        json.dump(stats_history, f)
    print("训练完成！数据已保存至 training_stats.json")
