import pygame
import matplotlib.pyplot as plt
import json
import sys
import os
from game import TicTacToe
from train import QLearningAgent, run_ai_vs_ai_training

# --- 1. 学习曲线绘制 ---
def plot_learning_curve():
    if not os.path.exists("training_stats.json"):
        print("未找到训练数据，请先运行训练！")
        return

    with open("training_stats.json", 'r') as f:
        stats = json.load(f)

    epochs = [s['epoch'] for s in stats]
    x_wins = [s['x_win'] for s in stats]
    o_wins = [s['o_win'] for s in stats]
    draws = [s['draw'] for s in stats]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, x_wins, label='Agent X Win Rate', color='blue')
    plt.plot(epochs, o_wins, label='Agent O Win Rate', color='red')
    plt.plot(epochs, draws, label='Draw Rate', color='green')

    plt.title('Tic-Tac-Toe Q-Learning Training Curve')
    plt.xlabel('Training Epochs')
    plt.ylabel('Rate (Rolling Window)')
    plt.legend()
    plt.grid(True)
    plt.show()

# --- 2. Pygame 人机对战界面 ---
def play_human_vs_ai():
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Tic-Tac-Toe: Human vs AI")
    font = pygame.font.SysFont("arial", 80, bold=True)
    small_font = pygame.font.SysFont("arial", 40)

    # 加载并冻结 AI
    ai_agent = QLearningAgent(name="Agent_X")
    ai_agent.freeze()  # 测试前冻结策略

    game = TicTacToe()
    running = True

    def ai_move():
        # AI 执 X（先手），每次轮到 X 就自动落子（含开局第一步）
        if not game.is_over and game.current_player == 1:
            action = ai_agent.choose_action(game)
            game.step(action)

    ai_move()  # AI 先手第一步

    def draw_board():
        screen.fill((255, 255, 255))
        # 画线
        for i in range(1, 3):
            pygame.draw.line(screen, (0, 0, 0), (i * 200, 0), (i * 200, 600), 5)
            pygame.draw.line(screen, (0, 0, 0), (0, i * 200), (600, i * 200), 5)

        # 画棋子
        for r in range(3):
            for c in range(3):
                cx, cy = c * 200 + 100, r * 200 + 100
                if game.board[r][c] == 1:  # X (AI)
                    pygame.draw.line(screen, (255, 0, 0), (cx - 60, cy - 60), (cx + 60, cy + 60), 10)
                    pygame.draw.line(screen, (255, 0, 0), (cx + 60, cy - 60), (cx - 60, cy + 60), 10)
                elif game.board[r][c] == -1:  # O (Human)
                    pygame.draw.circle(screen, (0, 0, 255), (cx, cy), 70, 10)

        # 显示结果
        if game.is_over:
            txt = "Draw!" if game.winner == 0 else ("AI Wins!" if game.winner == 1 else "You Win!")
            color = (100, 100, 100) if game.winner == 0 else ((255, 0, 0) if game.winner == 1 else (0, 0, 255))
            surf = small_font.render(txt + " (Click to Restart)", True, color)
            screen.blit(surf, (300 - surf.get_width() // 2, 20))

    while running:
        draw_board()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and not game.is_over:
                if game.current_player == -1:  # 轮到人类 (O)
                    x, y = pygame.mouse.get_pos()
                    c, r = x // 200, y // 200
                    if (r, c) in game.get_valid_actions():
                        game.step((r, c))
                        ai_move()  # 人类落子后轮到 AI

            if event.type == pygame.MOUSEBUTTONDOWN and game.is_over:
                game.reset()  # 点击重新开始
                ai_move()  # 新对局 AI 再次先手

    pygame.quit()

# --- 3. 主菜单 ---
if __name__ == "__main__":
    while True:
        print("\n=== 井字棋强化学习系统 ===")
        print("1. AI vs AI 快速训练 (并绘制学习曲线)")
        print("2. Player vs AI (Pygame 可视化对战)")
        print("3. 仅绘制已有训练曲线")
        print("4. 退出")
        choice = input("请选择 (1/2/3/4): ")

        if choice == '1':
            run_ai_vs_ai_training(episodes=10000)
            plot_learning_curve()
        elif choice == '2':
            play_human_vs_ai()
        elif choice == '3':
            plot_learning_curve()
        elif choice == '4':
            break
