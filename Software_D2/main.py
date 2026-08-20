import pygame
import matplotlib.pyplot as plt
import json
import sys
import os
from game import TicTacToe
from train import QLearningAgent, run_ai_vs_ai_training, BASE_DIR

# 缺失的模型/数据文件，若不存在则自动训练一次
AUTO_TRAIN_EPISODES = 10000


def ensure_trained_model(agent=None):
    """模型或训练数据缺失时，自动训练一次，保证可直接开始对战或绘图"""
    model_file = os.path.join(BASE_DIR, "Agent_X_q_table.json")
    stats_file = os.path.join(BASE_DIR, "training_stats.json")
    trained = False
    if not os.path.exists(model_file) or not os.path.exists(stats_file):
        print(f"未检测到已训练模型({os.path.basename(model_file)})，正在自动训练（首次运行请稍候，约{AUTO_TRAIN_EPISODES}局）...")
        run_ai_vs_ai_training(episodes=AUTO_TRAIN_EPISODES)
        trained = True
        print("训练完成，已自动创建模型与训练数据。\n")
    if trained and agent is not None:
        agent.load_model()  # 重新加载最新模型，避免对战用到旧模型
    return trained


def toggle_freeze(agent):
    """冻结/解冻切换，并告知当前状态"""
    if agent.frozen:
        agent.unfreeze()
    else:
        agent.freeze()
    state = "已冻结（纯利用，不更新Q表）" if agent.frozen else "已解冻（可继续训练/探索）"
    print(f"[{agent.name}] 当前状态：{state}")


def run_ai_training_menu(agent):
    """AI 自动训练：指定局数或时长（二选一），支持 Ctrl+C 中断"""
    print("\n--- AI vs AI 自动训练 ---")
    print("训练结束条件（二选一）：")
    print("  1. 指定训练局数 (epoch)")
    print("  2. 指定训练时长 (秒)")
    mode = input("请选择 (1/2): ").strip()
    if mode == '1':
        ep = input("请输入训练局数: ").strip()
        try:
            episodes = int(ep)
            assert episodes > 0
        except (ValueError, AssertionError):
            print("无效的局数，已取消。")
            return
        run_ai_vs_ai_training(episodes=episodes)
    elif mode == '2':
        t = input("请输入训练时长(秒): ").strip()
        try:
            seconds = float(t)
            assert seconds > 0
        except (ValueError, AssertionError):
            print("无效的时长，已取消。")
            return
        run_ai_vs_ai_training(time_limit=seconds)
    else:
        print("无效选择，已取消。")
        return
    agent.load_model()  # 训练完成后重载最新模型


# --- 1. 学习曲线绘制 ---
def plot_learning_curve():
    stats_file = os.path.join(BASE_DIR, "training_stats.json")
    if not os.path.exists(stats_file):
        print("未找到训练数据，请先运行训练！")
        return

    with open(stats_file, 'r') as f:
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
def _load_cjk_font(size):
    """优先加载 Windows 中文字体(支持中文显示)，失败则回退到内置字体"""
    for path in (r"C:\Windows\Fonts\msyh.ttc",      # 微软雅黑
                 r"C:\Windows\Fonts\simhei.ttf",    # 黑体
                 r"C:\Windows\Fonts\simsun.ttc"):   # 宋体
        if os.path.exists(path):
            try:
                return pygame.font.Font(path, size)
            except Exception:
                continue
    return pygame.font.Font(None, size)


def play_human_vs_ai(ai_agent):
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("Tic-Tac-Toe: Human vs AI")
    # 用内置/系统中文字体，避免 SysFont 在部分环境(如 Python3.13)枚举系统字体时崩溃
    font = pygame.font.Font(None, 80)
    big_font = _load_cjk_font(72)   # 结果横幅
    small_font = _load_cjk_font(36) # 提示文字

    # 使用传入的持久 AI，遵循其在菜单中设置的冻结/解冻状态
    if ai_agent.frozen:
        print(f"[{ai_agent.name}] 当前为冻结状态（纯利用，测试模式）。")

    game = TicTacToe()
    running = True
    over_at = None  # 对局结束时刻(ms)，用于保证结果提示可见后再允许重开

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

        # 局终：屏幕中央醒目结果横幅
        if game.is_over:
            if game.winner == 0:
                result, color = "平局！", (80, 80, 80)
            elif game.winner == 1:
                result, color = "AI（X）获胜！", (220, 40, 40)
            else:
                result, color = "你（O）获胜！", (40, 120, 220)
            banner = big_font.render(result, True, color)
            screen.blit(banner, (300 - banner.get_width() // 2, 200))
            tip = small_font.render("点击任意处重新开始", True, (120, 120, 120))
            screen.blit(tip, (300 - tip.get_width() // 2, 310))

    while running:
        draw_board()
        pygame.display.flip()

        # 对局一结束就记录时刻（结果至少展示约0.3秒，避免误触直接进下一局）
        if game.is_over and over_at is None:
            over_at = pygame.time.get_ticks()

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

            # 结果展示满0.3秒后才允许点击重开
            if event.type == pygame.MOUSEBUTTONDOWN and game.is_over:
                if over_at is not None and (pygame.time.get_ticks() - over_at) >= 300:
                    game.reset()  # 点击重新开始
                    over_at = None
                    ai_move()  # 新对局 AI 再次先手

    pygame.quit()

# --- 3. 主菜单 ---
if __name__ == "__main__":
    # 跨菜单持久的 AI，保存冻结/解冻状态；默认冻结（测试/部署模式）
    agent_x = QLearningAgent(name="Agent_X")
    agent_x.freeze()

    while True:
        frozen_txt = "已冻结（纯利用）" if agent_x.frozen else "已解冻（可探索/学习）"
        print("\n=== 井字棋强化学习系统 ===")
        print("1. AI vs AI 自动训练（可指定局数/时长，Ctrl+C 中断）")
        print("2. Player vs AI (Pygame 可视化对战)")
        print("3. 仅绘制已有训练曲线")
        print(f"4. 冻结/解冻 AI 策略（当前：{frozen_txt}）")
        print("5. 退出")
        choice = input("请选择 (1/2/3/4/5): ")

        if choice == '1':
            run_ai_training_menu(agent_x)
            plot_learning_curve()
        elif choice == '2':
            ensure_trained_model(agent_x)  # 无模型则自动训练
            play_human_vs_ai(agent_x)
        elif choice == '3':
            ensure_trained_model(agent_x)  # 无数据则自动训练
            plot_learning_curve()
        elif choice == '4':
            toggle_freeze(agent_x)
        elif choice == '5':
            break
