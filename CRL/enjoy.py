import torch
import numpy as np
import pygame
import sys
from slime_env import SlimeSelfPlayEnv
from train_gpu import Agent  # 确保你的训练脚本名为 train_gpu.py


def enjoy_game(human_mode=False):
    # 1. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 2. 环境初始化
    # 必须设置 render_mode="human" 才能看到窗口
    env = SlimeSelfPlayEnv(render_mode="human")

    # 【重要修复】: 先 reset 初始化内部对象(p1, p2, ball)，再 render 初始化窗口
    obs_p1, _ = env.reset()
    obs_p2 = env._get_obs(2)
    env.render()

    # 3. 网络初始化与模型加载
    agent = Agent().to(device)
    try:
        # 加载权重文件
        state_dict = torch.load("slime_ppo_gpu.pth", map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()  # 开启评估模式
        print(">>> 模型加载成功，开始对战！")
    except FileNotFoundError:
        print("错误：找不到 'slime_ppo_gpu.pth' 文件，请确认训练已完成。")
        return

    clock = pygame.time.Clock()
    run = True

    while run:
        # A. 捕获 Pygame 事件（处理关闭窗口和按键）
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # B. 决策逻辑
        if human_mode:
            # --- 人类玩家控制 P1 (左侧红色) ---
            keys = pygame.key.get_pressed()
            action_p1 = 0
            if keys[pygame.K_a]:
                action_p1 = 1  # 左
            elif keys[pygame.K_d]:
                action_p1 = 2  # 右
            if keys[pygame.K_w]: action_p1 = 3  # 跳
        else:
            # --- AI 控制 P1 ---
            with torch.no_grad():
                t_obs_p1 = torch.FloatTensor(obs_p1).unsqueeze(0).to(device)
                logits_p1 = agent.actor(t_obs_p1)
                action_p1 = torch.argmax(logits_p1, dim=1).item()

        # --- AI 始终控制 P2 (右侧蓝色) ---
        with torch.no_grad():
            t_obs_p2 = torch.FloatTensor(obs_p2).unsqueeze(0).to(device)
            logits_p2 = agent.actor(t_obs_p2)
            action_p2 = torch.argmax(logits_p2, dim=1).item()

        # C. 环境步进
        # step 会根据 render_mode 自动调用绘制逻辑
        obs_p1, obs_p2, reward, done, _ = env.step(action_p1, action_p2)

        # D. 游戏重置
        if done:
            # print(f"本局结束，奖励结果: {reward}")
            obs_p1, _ = env.reset()
            obs_p2 = env._get_obs(2)

        # E. 控制播放速度 (60 FPS)
        clock.tick(60)

    env.close()
    pygame.quit()


if __name__ == "__main__":
    # 如果想自己打 AI，把这里改为 True
    enjoy_game(human_mode=False)