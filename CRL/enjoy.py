import torch
import torch.nn as nn
import numpy as np
import pygame
import sys
from collections import deque
from slime_env import SlimeSelfPlayEnv, FrameStack


# 确保与 train_gpu.py 一致 (256 维)
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(48, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(48, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )


def enjoy_game(human_mode=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 1. 初始化环境和包装器
    raw_env = SlimeSelfPlayEnv(render_mode="human")
    env = FrameStack(raw_env, n_frames=4)

    # 2. 加载模型
    agent = Agent().to(device)
    try:
        state_dict = torch.load("slime_ppo_gpu.pth", map_location=device)
        agent.load_state_dict(state_dict)
        agent.eval()
        print(">>> 模型加载成功，开始对战！")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return

    # 3. 初始重置与首帧强制渲染 (这一步确保窗口弹出)
    obs_p1, _ = env.reset()
    raw_env.render()

    p2_frames = deque(maxlen=4)
    init_p2_raw = raw_env._get_obs(2)
    for _ in range(4): p2_frames.append(init_p2_raw)
    obs_p2 = np.concatenate(list(p2_frames), axis=0)

    run = True
    while run:
        # A. 必须在循环最开始处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        # B. 决策逻辑
        if human_mode:
            keys = pygame.key.get_pressed()
            action_p1 = 0
            if keys[pygame.K_a]:
                action_p1 = 1
            elif keys[pygame.K_d]:
                action_p1 = 2
            if keys[pygame.K_w]: action_p1 = 3
        else:
            with torch.no_grad():
                input_p1 = torch.FloatTensor(obs_p1).unsqueeze(0).to(device)
                action_p1 = torch.argmax(agent.actor(input_p1), dim=1).item()

        # P2 决策 (AI)
        with torch.no_grad():
            input_p2 = torch.FloatTensor(obs_p2).unsqueeze(0).to(device)
            action_p2 = torch.argmax(agent.actor(input_p2), dim=1).item()

        # C. 执行动作
        obs_p1, reward, term, trunc, info = env.step((action_p1, action_p2))

        # D. 渲染画面
        raw_env.render()

        # E. 更新 P2 观测数据
        n_obs_p2_raw = info["p2_raw_obs"]
        p2_frames.append(n_obs_p2_raw)
        obs_p2 = np.concatenate(list(p2_frames), axis=0)

        # F. 游戏结束重置
        if term or trunc:
            obs_p1, _ = env.reset()
            raw_env.render()  # 重置后立即画一帧
            init_p2_raw = raw_env._get_obs(2)
            p2_frames.clear()
            for _ in range(4): p2_frames.append(init_p2_raw)
            obs_p2 = np.concatenate(list(p2_frames), axis=0)

    pygame.quit()


if __name__ == "__main__":
    # human_mode=True 时你可以用 A,D,W 控制左边的史莱姆
    enjoy_game(human_mode=False)