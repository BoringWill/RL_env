import torch
import torch.nn as nn
import numpy as np
import pygame
import os
from collections import deque
from slime_env import SlimeSelfPlayEnv, FrameStack

# --- 配置与路径 ---
MODEL_P1_PATH = "slime_ppo_vs_fixed.pth"   # slime_ppo_vs_fixed.pth  ||  模型集/slime_ppo_gpu_v4.pth
MODEL_P2_PATH = "模型集/slime_ppo_gpu_v4.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 模型结构 ---
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

    def get_action(self, obs, device):
        with torch.no_grad():
            t_obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
            logits = self.actor(t_obs)
            return torch.argmax(logits, dim=1).item()


def enjoy():
    # 1. 初始化环境 (必须先定义好，稍后 reset 触发 pygame 初始化)
    raw_env = SlimeSelfPlayEnv(render_mode="human")
    env = FrameStack(raw_env, n_frames=4)

    # 2. 加载模型
    p1_agent = Agent().to(DEVICE)
    p2_agent = Agent().to(DEVICE)

    try:
        if os.path.exists(MODEL_P1_PATH):
            p1_agent.load_state_dict(torch.load(MODEL_P1_PATH, map_location=DEVICE, weights_only=False))
            print(f">>> P1 加载成功")
        if os.path.exists(MODEL_P2_PATH):
            p2_agent.load_state_dict(torch.load(MODEL_P2_PATH, map_location=DEVICE, weights_only=False))
            print(f">>> P2 加载成功")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    p1_agent.eval()
    p2_agent.eval()

    # --- 关键修复点：先 reset 确保 pygame 的视频系统被环境初始化 ---
    obs_p1, _ = env.reset()
    raw_env.render()  # 强制初始渲染

    run = True
    while run:
        # P2 帧堆叠手动初始化
        p2_frames = deque(maxlen=4)
        init_p2_raw = raw_env._get_obs(2)
        for _ in range(4): p2_frames.append(init_p2_raw)
        obs_p2 = np.concatenate(list(p2_frames), axis=0)

        game_over = False
        while not game_over:
            # A. 事件处理放在决策前，且必须在 pygame 初始化后调用
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    game_over = True

            # B. 决策
            action_p1 = p1_agent.get_action(obs_p1, DEVICE)
            action_p2 = p2_agent.get_action(obs_p2, DEVICE)

            # C. 执行动作 (这里的 term 只有达到 win_score 10分时才为 True)
            obs_p1, reward, term, trunc, info = env.step((action_p1, action_p2))

            # D. 更新 P2 观测数据
            p2_frames.append(info["p2_raw_obs"])
            obs_p2 = np.concatenate(list(p2_frames), axis=0)

            # E. 渲染
            raw_env.render()

            # F. 判断比赛是否真的结束 (10分制)
            if term or trunc:
                print(f"--- 比赛结束 | P1: {info['p1_score']} VS P2: {info['p2_score']} ---")
                pygame.time.wait(2000)
                game_over = True
                # 这里不直接 break，外层循环会再次 env.reset() 开始新的一局
                obs_p1, _ = env.reset()

    pygame.quit()


if __name__ == "__main__":
    enjoy()