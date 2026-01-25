import os
import torch
import torch.nn as nn
import numpy as np
import pygame
from collections import deque
from slime_env import SlimeSelfPlayEnv, FrameStack

# --- 配置 ---
CONFIG = {
    # 模型 A（主模型 / P1 默认模型）
    "model_path_a": "模型集_历代版本最强/slime_ppo_vs_fixed.pth",
    # 模型 B（PK 模式下的对手模型）
    "model_path_b": "模型集_历代版本最强/slime_ppo_vs_fixed.pth",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


# --- 网络结构 ---
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        # 即使测试不用 critic，也保留结构以兼容权重加载
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

    def get_actions(self, obs, device):
        with torch.no_grad():
            t_obs = torch.FloatTensor(obs).to(device)
            # 确保输入形状是 [batch_size, 48]
            if t_obs.dim() == 1:
                t_obs = t_obs.unsqueeze(0)
            logits = self.actor(t_obs)
            return torch.argmax(logits, dim=1).cpu().numpy()


# --- 兼容性权重加载函数 ---
def load_weights(model, path, device):
    if not os.path.exists(path):
        print(f"!!!! 错误: 找不到模型文件 {path}")
        return False
    try:
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint,
                                                                  dict) and "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print(f">>>> 已加载权重: {os.path.basename(path)}")
        return True
    except Exception as e:
        print(f"!!!! 加载失败 {path}: {e}")
        return False


def enjoy():
    # 1. 初始化 Pygame 视频系统，防止报错
    pygame.init()
    pygame.display.set_mode((1000, 500))

    print("\n" + "=" * 40)
    print("【模式选择】")
    print("1: 真人 (左侧 P1) vs AI (右侧 P2)")
    print("2: AI (左侧 P1) vs 真人 (右侧 P2)")
    print("3: 双模型 PK (AI-A vs AI-B)")
    print("=" * 40)
    mode = input("输入序号: ")

    # 2. 创建环境
    raw_env = SlimeSelfPlayEnv(render_mode="human")
    # 虽然用了 FrameStack，但我们手动维护队列以保证维度绝对正确
    env = FrameStack(raw_env, n_frames=4)
    clock = pygame.time.Clock()

    # 3. 初始化并加载 AI
    agent_a = Agent().to(CONFIG["device"])
    agent_b = Agent().to(CONFIG["device"])

    if mode == "1":
        load_weights(agent_b, CONFIG["model_path_a"], CONFIG["device"])
    elif mode == "2":
        load_weights(agent_a, CONFIG["model_path_a"], CONFIG["device"])
    elif mode == "3":
        print(">>> 加载 P1 (左侧)...")
        load_weights(agent_a, CONFIG["model_path_a"], CONFIG["device"])
        print(">>> 加载 P2 (右侧)...")
        load_weights(agent_b, CONFIG["model_path_b"], CONFIG["device"])

    agent_a.eval()
    agent_b.eval()

    # 4. 手动维护堆叠队列的辅助函数
    def reset_all():
        env.reset()  # 重置环境状态
        # 获取初始 12 维原始观测
        p1_init = raw_env._get_obs(1)
        p2_init = raw_env._get_obs(2)
        # 填充初始队列 (48维 = 4 * 12维)
        dq1 = deque([p1_init for _ in range(4)], maxlen=4)
        dq2 = deque([p2_init for _ in range(4)], maxlen=4)
        return dq1, dq2

    p1_dq, p2_dq = reset_all()
    running = True
    print("\n>>> 启动成功！控制键: W(跳跃), A(左移), D(右移)")

    while running:
        # 处理窗口事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- 获取真人动作 ---
        keys = pygame.key.get_pressed()
        h_act = 0
        if keys[pygame.K_a]:
            h_act = 1
        elif keys[pygame.K_d]:
            h_act = 2
        elif keys[pygame.K_w]:
            h_act = 3

        # --- AI 决策准备 (将队列拼接成 48 维) ---
        obs_p1_stacked = np.concatenate(list(p1_dq))
        obs_p2_stacked = np.concatenate(list(p2_dq))

        # --- 分配动作 ---
        if mode == "1":
            a1 = h_act
            a2 = agent_b.get_actions(obs_p2_stacked, CONFIG["device"])[0]
        elif mode == "2":
            a1 = agent_a.get_actions(obs_p1_stacked, CONFIG["device"])[0]
            a2 = h_act
        else:
            a1 = agent_a.get_actions(obs_p1_stacked, CONFIG["device"])[0]
            a2 = agent_b.get_actions(obs_p2_stacked, CONFIG["device"])[0]

        # --- 执行 Step 并兼容处理返回值数量 ---
        step_res = env.step((a1, a2))

        # 无论环境返回 5 个还是 6 个值，我们只取最后两个（term/trunc 和 info）
        # info 包含得分信息
        term, trunc, info = step_res[-3], step_res[-2], step_res[-1]

        # --- 核心修复：手动更新 12 维原始观测到队列中 ---
        p1_dq.append(raw_env._get_obs(1))
        p2_dq.append(raw_env._get_obs(2))

        # --- 渲染 ---
        raw_env.render()
        clock.tick(60)  # 保持 60 帧

        if term or trunc:
            print(f"本局结束 | P1 Score: {info.get('p1_score', 0)} - P2 Score: {info.get('p2_score', 0)}")
            p1_dq, p2_dq = reset_all()

    pygame.quit()


if __name__ == "__main__":
    enjoy()