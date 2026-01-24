import torch
import torch.nn as nn
import numpy as np
import pygame
import os
from collections import deque
from slime_env import SlimeSelfPlayEnv, FrameStack
import gymnasium as gym

# --- 超参数配置 ---
CONFIG = {
    "model_path_a": "模型集_历代版本最强/evolution_v1.pth",
    "model_path_b": "模型集_opponent/train_20260124-152719/evolution_v3.pth",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "num_envs": 8,  # 并行线程数
    "total_test_games": 20,  # 对抗测试的总局数（超参数）
    "side_swap": True  # 是否开启随机换场
}


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

    def get_actions(self, obs_batch, device, deterministic=True):
        with torch.no_grad():
            t_obs = torch.FloatTensor(obs_batch).to(device)
            logits = self.actor(t_obs)
            if deterministic:
                return torch.argmax(logits, dim=1).cpu().numpy()
            else:
                return torch.distributions.Categorical(logits=logits).sample().cpu().numpy()


def make_env():
    # 显式关闭渲染提高测试速度
    return lambda: FrameStack(SlimeSelfPlayEnv(render_mode=None), n_frames=4)


def enjoy_and_test():
    print("=" * 40)
    print("模式选择:")
    print("1: 真人 (P1) vs AI (P2)")
    print("2: AI (P1) vs 真人 (P2)")
    print(f"3: 多线程 AI 对抗测试 [目标: {CONFIG['total_test_games']} 局]")
    print("=" * 40)
    mode = input("输入序号: ")

    if mode in ["1", "2"]:
        # --- 单环境真人模式 ---
        raw_env = SlimeSelfPlayEnv(render_mode="human")
        env = FrameStack(raw_env, n_frames=4)
        clock = pygame.time.Clock()

        agent = Agent().to(CONFIG["device"])
        agent.load_state_dict(torch.load(CONFIG["model_path_a"], map_location=CONFIG["device"]))
        agent.eval()

        def reset_single():
            o1, _ = env.reset()
            p2_dq = deque(maxlen=4)
            # 单环境模式下直接调用内部方法获取初始观测
            init_p2_raw = raw_env._get_obs(2)
            for _ in range(4): p2_dq.append(init_p2_raw)
            return o1, np.concatenate(list(p2_dq), axis=0), p2_dq

        obs_p1, obs_p2, p2_frames = reset_single()
        run = True
        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: run = False

            keys = pygame.key.get_pressed()
            human_act = 0
            if keys[pygame.K_a]:
                human_act = 1
            elif keys[pygame.K_d]:
                human_act = 2
            if keys[pygame.K_w]: human_act = 3

            if mode == "1":
                a1, a2 = human_act, agent.get_actions(obs_p2[None, :], CONFIG["device"], False)[0]
            else:
                a1, a2 = agent.get_actions(obs_p1[None, :], CONFIG["device"], False)[0], human_act

            obs_p1, _, term, trunc, info = env.step((a1, a2))
            p2_frames.append(info["p2_raw_obs"])
            obs_p2 = np.concatenate(list(p2_frames), axis=0)

            raw_env.render()
            clock.tick(60)
            if term or trunc:
                obs_p1, obs_p2, p2_frames = reset_single()
        pygame.quit()

    elif mode == "3":
        # --- 多线程模型对抗测试模式 ---
        print(f">>> 正在启动 {CONFIG['num_envs']} 个子进程环境...")
        envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(CONFIG["num_envs"])])

        agent_a = Agent().to(CONFIG["device"])
        agent_b = Agent().to(CONFIG["device"])
        agent_a.load_state_dict(torch.load(CONFIG["model_path_a"], map_location=CONFIG["device"]))
        agent_b.load_state_dict(torch.load(CONFIG["model_path_b"], map_location=CONFIG["device"]))
        agent_a.eval();
        agent_b.eval()

        # 1. 安全重置并获取初始 P2 数据
        obs_p1, infos = envs.reset()
        p2_deques = [deque(maxlen=4) for _ in range(CONFIG["num_envs"])]

        # 修复 KeyError: 检查 infos 中是否存在 p2_raw_obs
        # 如果 reset 时没有返回，则使用全零初始化 (Slime 基础观测通常为 12 维)
        for i in range(CONFIG["num_envs"]):
            init_p2 = np.zeros(12)
            if isinstance(infos, dict) and "p2_raw_obs" in infos:
                init_p2 = infos["p2_raw_obs"][i]

            for _ in range(4):
                p2_deques[i].append(init_p2)

        obs_p2 = np.array([np.concatenate(list(d)) for d in p2_deques])

        # 初始换场掩码
        side_swapped = np.random.rand(CONFIG["num_envs"]) > 0.5 if CONFIG["side_swap"] else np.zeros(CONFIG["num_envs"],
                                                                                                     dtype=bool)

        a_wins, games_done = 0, 0

        print(
            f">>> 开始对战 | A: {os.path.basename(CONFIG['model_path_a'])} | B: {os.path.basename(CONFIG['model_path_b'])}")

        while games_done < CONFIG["total_test_games"]:
            # A/B 模型分配输入
            obs_for_a = np.where(side_swapped[:, None], obs_p2, obs_p1)
            obs_for_b = np.where(side_swapped[:, None], obs_p1, obs_p2)

            acts_a = agent_a.get_actions(obs_for_a, CONFIG["device"], True)
            acts_b = agent_b.get_actions(obs_for_b, CONFIG["device"], True)

            # 组装 P1, P2 动作
            env_acts = np.stack([
                np.where(side_swapped, acts_b, acts_a),
                np.where(side_swapped, acts_a, acts_b)
            ], axis=1)

            obs_p1, _, terms, truncs, infos = envs.step(env_acts)

            # 更新 P2 观测队列与统计
            for i in range(CONFIG["num_envs"]):
                if terms[i] or truncs[i]:
                    games_done += 1

                    # 判定胜负
                    p1_won = infos["p1_score"][i] > infos["p2_score"][i]
                    p2_won = infos["p2_score"][i] > infos["p1_score"][i]
                    if (not side_swapped[i] and p1_won) or (side_swapped[i] and p2_won):
                        a_wins += 1

                    # 换场逻辑
                    if CONFIG["side_swap"]:
                        side_swapped[i] = np.random.rand() > 0.5

                    # 局末重置 P2 队列
                    # 在 VectorEnv 中，term 时的 info 可能在 final_info 里
                    p2_reset_val = np.zeros(12)
                    if "p2_raw_obs" in infos:
                        p2_reset_val = infos["p2_raw_obs"][i]

                    p2_deques[i].clear()
                    for _ in range(4): p2_deques[i].append(p2_reset_val)

                    if games_done % 10 == 0:
                        print(
                            f"进度: {games_done}/{CONFIG['total_test_games']} | A 累计胜率: {(a_wins / games_done) * 100:.1f}%")

                    if games_done >= CONFIG["total_test_games"]: break
                else:
                    # 正常步进更新 P2 队列
                    if "p2_raw_obs" in infos:
                        p2_deques[i].append(infos["p2_raw_obs"][i])

            obs_p2 = np.array([np.concatenate(list(d)) for d in p2_deques])

        print("\n" + "=" * 40)
        print(f"测试总结报告:")
        print(f"总局数: {games_done}")
        print(f"模型 A 胜率: {(a_wins / games_done) * 100:.2f}% (获胜 {a_wins} 局)")
        print(f"模型 B 胜率: {((games_done - a_wins) / games_done) * 100:.2f}% (获胜 {games_done - a_wins} 局)")
        print("=" * 40)
        envs.close()


if __name__ == "__main__":
    enjoy_and_test()