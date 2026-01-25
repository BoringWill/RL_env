import os
import torch
import torch.nn as nn
import numpy as np
import glob
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor
from torch.distributions import Categorical
from slime_env import SlimeSelfPlayEnv, FrameStack

# --- 配置 ---
CONFIG = {
    "model_dir": "模型集_历代版本最强",
    "games_per_match": 5,  # 每对选手打20局以获得更稳定的胜率
    "k_factor": 32,  # ELO K因子
    "max_workers": 8,  # 建议设为你的核心数
    "temperature": 0.8  # 采样温度，0.8 兼顾了实力和随机变招
}


class Agent(nn.Module):
    def __init__(self, input_dim=48):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def get_action_sample(self, obs, temperature=0.8):
        """采样决策：通过 Softmax 概率分布选择动作"""
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32)
            logits = self.actor(t_obs)
            # 应用温度系数
            probs = torch.softmax(logits / temperature, dim=-1)
            # 建立分类分布并采样
            m = Categorical(probs)
            return m.sample().item()


def load_weights(model, path):
    """修复后的加载函数：正确处理字典和纯权重"""
    try:
        ckpt = torch.load(path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
        model.load_state_dict(sd, strict=False)
        model.eval()
    except Exception as e:
        print(f"无法加载模型 {path}: {e}")


def play_match(path_a, path_b, games, temp):
    """单场对战逻辑，运行在子进程中"""
    # 每个进程独立创建环境
    raw_env = SlimeSelfPlayEnv(render_mode=None)
    env = FrameStack(raw_env, n_frames=4)

    agent_a = Agent()
    agent_b = Agent()
    load_weights(agent_a, path_a)
    load_weights(agent_b, path_b)

    a_wins = 0
    for _ in range(games):
        # 使用随机种子，避免每一局的球路完全一致
        obs_p1, _ = env.reset(seed=np.random.randint(0, 999999))
        p2_dq = [raw_env._get_obs(2) for _ in range(4)]

        done = False
        while not done:
            # P1 获取动作 (使用采样)
            act_a = agent_a.get_action_sample(obs_p1[None, :], temperature=temp)

            # P2 获取动作 (使用镜像观测和采样)
            obs_p2_stacked = np.concatenate(p2_dq)
            act_b = agent_b.get_action_sample(obs_p2_stacked[None, :], temperature=temp)

            # 步进
            obs_p1, _, term, trunc, info = env.step((act_a, act_b))

            # 维护 P2 的帧队列
            p2_dq.pop(0)
            p2_dq.append(info["p2_raw_obs"])

            done = term or trunc

        if info['p1_score'] > info['p2_score']:
            a_wins += 1

    return a_wins / games


def run_tournament():
    # 1. 扫描模型文件
    model_files = glob.glob(os.path.join(CONFIG["model_dir"], "**/*.pth"), recursive=True)
    if len(model_files) < 2:
        print("错误: 至少需要 2 个模型文件才能进行对战。")
        return

    ratings = {os.path.basename(f): 1200.0 for f in model_files}
    model_paths = {os.path.basename(f): f for f in model_files}
    names = sorted(list(ratings.keys()))
    pairs = list(combinations(names, 2))

    print(f"=== 史莱姆竞技场 (ELO 系统) ===")
    print(f"选手数量: {len(names)}")
    print(f"对战组数: {len(pairs)}")
    print(f"每组局数: {CONFIG['games_per_match']}")
    print(f"并发进程: {CONFIG['max_workers']}")
    print("-" * 30)

    # 2. 并行对战
    results = []
    with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
        futures = []
        for name_a, name_b in pairs:
            f = executor.submit(
                play_match,
                model_paths[name_a],
                model_paths[name_b],
                CONFIG["games_per_match"],
                CONFIG["temperature"]
            )
            futures.append((name_a, name_b, f))

        for name_a, name_b, f in futures:
            try:
                score_a = f.result()
                results.append((name_a, name_b, score_a))
                print(f"[对战完成] {name_a:25} vs {name_b:25} | P1胜率: {score_a * 100:5.1f}%")
            except Exception as e:
                print(f"对战出错 ({name_a} vs {name_b}): {e}")

    # 3. 计算 ELO 评分 (循环更新以收敛)
    for _ in range(100):
        for name_a, name_b, score_a in results:
            ra, rb = ratings[name_a], ratings[name_b]
            # 计算 A 的预期胜率
            ea = 1 / (1 + 10 ** ((rb - ra) / 400))
            # 更新分数 (平滑更新)
            update = (CONFIG["k_factor"] / 100) * (score_a - ea)
            ratings[name_a] += update
            ratings[name_b] -= update

    # 4. 打印最终排行榜
    print("\n" + "=" * 60)
    print(f"{'排名':<5} {'模型文件名':<40} {'ELO评分':<10}")
    print("-" * 60)

    sorted_ranks = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(sorted_ranks):
        # 为第一名添加颜色（如果控制台支持）
        color = "\033[92m" if i == 0 else ""
        reset = "\033[0m"
        print(f"{color}{i + 1:<5} {name:<40} {score:<10.1f}{reset}")
    print("=" * 60)

    # 保存结果到文件
    with open("elo_rankings.txt", "w") as f:
        f.write("Slime Volleyball Elo Rankings\n")
        f.write("-" * 30 + "\n")
        for i, (name, score) in enumerate(sorted_ranks):
            f.write(f"{i + 1}. {name}: {score:.1f}\n")
    print("\n排行榜已保存至: elo_rankings.txt")


if __name__ == "__main__":
    run_tournament()