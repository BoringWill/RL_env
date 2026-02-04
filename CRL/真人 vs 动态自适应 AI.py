import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pygame
from collections import deque
from slime_env import SlimeSelfPlayEnv
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager

# --- 核心配置 ---
CONFIG = {
    "adaptive_model_path": "最强模型集/2.pth",
    "opponents_dir": "最强模型集",
    "device": torch.device("cpu"),
    "init_temp": 1.0,
    "min_temp": 0.1,
    "max_temp": 10.0,
    "ema_alpha": 0.05,  # 控制温度变化的速率
    "max_workers": 10,
}


# --- 1. 逻辑组件 ---
class SmartDifficultyManager:
    def __init__(self, init_mean=0.0, init_var=1.0):
        # 统计量
        self.mean = init_mean
        self.var = init_var
        self.count = 100  # 给一点初始样本数防止除以0或波动过大
        self.current_temp = CONFIG["init_temp"]

    def update(self, v_raw):
        # --- Z-score 统计更新 (Welford's 算法) ---
        self.count += 1
        old_mean = self.mean
        self.mean += (v_raw - old_mean) / self.count
        self.var = (self.var * (self.count - 1) + (v_raw - old_mean) * (v_raw - self.mean)) / self.count

        std = np.sqrt(self.var) if self.var > 1e-5 else 1.0

        # 计算 Z-score
        z_score = (v_raw - self.mean) / std

        # --- 极简二元控制逻辑 ---
        # 临时变量，用于计算目标方向

        step_size = 1.0 if self.current_temp >= 1.0 else 0.1

        # 2. 确定目标 (基于 0.25 阈值)
        step_target = self.current_temp
        if z_score > 0:
            step_target += step_size
        else:
            step_target -= step_size

        # EMA 更新公式
        # current_temp 会向 step_target 缓慢移动
        self.current_temp = self.current_temp * (1 - CONFIG["ema_alpha"]) + step_target * CONFIG["ema_alpha"]

        # --- 安全钳制 ---
        # 必须加这个，否则 temp 变成负数会导致 Softmax 报错
        self.current_temp = np.clip(self.current_temp, CONFIG["min_temp"], CONFIG["max_temp"])

        # 返回 temp 和 z_score (用于渲染显示)
        return self.current_temp, z_score


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.actor = nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))

    def get_value(self, obs):
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            return self.critic(t_obs).item()

    def get_action(self, obs, temp):
        with torch.no_grad():
            t_obs = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = self.actor(t_obs)
            probs = F.softmax(logits / temp, dim=-1)
            return torch.distributions.Categorical(probs).sample().item()


def load_weights(model, path):
    if not os.path.exists(path): return False
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=True)
    return True


# --- 2. 核心对战逻辑 ---
def play_one_match(opp_name, current_state_snapshot, render=False):
    render_mode = "human" if render else None
    env = SlimeSelfPlayEnv(render_mode=render_mode)

    agent_adaptive = Agent()
    load_weights(agent_adaptive, CONFIG["adaptive_model_path"])

    agent_fixed = Agent()
    opp_path = os.path.join(CONFIG["opponents_dir"], opp_name)
    load_weights(agent_fixed, opp_path)

    # 恢复状态
    diff_manager = SmartDifficultyManager(
        init_mean=current_state_snapshot.get('mean', 0.0),
        init_var=current_state_snapshot.get('var', 1.0)
    )

    font = None
    if render:
        try:
            font = pygame.font.SysFont("Arial", 20, bold=True)
        except:
            font = pygame.font.Font(None, 24)

    p1_dq, p2_dq = deque(maxlen=4), deque(maxlen=4)
    raw_obs_p1, _ = env.reset()
    raw_obs_p2 = env._get_obs(2)
    for _ in range(4): p1_dq.append(raw_obs_p1); p2_dq.append(raw_obs_p2)

    done = False

    while not done:
        if render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

        obs_p1 = np.concatenate(list(p1_dq))
        obs_p2 = np.concatenate(list(p2_dq))

        # 1. 获取 Value
        current_val = agent_adaptive.get_value(obs_p2)

        # 2. 纯粹的 Z-score 更新逻辑
        curr_temp, z_val = diff_manager.update(current_val)

        # 3. 采样
        a1 = agent_fixed.get_action(obs_p1, 0.01)
        a2 = agent_adaptive.get_action(obs_p2, curr_temp)

        n_p1, n_p2, _, term, trunc, _ = env.step((a1, a2))
        p1_dq.append(n_p1);
        p2_dq.append(n_p2)

        if render:
            env.render()
            try:
                screen = pygame.display.get_surface()
                if screen is not None and font is not None:
                    # 简单直白的显示
                    status_text = "UP" if z_val > 0.5 else "DOWN"
                    color = (0, 255, 0) if z_val > 0.5 else (255, 50, 50)

                    texts = [
                        f"Temp: {curr_temp:.2f}",
                        f"Z-Score: {z_val:.2f} [{status_text}]",
                        f"Opponent: {opp_name}"
                    ]
                    for i, text in enumerate(texts):
                        c = color if i == 1 else (255, 255, 255)
                        txt_surf = font.render(text, True, c)
                        screen.blit(txt_surf, (10, 10 + i * 25))
                    pygame.display.flip()
            except:
                pass
            time.sleep(0.015)

        if term or trunc:
            done = True

    result = {
        "opponent": opp_name,
        "p1_score": env.p1_score,
        "p2_score": env.p2_score,
        "win": env.p2_score > env.p1_score,
        "final_state": {
            "mean": diff_manager.mean,
            "var": diff_manager.var
        }
    }
    env.close()
    return result


# --- 3. 控制器 ---
def run_fast_tournament():
    opp_files = [f for f in os.listdir(CONFIG["opponents_dir"]) if f.endswith(".pth")]
    opp_files.sort()

    print(f"🚀 开始并发赛模式 | 并行数: {CONFIG['max_workers']} | 对手总数: {len(opp_files)}")
    print("=" * 60)

    all_results = []
    with Manager() as manager:
        # 只共享基础统计量
        shared_state = manager.dict({
            "mean": 0.0,
            "var": 5.0,  # 稍微给点初始方差，避免开局Z值爆炸
        })
        with ProcessPoolExecutor(max_workers=CONFIG["max_workers"]) as executor:
            futures = {executor.submit(play_one_match, name, dict(shared_state), False): name for name in opp_files}
            for future in as_completed(futures):
                res = future.result()
                if res:
                    shared_state.update(res["final_state"])
                    all_results.append(res)
                    status = "🏆 WIN" if res['win'] else "❌ LOSS"
                    print(f"[{status}] {res['opponent'].ljust(25)} | P1: {res['p1_score']} vs P2: {res['p2_score']}")

    print("\n" + "=" * 60)
    print("📊 最终战绩统计")
    print("-" * 60)
    wins = sum(1 for r in all_results if r['win'])
    total = len(all_results)
    if total > 0:
        print(f"总场次: {total} | 胜率: {wins / total:.2%} | 胜: {wins} / 负: {total - wins}")
    print("=" * 60)


if __name__ == "__main__":
    mode = input("1. 快速模式 (并发+无渲染)\n2. 观战模式 (单线程+有渲染)\n请选择: ")

    if mode == "1":
        run_fast_tournament()
    else:
        pygame.init()
        if not pygame.font.get_init():
            pygame.font.init()

        opp_files = [f for f in os.listdir(CONFIG["opponents_dir"]) if f.endswith(".pth")]
        opp_files.sort()

        current_state = {"mean": 0.0, "var": 5.0}

        print("\n📺 进入观战模式...")
        for f in opp_files:
            if f == os.path.basename(CONFIG["adaptive_model_path"]): continue
            print(f"🎮 正在对战: {f}")

            res = play_one_match(f, current_state, render=True)

            if res is None: break

            # 更新状态传给下一场
            current_state = res["final_state"]
            status = "🏆 WIN" if res['win'] else "❌ LOSS"
            print(f"[{status}] 战局结束 | P1: {res['p1_score']} vs P2: {res['p2_score']}")
            print("-" * 40)

        pygame.quit()