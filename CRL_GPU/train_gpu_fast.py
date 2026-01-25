import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import glob
import time
import random
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from slime_env_gpu import SlimeVolleyballGPU

# --- 配置参数 ---
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "slime_ppo_vs_fixed.pth",
    "p1_path": "模型集_opponent/train_20260125-013011/evolution_v5.pth",
    "resume_dir": "模型集_opponent/train_20260125-013011",
    "external_history_folder": "模型集_历代版本最强",
    "start_step": 0,
    "auto_replace_threshold": 0.80,
    "min_games_to_replace": 30,  # 按照原版逻辑
    "total_timesteps": 500_000_000,
    "num_envs": 2048,
    "num_steps": 256,
    "update_epochs": 4,
    "batch_size": 16384,
    "lr": 2.5e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "historical_ratio": 0.2,
    "alpha_sampling": 0.1,
    "openai_eta": 0.1,
    "save_every_n_evolutions": 10,
}


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
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

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def train():
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    current_run_dir = config["resume_dir"] if os.path.exists(
        config["resume_dir"]) else f"模型集_opponent/train_gpu_{timestamp}"
    os.makedirs(current_run_dir, exist_ok=True)

    current_save_path = os.path.join(current_run_dir, config["save_path"])
    opponent_model_path = os.path.join(current_run_dir, "fixed_opponent_current.pth")

    env = SlimeVolleyballGPU(config["num_envs"], config["device"])

    # --- 两个实体的绝对隔离 ---
    agent = Agent().to(config["device"])
    opponent_agent = Agent().to(config["device"])
    opponent_agent.eval()  # 考官永远处于评价模式

    optimizer = optim.Adam(agent.parameters(), lr=config["lr"])
    writer = SummaryWriter(f"runs/gpu_{timestamp}")

    # --- 变量初始化 (对标原版) ---
    global_step = config["start_step"]
    total_games = 0
    agent_wins = 0
    evolution_trigger_count = 0
    evolution_count = 0

    # 加载学生模型状态
    if os.path.exists(current_save_path):
        ckpt = torch.load(current_save_path, map_location=config["device"])
        agent.load_state_dict(ckpt["model_state_dict"])
        total_games = ckpt.get("total_games", 0)
        agent_wins = ckpt.get("agent_wins", 0)
        evolution_trigger_count = ckpt.get("evolution_trigger_count", 0)
        print(f">>>> 成功恢复进度: 已进行 {total_games} 局")
    elif os.path.exists(config["p1_path"]):
        ckpt = torch.load(config["p1_path"], map_location=config["device"])
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        agent.load_state_dict(sd)

    # 考官初始化
    if not os.path.exists(opponent_model_path):
        torch.save({"model_state_dict": agent.state_dict()}, opponent_model_path)

    # 扫描对手池
    pool_paths = []
    if os.path.exists(config["external_history_folder"]):
        pool_paths.extend(glob.glob(os.path.join(config["external_history_folder"], "*.pth")))
    history_files = glob.glob(os.path.join(current_run_dir, "evolution_v*.pth"))
    pool_paths.extend(history_files)
    pool_paths = sorted(list(set([os.path.abspath(p) for p in pool_paths])))
    q_scores = [1.0] * len(pool_paths)
    evolution_count = len(history_files)

    # GPU 观测队列 (FrameStack)
    obs_queue_p1 = deque([torch.zeros((config["num_envs"], 12), device=config["device"]) for _ in range(4)], maxlen=4)
    obs_queue_p2 = deque([torch.zeros((config["num_envs"], 12), device=config["device"]) for _ in range(4)], maxlen=4)
    o1, o2 = env.reset()
    for _ in range(4): obs_queue_p1.append(o1); obs_queue_p2.append(o2)

    while global_step < config["total_timesteps"]:
        # --- 考官权重加载：确保训练开始前考官是“死”的 ---
        is_history_match = random.random() < config["historical_ratio"] and len(pool_paths) > 0
        current_opp_idx = -1

        # 严格从磁盘加载权重，不让考官有任何学习机会
        with torch.no_grad():
            if is_history_match:
                qs = np.array(q_scores)
                probs = (1 - config["alpha_sampling"]) * (np.exp(qs) / np.sum(np.exp(qs))) + config[
                    "alpha_sampling"] / len(pool_paths)
                current_opp_idx = np.random.choice(len(pool_paths), p=probs)
                ckpt = torch.load(pool_paths[current_opp_idx], map_location=config["device"])
            else:
                ckpt = torch.load(opponent_model_path, map_location=config["device"])

            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            opponent_agent.load_state_dict(sd)

        agent.eval()
        swap_sides = torch.rand(config["num_envs"], device=config["device"]) > 0.5

        b_obs = torch.zeros((config["num_steps"], config["num_envs"], 48), device=config["device"])
        b_act = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_logp = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_rew = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_done = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_val = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])

        # --- Rollout 数据采集 ---
        for step in range(config["num_steps"]):
            global_step += config["num_envs"]
            c_o1 = torch.cat(list(obs_queue_p1), dim=1)
            c_o2 = torch.cat(list(obs_queue_p2), dim=1)

            a_obs = torch.where(swap_sides.unsqueeze(1), c_o2, c_o1)
            o_obs = torch.where(swap_sides.unsqueeze(1), c_o1, c_o2)

            with torch.no_grad():
                action, logp, _, val = agent.get_action_and_value(a_obs)
                opp_action, _, _, _ = opponent_agent.get_action_and_value(o_obs)

            p1_act = torch.where(swap_sides, opp_action, action)
            p2_act = torch.where(swap_sides, action, opp_action)

            (no1, no2), rewards, dones, _ = env.step(torch.stack([p1_act, p2_act], 1).int())
            obs_queue_p1.append(no1);
            obs_queue_p2.append(no2)

            a_rew = torch.where(swap_sides, -rewards, rewards)

            # 记录缓冲区
            b_obs[step], b_act[step], b_logp[step], b_rew[step], b_done[step], b_val[step] = \
                a_obs, action, logp, a_rew, dones.float(), val.flatten()

            # --- 全局胜率滚动统计 (与原版逻辑完全同步) ---
            if dones.any():
                done_indices = torch.where(dones)[0]
                total_games += len(done_indices)
                agent_wins += (a_rew[done_indices] > 0).sum().item()

                # Q-Score 质量分更新
                if is_history_match:
                    win_count = (a_rew[done_indices] > 0).sum().item()
                    if win_count > 0:
                        raw_probs = np.exp(qs) / np.sum(np.exp(qs))
                        actual_p = (1 - config["alpha_sampling"]) * raw_probs[current_opp_idx] + config[
                            "alpha_sampling"] / len(pool_paths)
                        q_scores[current_opp_idx] -= (config["openai_eta"] / (len(pool_paths) * actual_p)) * win_count

        # --- PPO 学习过程 (仅 Agent 更新) ---
        agent.train()
        with torch.no_grad():
            next_value = agent.critic(a_obs).reshape(1, -1)
            advs = torch.zeros_like(b_rew)
            lastgaelam = 0
            for t in reversed(range(config["num_steps"])):
                not_done = 1.0 - b_done[t]
                next_val = next_value if t == config["num_steps"] - 1 else b_val[t + 1]
                delta = b_rew[t] + config["gamma"] * next_val * not_done - b_val[t]
                advs[t] = lastgaelam = delta + config["gamma"] * config["gae_lambda"] * not_done * lastgaelam
            returns = advs + b_val

        f_obs, f_act, f_logp, f_adv, f_ret = b_obs.reshape(-1, 48), b_act.reshape(-1), b_logp.reshape(-1), advs.reshape(
            -1), returns.reshape(-1)

        inds = np.arange(config["batch_size"])
        for _ in range(config["update_epochs"]):
            np.random.shuffle(inds)
            for s in range(0, config["batch_size"], 4096):
                mb = inds[s:s + 4096]
                _, new_logp, ent, new_v = agent.get_action_and_value(f_obs[mb], f_act[mb])
                ratio = (new_logp - f_logp[mb]).exp()
                m_adv = (f_adv[mb] - f_adv[mb].mean()) / (f_adv[mb].std() + 1e-8)
                pg_loss = torch.max(-m_adv * ratio, -m_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                v_loss = 0.5 * ((new_v.flatten() - f_ret[mb]) ** 2).mean()
                loss = pg_loss - config["ent_coef"] * ent.mean() + v_loss * config["vf_coef"]
                optimizer.zero_grad();
                loss.backward();
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5);
                optimizer.step()

        # --- 更换逻辑判定 ---
        total_agent_win_rate = agent_wins / total_games if total_games > 0 else 0
        writer.add_scalar("Train/Total_Win_Rate", total_agent_win_rate, global_step)

        checkpoint_data = {
            "model_state_dict": agent.state_dict(),
            "total_games": total_games,
            "agent_wins": agent_wins,
            "evolution_trigger_count": evolution_trigger_count
        }

        # 核心触发：只有全局滚动的综合胜率达标才更替考官
        if total_games >= config["min_games_to_replace"] and total_agent_win_rate >= config["auto_replace_threshold"]:
            evolution_trigger_count += 1
            print(f"\n[进化触发] 第 {evolution_trigger_count} 次胜率达标 ({total_agent_win_rate:.2%})！覆盖考官文件。")
            torch.save(checkpoint_data, opponent_model_path)

            # 归档到历史池
            if evolution_trigger_count % config["save_every_n_evolutions"] == 0:
                evolution_count += 1
                v_path = os.path.join(current_run_dir, f"evolution_v{evolution_count}.pth")
                torch.save(checkpoint_data, v_path)
                pool_paths.append(os.path.abspath(v_path));
                q_scores.append(max(q_scores) if q_scores else 1.0)

            # 严格重置全局滚动统计
            total_games, agent_wins = 0, 0

        # 定期保存断点
        torch.save(checkpoint_data, current_save_path)
        print(f"Step: {global_step:8d} | 累积局数: {total_games:4d} | 滚动胜率: {total_agent_win_rate:.2%}")


if __name__ == "__main__":
    train()