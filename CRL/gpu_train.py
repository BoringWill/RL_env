import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, glob, time, random
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from gpu_slime_env import SlimeVolleyballGPU
from constants import *

# --- 沿用你的所有配置参数 ---
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "gpu_slime_ppo_vs_fixed.pth",
    "p1_path": "模型集_opponent/train_20260130-010209/slime_ppo_vs_fixed.pth",
    "p2_path": "模型集_opponent/train_20260130-010209/slime_ppo_vs_fixed.pth",
    "resume_dir": "模型集_opponent/train_20260130-010209",
    "external_history_folder": "最强模型集",
    "start_step": 0,
    "auto_replace_threshold": 0.80,
    "win_rate_window": 150,
    "total_timesteps": 30_000_000,
    "num_envs": 256,  # GPU版可以开更大
    "num_steps": 256,
    "update_epochs": 4,
    "batch_size": 2048,
    "lr": 2.5e-4,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "openai_eta": 0.1,
    "historical_ratio": 0.3,
    "alpha_sampling": 0.1,
    "save_every_n_evolutions": 5,
}


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.actor = nn.Sequential(nn.Linear(52, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 4))

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def train():
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    checkpoint_root = "模型集_opponent"

    # 路径处理
    if config["resume_dir"] and os.path.exists(config["resume_dir"]):
        current_run_dir = config["resume_dir"]
    else:
        current_run_dir = os.path.join(checkpoint_root, f"train_gpu_{timestamp}")
        os.makedirs(current_run_dir, exist_ok=True)

    current_save_path = os.path.join(current_run_dir, config["save_path"])
    opponent_model_path = os.path.join(current_run_dir, "fixed_opponent_current.pth")
    writer = SummaryWriter(f"runs/gpu_vs_fixed_{timestamp}")

    # 环境与 Agent 初始化
    env = SlimeVolleyballGPU(num_envs=config["num_envs"], device=config["device"])
    agent = Agent().to(config["device"])
    optimizer = optim.Adam(agent.parameters(), lr=config["lr"])

    # 对手池与 Q-Score 逻辑
    opponent_pool_paths = []
    if os.path.exists(config["external_history_folder"]):
        ext_files = glob.glob(os.path.join(config["external_history_folder"], "*.pth"))
        opponent_pool_paths.extend([os.path.abspath(f) for f in ext_files])

    history_files = glob.glob(os.path.join(current_run_dir, "evolution_v*.pth"))
    history_files.sort(key=lambda x: int(os.path.basename(x).split('_v')[-1].split('.pth')[0]) if '_v' in x else 0)
    opponent_pool_paths.extend([os.path.abspath(p) for p in history_files])
    q_scores = [1.0] * len(opponent_pool_paths)
    evolution_count = len(history_files)

    # 每个环境独立的对手网络 (用于推理)
    opponents = [Agent().to(config["device"]) for _ in range(config["num_envs"])]
    current_opp_indices = [-1] * config["num_envs"]  # -1 表示当前固定考官

    def load_opp(env_idx, path):
        ckpt = torch.load(path, map_location=config["device"])
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        opponents[env_idx].load_state_dict(state)
        opponents[env_idx].eval()

    # 初始化考官
    if not os.path.exists(opponent_model_path):
        init_path = config["p2_path"] if config["p2_path"] else config["p1_path"]
        if init_path and os.path.exists(init_path):
            torch.save(torch.load(init_path), opponent_model_path)
        else:
            torch.save(agent.state_dict(), opponent_model_path)

    for i in range(config["num_envs"]): load_opp(i, opponent_model_path)

    # 加载 Agent 存档
    if os.path.exists(current_save_path):
        ckpt = torch.load(current_save_path)
        agent.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)

    # 显存中的 FrameStack 和 Buffer
    stack_p1 = torch.zeros((config["num_envs"], 4, 13), device=config["device"])
    stack_p2 = torch.zeros((config["num_envs"], 4, 13), device=config["device"])
    obs_buf = torch.zeros((config["num_steps"], config["num_envs"], 52), device=config["device"])
    act_buf = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
    logp_buf = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
    rew_buf = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
    done_buf = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
    val_buf = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])

    # 统计变量
    global_step = config["start_step"]
    recent_wins = deque(maxlen=config["win_rate_window"])
    total_games, agent_wins, evolution_trigger_count, games_after_evolution = 0, 0, 0, 0
    side_swapped = (torch.rand(config["num_envs"]) > 0.5).to(config["device"])

    raw_p1, raw_p2 = env.reset()
    for t in range(4): stack_p1[:, t] = raw_p1; stack_p2[:, t] = raw_p2

    while global_step < config["total_timesteps"]:
        # --- 1. 数据采集 ---
        agent.eval()
        for step in range(config["num_steps"]):
            global_step += config["num_envs"]

            # 准备观测 (Agent 视角 vs 对手视角)
            cur_obs_agent = torch.where(side_swapped.view(-1, 1, 1), stack_p2, stack_p1).view(config["num_envs"], -1)
            cur_obs_opp = torch.where(side_swapped.view(-1, 1, 1), stack_p1, stack_p2).view(config["num_envs"], -1)

            with torch.no_grad():
                act_agent, logp_agent, _, val_agent = agent.get_action_and_value(cur_obs_agent)
                act_opp = torch.zeros(config["num_envs"], device=config["device"], dtype=torch.long)
                for i in range(config["num_envs"]):
                    logits_o = opponents[i].actor(cur_obs_opp[i:i + 1])
                    act_opp[i] = torch.distributions.Categorical(logits=logits_o).sample()

            # 环境 Step (根据 side_swapped 决定谁是 P1/P2)
            env_act_p1 = torch.where(side_swapped, act_opp, act_agent)
            env_act_p2 = torch.where(side_swapped, act_agent, act_opp)

            (next_raw_p1, next_raw_p2), rewards, dones = env.step(env_act_p1, env_act_p2)

            # 计算 Agent 获得的奖励 (如果换边了，奖励取反)
            agent_rew = torch.where(side_swapped, -rewards, rewards)

            # 存入 Buffer
            obs_buf[step], act_buf[step], logp_buf[step] = cur_obs_agent, act_agent, logp_agent
            rew_buf[step], val_buf[step], done_buf[step] = agent_rew, val_agent.flatten(), dones.float()

            # 更新堆叠
            stack_p1 = torch.roll(stack_p1, -1, dims=1);
            stack_p1[:, -1] = next_raw_p1
            stack_p2 = torch.roll(stack_p2, -1, dims=1);
            stack_p2[:, -1] = next_raw_p2

            # 游戏结束处理
            if dones.any():
                for i in torch.where(dones)[0]:
                    total_games += 1
                    is_win = 1 if agent_rew[i] > 0 else 0
                    if games_after_evolution >= 30:
                        agent_wins += is_win
                        recent_wins.append(is_win)
                    games_after_evolution += 1

                    # --- OpenAI Q-Score 质量更新 ---
                    if current_opp_indices[i] != -1 and is_win:
                        qs = np.array(q_scores)
                        raw_probs = np.exp(qs - np.max(qs)) / np.sum(np.exp(qs - np.max(qs)))
                        actual_prob = (1 - config["alpha_sampling"]) * raw_probs[current_opp_indices[i]] + (
                                    config["alpha_sampling"] / len(opponent_pool_paths))
                        q_scores[current_opp_indices[i]] -= config["openai_eta"] / (
                                    len(opponent_pool_paths) * actual_prob)

                    # --- 对手采样逻辑 ---
                    if len(opponent_pool_paths) > 0 and random.random() < config["historical_ratio"]:
                        qs = np.array(q_scores)
                        softmax_probs = np.exp(qs - np.max(qs)) / np.sum(np.exp(qs - np.max(qs)))
                        final_probs = (1 - config["alpha_sampling"]) * softmax_probs + config["alpha_sampling"] * (
                                    np.ones_like(qs) / len(qs))
                        idx = np.random.choice(len(opponent_pool_paths), p=final_probs)
                        path, current_opp_indices[i] = opponent_pool_paths[idx], idx
                    else:
                        path, current_opp_indices[i] = opponent_model_path, -1

                    load_opp(i, path)
                    side_swapped[i] = torch.rand(1) > 0.5

                    # 重置该环境堆叠
                    r1, r2 = env.reset(i.unsqueeze(0))
                    stack_p1[i] = r1.repeat(4, 1);
                    stack_p2[i] = r2.repeat(4, 1)

        # --- 2. PPO 更新 ---
        agent.train()
        # [计算优势函数 GAE]
        with torch.no_grad():
            last_obs = torch.where(side_swapped.view(-1, 1, 1), stack_p2, stack_p1).view(config["num_envs"], -1)
            next_val = agent.critic(last_obs).flatten()
            adv = torch.zeros_like(rew_buf);
            lastgae = 0
            for t in reversed(range(config["num_steps"])):
                nn_t = 1.0 - done_buf[t]
                nv = next_val if t == config["num_steps"] - 1 else val_buf[t + 1]
                delta = rew_buf[t] + 0.99 * nv * nn_t - val_buf[t]
                adv[t] = lastgae = delta + 0.99 * 0.95 * nn_t * lastgae
            returns = adv + val_buf

        # [批量训练]
        b_obs, b_logp, b_act, b_adv, b_ret = obs_buf.view(-1, 52), logp_buf.view(-1), act_buf.view(-1), adv.view(
            -1), returns.view(-1)
        inds = np.arange(config["num_steps"] * config["num_envs"])
        for epoch in range(config["update_epochs"]):
            np.random.shuffle(inds)
            for s in range(0, len(inds), config["batch_size"]):
                mb = inds[s:s + config["batch_size"]]
                _, newlogp, ent, newv = agent.get_action_and_value(b_obs[mb], b_act[mb])
                ratio = (newlogp - b_logp[mb]).exp()
                mb_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                v_loss = 0.5 * ((newv.flatten() - b_ret[mb]) ** 2).mean()
                loss = pg_loss - config["ent_coef"] * ent.mean() + v_loss * config["vf_coef"]
                optimizer.zero_grad();
                loss.backward();
                nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"]);
                optimizer.step()

        # --- 3. 监控与进化逻辑 ---
        win_rate = sum(recent_wins) / len(recent_wins) if recent_wins else 0
        writer.add_scalar("Train/WinRate", win_rate, global_step)

        if len(recent_wins) >= config["win_rate_window"] and win_rate >= config["auto_replace_threshold"]:
            evolution_trigger_count += 1
            print(f"\n[进化] 胜率 {win_rate:.2%}, 替换考官!")
            ckpt = {"model_state_dict": agent.state_dict(), "total_games": total_games}
            torch.save(ckpt, opponent_model_path)

            if evolution_trigger_count % config["save_every_n_evolutions"] == 0:
                evolution_count += 1
                new_v_path = os.path.join(current_run_dir, f"evolution_v{evolution_count}.pth")
                torch.save(ckpt, new_v_path)
                opponent_pool_paths.append(new_v_path);
                q_scores.append(max(q_scores) if q_scores else 1.0)

            recent_wins.clear();
            games_after_evolution = 0
            for i in range(config["num_envs"]): load_opp(i, opponent_model_path)

        torch.save({"model_state_dict": agent.state_dict()}, current_save_path)
        print(f"Step: {global_step:7d} | WinRate: {win_rate:.2%} | Pool: {len(opponent_pool_paths)}")


if __name__ == "__main__":
    train()