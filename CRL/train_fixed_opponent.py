import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os
import glob
from slime_env import SlimeSelfPlayEnv, FrameStack
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time
import random

# --- 配置参数 (保持不变) ---
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "slime_ppo_vs_fixed.pth",
    "p1_path": "模型集_opponent/train_20260202-021904/slime_ppo_vs_fixed.pth",
    "p2_path": "模型集_opponent/train_20260202-021904/slime_ppo_vs_fixed.pth",
    "resume_dir": "模型集_opponent/train_20260202-021904",
    "external_history_folder": "最强模型集",
    "start_step": 0,
    "auto_replace_threshold": 0.80,
    "win_rate_window": 100,
    "total_timesteps": 300_000_000,
    "num_envs": 32,
    "num_steps": 256,
    "update_epochs": 4,
    "batch_size": 2048,
    "lr": 2.5e-4,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "openai_eta": 0.1,
    "historical_ratio": 0.4,
    "alpha_sampling": 0.1,
    "save_every_n_evolutions": 5,
}


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(52, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(52, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 4)
        )

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None: action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def make_env():
    return lambda: FrameStack(SlimeSelfPlayEnv(render_mode=None), n_frames=4)


def train():
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    checkpoint_root = "模型集_opponent"

    if config["resume_dir"] and os.path.exists(config["resume_dir"]):
        current_run_dir = config["resume_dir"]
        is_resume = True
    else:
        current_run_dir = os.path.join(checkpoint_root, f"train_{timestamp}")
        os.makedirs(current_run_dir, exist_ok=True)
        is_resume = False

    current_save_path = os.path.join(current_run_dir, config["save_path"])
    opponent_model_path = os.path.join(current_run_dir, "fixed_opponent_current.pth")

    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(config["num_envs"])])
    agent = Agent().to(config["device"])

    opponents = [Agent().to(config["device"]) for _ in range(config["num_envs"])]
    for opp in opponents:
        opp.eval()
        for param in opp.parameters(): param.requires_grad = False

    global_step = config["start_step"]
    evolution_count = 0
    evolution_trigger_count = 0
    total_games = 0
    agent_wins = 0
    recent_wins = deque(maxlen=config["win_rate_window"])
    games_after_evolution = 0
    WARMUP_GAMES = 20

    # 1. Agent 权重加载
    if os.path.exists(current_save_path):
        checkpoint = torch.load(current_save_path, map_location=config["device"])
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            agent.load_state_dict(checkpoint["model_state_dict"])
            total_games = checkpoint.get("total_games", 0)
            agent_wins = checkpoint.get("agent_wins", 0)
            evolution_trigger_count = checkpoint.get("evolution_trigger_count", 0)
        else:
            agent.load_state_dict(checkpoint)
    elif config["p1_path"] and os.path.exists(config["p1_path"]):
        checkpoint = torch.load(config["p1_path"], map_location=config["device"])
        state = checkpoint["model_state_dict"] if isinstance(checkpoint,
                                                             dict) and "model_state_dict" in checkpoint else checkpoint
        agent.load_state_dict(state)
        print(f"检测到起点模型，Agent 加载自: {config['p1_path']}")

    # 2. 对手池初始化
    opponent_pool_paths = []
    if os.path.exists(config["external_history_folder"]):
        ext_files = glob.glob(os.path.join(config["external_history_folder"], "*.pth"))
        opponent_pool_paths.extend([os.path.abspath(f) for f in ext_files])

    history_files = glob.glob(os.path.join(current_run_dir, "evolution_v*.pth"))
    history_files.sort(key=lambda x: int(os.path.basename(x).replace('evolution_v', '').replace('.pth', '')))
    opponent_pool_paths.extend([os.path.abspath(p) for p in history_files])
    q_scores = [1.0] * len(opponent_pool_paths)
    evolution_count = len(history_files)

    def load_opponent_to_env(env_idx, path):
        if not os.path.exists(path): return
        opp_ckpt = torch.load(path, map_location=config["device"])
        opp_state = opp_ckpt["model_state_dict"] if isinstance(opp_ckpt,
                                                               dict) and "model_state_dict" in opp_ckpt else opp_ckpt
        opponents[env_idx].load_state_dict(opp_state)
        opponents[env_idx].eval()

    if not os.path.exists(opponent_model_path):
        initial_opp_path = config["p2_path"] if config["p2_path"] else config["p1_path"]
        if initial_opp_path and os.path.exists(initial_opp_path):
            torch.save(torch.load(initial_opp_path, map_location=config["device"]), opponent_model_path)
        else:
            torch.save(agent.state_dict(), opponent_model_path)

    for i in range(config["num_envs"]):
        load_opponent_to_env(i, opponent_model_path)

    writer = SummaryWriter(f"runs/vs_fixed_{timestamp}" + ("_resume" if is_resume else ""))
    optimizer = optim.Adam(agent.parameters(), lr=config["lr"])
    current_opp_paths = [opponent_model_path for _ in range(config["num_envs"])]
    current_opp_indices = [-1 for _ in range(config["num_envs"])]

    # 显存缓冲区
    obs_buf = torch.zeros((config["num_steps"], config["num_envs"], 52)).to(config["device"])
    act_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    logp_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    rew_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    done_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    val_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])

    obs_p1, infos = envs.reset()
    # P2 的帧堆叠逻辑（因为 P2 是镜像视角）
    p2_deques = [deque(maxlen=4) for _ in range(config["num_envs"])]
    for i in range(config["num_envs"]):
        init_p2 = infos["p2_raw_obs"][i] if "p2_raw_obs" in infos else np.zeros(13)
        for _ in range(4): p2_deques[i].append(init_p2)
    obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

    while global_step < config["total_timesteps"]:
        agent.eval()
        for step in range(config["num_steps"]):
            global_step += config["num_envs"]

            # --- 核心修改：取消 side_swapped，Agent 永远是 P1 ---
            t_obs_agent = torch.from_numpy(obs_p1).float().to(config["device"])
            t_obs_opp = torch.from_numpy(obs_p2).float().to(config["device"])

            with torch.no_grad():
                actions_agent, logp_agent, _, values_agent = agent.get_action_and_value(t_obs_agent)
                actions_opp = torch.zeros(config["num_envs"], device=config["device"])
                for i in range(config["num_envs"]):
                    logits_opp = opponents[i].actor(t_obs_opp[i:i + 1])
                    actions_opp[i] = torch.distributions.Categorical(logits=logits_opp).sample()

            # 组装环境动作 [P1, P2]
            env_actions = np.stack([actions_agent.cpu().numpy(), actions_opp.cpu().numpy()], axis=1).astype(np.int32)

            n_obs_p1, reward, term, trunc, infos = envs.step(env_actions)

            for i in range(config["num_envs"]):
                # Agent 永远是 P1，所以 Reward 直接取环境返回值
                rew_buf[step][i] = reward[i]

                if term[i] or trunc[i]:
                    total_games += 1
                    # Agent 永远是 P1
                    is_agent_win = 1 if infos["p1_score"][i] > infos["p2_score"][i] else 0

                    if games_after_evolution >= WARMUP_GAMES:
                        agent_wins += is_agent_win
                        recent_wins.append(is_agent_win)
                    games_after_evolution += 1

                    # 质量分更新
                    if current_opp_indices[i] != -1 and is_agent_win:
                        qs = np.array(q_scores)
                        raw_probs = np.exp(qs - np.max(qs)) / np.sum(np.exp(qs - np.max(qs)))
                        actual_prob = (1 - config["alpha_sampling"]) * raw_probs[current_opp_indices[i]] + (
                                    config["alpha_sampling"] / len(opponent_pool_paths))
                        q_scores[current_opp_indices[i]] -= config["openai_eta"] / (
                                    len(opponent_pool_paths) * actual_prob)

                    # 采样新对手
                    if len(opponent_pool_paths) > 0 and random.random() < config["historical_ratio"]:
                        qs = np.array(q_scores)
                        softmax_probs = np.exp(qs - np.max(qs)) / np.sum(np.exp(qs - np.max(qs)))
                        final_probs = (1 - config["alpha_sampling"]) * softmax_probs + config["alpha_sampling"] * (
                                    np.ones_like(qs) / len(qs))
                        idx = np.random.choice(len(opponent_pool_paths), p=final_probs)
                        path, current_opp_indices[i] = opponent_pool_paths[idx], idx
                    else:
                        path, current_opp_indices[i] = opponent_model_path, -1

                    load_opponent_to_env(i, path)
                    current_opp_paths[i] = path

                    if "episode_steps" in infos:
                        writer.add_scalar("Game/Episode_Steps", infos["episode_steps"][i], total_games)

                    p2_deques[i].clear()
                    for _ in range(4): p2_deques[i].append(infos["p2_raw_obs"][i])
                else:
                    p2_deques[i].append(infos["p2_raw_obs"][i])

            obs_buf[step], act_buf[step], logp_buf[step], val_buf[
                step] = t_obs_agent, actions_agent, logp_agent, values_agent.flatten()
            done_buf[step] = torch.from_numpy((term | trunc).astype(np.float32)).to(config["device"])
            obs_p1, obs_p2 = n_obs_p1, np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

        # PPO 更新逻辑 (保持不变)
        with torch.no_grad():
            t_next_obs = torch.from_numpy(obs_p1).float().to(config["device"])
            _, _, _, next_val = agent.get_action_and_value(t_next_obs)
            adv = torch.zeros_like(rew_buf).to(config["device"])
            lastgae = 0
            for t in reversed(range(config["num_steps"])):
                nt = 1.0 - done_buf[t]
                nv = next_val.flatten() if t == config["num_steps"] - 1 else val_buf[t + 1]
                delta = rew_buf[t] + 0.99 * nv * nt - val_buf[t]
                adv[t] = lastgae = delta + 0.99 * 0.95 * nt * lastgae
            ret = adv + val_buf

        agent.train()
        b_obs, b_logp, b_act, b_adv, b_ret = obs_buf.reshape(-1, 52), logp_buf.reshape(-1), act_buf.reshape(
            -1), adv.reshape(-1), ret.reshape(-1)
        indices = np.arange(config["num_steps"] * config["num_envs"])
        for _ in range(config["update_epochs"]):
            np.random.shuffle(indices)
            for s in range(0, len(indices), config["batch_size"]):
                mb = indices[s:s + config["batch_size"]]
                _, newlogp, ent, newv = agent.get_action_and_value(b_obs[mb], b_act[mb])
                ratio = (newlogp - b_logp[mb]).exp()
                m_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)
                pg_loss = torch.max(-m_adv * ratio, -m_adv * torch.clamp(ratio, 0.8, 1.2)).mean()
                v_loss = 0.5 * ((newv.flatten() - b_ret[mb]) ** 2).mean()
                loss = pg_loss - config["ent_coef"] * ent.mean() + v_loss * config["vf_coef"]
                optimizer.zero_grad();
                loss.backward();
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5);
                optimizer.step()

        # 记录与进化逻辑 (保持打印内容和 TB 逻辑)
        current_window_win_rate = sum(recent_wins) / len(recent_wins) if len(recent_wins) > 0 else 0
        writer.add_scalar("Train/Total_Win_Rate", current_window_win_rate, global_step)

        for idx, score in enumerate(q_scores):
            opp_name = os.path.basename(opponent_pool_paths[idx])
            writer.add_scalar(f"Opponent_Scores/{opp_name}", score, global_step)

        checkpoint_data = {
            "model_state_dict": agent.state_dict(),
            "total_games": total_games, "agent_wins": agent_wins, "evolution_trigger_count": evolution_trigger_count
        }

        if len(recent_wins) >= config["win_rate_window"] and current_window_win_rate >= config[
            "auto_replace_threshold"]:
            evolution_trigger_count += 1
            print(f"\n[进化触发] 窗口胜率达标 ({current_window_win_rate:.2%})，替换考官！")
            torch.save(checkpoint_data, opponent_model_path)
            if evolution_trigger_count % config["save_every_n_evolutions"] == 0:
                evolution_count += 1
                new_v_path = os.path.join(current_run_dir, f"evolution_v{evolution_count}.pth")
                torch.save(checkpoint_data, new_v_path)
                opponent_pool_paths.append(new_v_path)
                q_scores.append(max(q_scores) if q_scores else 1.0)

            recent_wins.clear();
            total_games, agent_wins = 0, 0;
            games_after_evolution = 0

        torch.save(checkpoint_data, current_save_path)
        q_info = f" | 池分均值: {np.mean(q_scores):.2f}" if q_scores else ""
        opp_0_name = os.path.basename(current_opp_paths[0]) if os.path.exists(
            current_opp_paths[0]) else "Random_Whiteboard"
        print(
            f"步数: {global_step:7d} | 周期局数: {total_games:4d} | 窗口胜率: {current_window_win_rate:.2%}{q_info} | Env0对手: {opp_0_name}")

    envs.close();
    writer.close()


if __name__ == "__main__":
    train()