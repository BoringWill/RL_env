import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os
from slime_env import SlimeSelfPlayEnv, FrameStack
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time
import random

# --- 配置参数 ---
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "slime_ppo_vs_fixed.pth",
    "p1_path": "模型集_历代版本最强/evolution_v1.pth",
    "p2_path": "模型集_历代版本最强/evolution_v1.pth",
    "p2_epsilon": 0.05,
    "auto_replace_threshold": 0.80,
    "min_games_to_replace": 400,
    "total_timesteps": 300000000,
    "num_envs": 32,
    "num_steps": 2048,
    "update_epochs": 10,
    "batch_size": 8192,
    "lr": 3e-4,
    "ent_coef": 0.05,
    "min_ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    # OpenAI 策略参数
    "openai_eta": 0.1,
    "historical_ratio": 0.2,
    "save_every_n_evolutions": 2,  # 已改为每2次达标保存一个文件
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
    current_run_dir = os.path.join(checkpoint_root, f"train_{timestamp}")

    if not os.path.exists(current_run_dir):
        os.makedirs(current_run_dir)

    current_save_path = os.path.join(current_run_dir, config["save_path"])
    opponent_model_path = os.path.join(current_run_dir, "fixed_opponent_current.pth")

    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(config["num_envs"])])

    agent = Agent().to(config["device"])
    opponent = Agent().to(config["device"])

    if os.path.exists(config["p1_path"]):
        try:
            agent.load_state_dict(torch.load(config["p1_path"], map_location=config["device"], weights_only=False))
            print(f">>>> 成功加载 P1 初始权重: {config['p1_path']}")
        except Exception as e:
            print(f">>>> 加载 P1 权重失败: {e}")

    if os.path.exists(config["p2_path"]):
        opponent.load_state_dict(torch.load(config["p2_path"], map_location=config["device"], weights_only=False))
        opponent.eval()
        torch.save(opponent.state_dict(), opponent_model_path)
        print(f">>>> 成功加载 P2 权重: {config['p2_path']}")
    else:
        print(f">>>> 警告: 未找到 {config['p2_path']}")
        return

    # OpenAI 对手池初始化
    opponent_pool_paths = []
    q_scores = []

    optimizer = optim.Adam(agent.parameters(), lr=config["lr"])
    writer = SummaryWriter(f"runs/vs_fixed_{timestamp}")

    obs_buf = torch.zeros((config["num_steps"], config["num_envs"], 48)).to(config["device"])
    act_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    logp_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    rew_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    done_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    val_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])

    obs_p1, infos = envs.reset()
    p2_deques = [deque(maxlen=4) for _ in range(config["num_envs"])]

    for i in range(config["num_envs"]):
        init_p2 = infos["p2_raw_obs"][i] if "p2_raw_obs" in infos else np.zeros(12)
        for _ in range(4): p2_deques[i].append(init_p2)
    obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

    global_step = 0
    total_games = 0
    agent_wins = 0
    evolution_count = 0
    evolution_trigger_count = 0
    recent_wins = deque(maxlen=10)
    last_save_step = 0

    while global_step < config["total_timesteps"]:
        # --- 对手选择逻辑 ---
        chosen_idx = -1
        current_p_i = 1.0
        if len(opponent_pool_paths) > 0 and random.random() < config["historical_ratio"]:
            qs = np.array(q_scores)
            exp_q = np.exp(qs - np.max(qs))
            probs = exp_q / np.sum(exp_q)
            chosen_idx = np.random.choice(len(opponent_pool_paths), p=probs)
            current_p_i = probs[chosen_idx]
            chosen_opponent_path = opponent_pool_paths[chosen_idx]
        else:
            chosen_opponent_path = opponent_model_path

        opponent.load_state_dict(torch.load(chosen_opponent_path, map_location=config["device"]))
        opponent.eval()

        agent.eval()
        frac = max(0.0, 1.0 - (global_step / config["total_timesteps"]))
        current_ent_coef = config["min_ent_coef"] + (config["ent_coef"] - config["min_ent_coef"]) * frac

        side_swapped = np.random.rand(config["num_envs"]) > 0.5

        # 采样结果统计
        rollout_total_games = 0
        rollout_agent_wins = 0

        for step in range(config["num_steps"]):
            global_step += config["num_envs"]

            t_obs_agent = torch.zeros((config["num_envs"], 48)).to(config["device"])
            t_obs_opp = torch.zeros((config["num_envs"], 48)).to(config["device"])

            for i in range(config["num_envs"]):
                if not side_swapped[i]:
                    t_obs_agent[i] = torch.from_numpy(obs_p1[i]).float()
                    t_obs_opp[i] = torch.from_numpy(obs_p2[i]).float()
                else:
                    t_obs_agent[i] = torch.from_numpy(obs_p2[i]).float()
                    t_obs_opp[i] = torch.from_numpy(obs_p1[i]).float()

            with torch.no_grad():
                actions_agent, logp_agent, _, values_agent = agent.get_action_and_value(t_obs_agent)
                logits_opp = opponent.actor(t_obs_opp)
                actions_opp = torch.distributions.Categorical(logits=logits_opp).sample()

            env_actions = np.zeros((config["num_envs"], 2), dtype=np.int32)
            for i in range(config["num_envs"]):
                if not side_swapped[i]:
                    env_actions[i] = [actions_agent[i].item(), actions_opp[i].item()]
                else:
                    env_actions[i] = [actions_opp[i].item(), actions_agent[i].item()]

            n_obs_p1, reward, term, trunc, infos = envs.step(env_actions)

            for i in range(config["num_envs"]):
                agent_step_reward = reward[i] if not side_swapped[i] else -reward[i]
                rew_buf[step][i] = agent_step_reward

                if term[i] or trunc[i]:
                    total_games += 1
                    rollout_total_games += 1
                    if not side_swapped[i]:
                        is_agent_win = 1 if infos["p1_score"][i] > infos["p2_score"][i] else 0
                    else:
                        is_agent_win = 1 if infos["p2_score"][i] > infos["p1_score"][i] else 0

                    agent_wins += is_agent_win
                    rollout_agent_wins += is_agent_win
                    recent_wins.append(is_agent_win)

                    if "episode_steps" in infos:
                        writer.add_scalar("Game/Episode_Steps", infos["episode_steps"][i], total_games)

                    p2_deques[i].clear()
                    for _ in range(4): p2_deques[i].append(infos["p2_raw_obs"][i])
                else:
                    p2_deques[i].append(infos["p2_raw_obs"][i])

            obs_buf[step], act_buf[step], logp_buf[step], val_buf[step] = \
                t_obs_agent, actions_agent, logp_agent, values_agent.flatten()
            done_buf[step] = torch.from_numpy((term | trunc).astype(np.float32)).to(config["device"])

            obs_p1 = n_obs_p1
            obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

        # --- 质量分更新逻辑 ---
        if chosen_idx != -1 and rollout_total_games > 0:
            if (rollout_agent_wins / rollout_total_games) > 0.5:
                q_scores[chosen_idx] -= config["openai_eta"] / (len(opponent_pool_paths) * current_p_i)

        with torch.no_grad():
            t_next_obs = torch.zeros((config["num_envs"], 48)).to(config["device"])
            for i in range(config["num_envs"]):
                t_next_obs[i] = torch.from_numpy(obs_p2[i] if side_swapped[i] else obs_p1[i]).float()
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
        b_obs, b_logp, b_act, b_adv, b_ret = obs_buf.reshape(-1, 48), logp_buf.reshape(-1), act_buf.reshape(
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
                loss = pg_loss - current_ent_coef * ent.mean() + v_loss * config["vf_coef"]
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        total_agent_win_rate = agent_wins / total_games if total_games > 0 else 0
        recent_agent_win_rate = sum(recent_wins) / len(recent_wins) if len(recent_wins) > 0 else 0

        writer.add_scalar("Train/Total_Win_Rate", total_agent_win_rate, global_step)
        writer.add_scalar("Train/Recent_Win_Rate", recent_agent_win_rate, global_step)

        # --- 每 2 次达标保存一次文件 ---
        if total_games >= config["min_games_to_replace"] and total_agent_win_rate >= config["auto_replace_threshold"]:
            evolution_trigger_count += 1
            print(f"\n[进化触发] 第 {evolution_trigger_count} 次胜率达标！更替当前考官权重...")

            torch.save(agent.state_dict(), opponent_model_path)

            if evolution_trigger_count % config["save_every_n_evolutions"] == 0:
                evolution_count += 1
                new_v_path = os.path.join(current_run_dir, f"evolution_v{evolution_count}.pth")
                torch.save(agent.state_dict(), new_v_path)

                opponent_pool_paths.append(new_v_path)
                q_scores.append(max(q_scores) if q_scores else 1.0)
                print(f">>>> 已保存里程碑模型文件: {os.path.basename(new_v_path)}")

            total_games, agent_wins = 0, 0
            recent_wins.clear()

        print(
            f"步数: {global_step:7d} | 对局数: {total_games:5d} | 学生模型胜率: {total_agent_win_rate:.2%} | 进化触发: {evolution_trigger_count} | 对手: {os.path.basename(chosen_opponent_path)}")

        torch.save(agent.state_dict(), current_save_path)

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()