import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from slime_env import SlimeSelfPlayEnv, FrameStack
from torch.utils.tensorboard import SummaryWriter
from collections import deque

# --- 配置参数 ---
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "slime_ppo_gpu.pth",
    "total_timesteps": 10000000,
    "num_envs": 24,
    "num_steps": 2048,
    "update_epochs": 10,
    "batch_size": 4096,
    "lr": 3e-4,
    "ent_coef": 0.05,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
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
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(config["num_envs"])])
    agent = Agent().to(config["device"])

    try:
        agent.load_state_dict(torch.load(config["save_path"], map_location=config["device"]))
        print(">>> 已加载权重，继续训练...")
    except:
        print(">>> 从零启动...")

    optimizer = optim.Adam(agent.parameters(), lr=config["lr"])
    writer = SummaryWriter("runs/slime_ppo_vs")

    # 缓冲区
    obs_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2, 48)).to(config["device"])
    act_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    logp_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    rew_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    done_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    val_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])

    obs_p1, _ = envs.reset()
    p2_deques = [deque(maxlen=4) for _ in range(config["num_envs"])]
    temp_env = SlimeSelfPlayEnv();
    temp_env.reset()
    init_p2_raw = temp_env._get_obs(2)
    for d in p2_deques: [d.append(init_p2_raw) for _ in range(4)]
    obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

    global_step = 0

    while global_step < config["total_timesteps"]:
        agent.eval()
        p1_rewards_list = []
        p2_rewards_list = []

        for step in range(config["num_steps"]):
            global_step += config["num_envs"]
            t_obs = torch.from_numpy(np.concatenate([obs_p1, obs_p2])).float().to(config["device"])
            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(t_obs)

            p1_acts, p2_acts = actions[:config["num_envs"]].cpu().numpy(), actions[config["num_envs"]:].cpu().numpy()
            n_obs_p1, reward, term, trunc, infos = envs.step(np.stack([p1_acts, p2_acts], axis=1))

            for i in range(config["num_envs"]): p2_deques[i].append(infos["p2_raw_obs"][i])
            n_obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

            obs_buf[step], act_buf[step], logp_buf[step], val_buf[step] = t_obs, actions, logprobs, values.flatten()

            rew_buf[step, :config["num_envs"]] = torch.from_numpy(reward).to(config["device"])
            rew_buf[step, config["num_envs"]:] = torch.from_numpy(-reward).to(config["device"])

            done_buf[step] = torch.from_numpy((term | trunc).astype(np.float32)).repeat(2).to(config["device"])
            obs_p1, obs_p2 = n_obs_p1, n_obs_p2

            # 收集有效奖励用于监控
            valid_rewards = reward[reward != 0]
            if len(valid_rewards) > 0:
                p1_rewards_list.extend(valid_rewards.tolist())
                p2_rewards_list.extend((-valid_rewards).tolist())

        # 计算本轮监控指标
        m_p1_rew = np.mean(p1_rewards_list) if p1_rewards_list else 0.0
        m_p2_rew = np.mean(p2_rewards_list) if p2_rewards_list else 0.0
        p1_wins = np.sum(np.array(p1_rewards_list) > 0)
        total_points = len(p1_rewards_list)
        p1_win_rate = p1_wins / total_points if total_points > 0 else 0.5

        # --- GAE 计算 ---
        with torch.no_grad():
            next_obs = torch.from_numpy(np.concatenate([obs_p1, obs_p2])).float().to(config["device"])
            _, _, _, next_val = agent.get_action_and_value(next_obs)
            adv = torch.zeros_like(rew_buf).to(config["device"])
            lastgae = 0
            for t in reversed(range(config["num_steps"])):
                nt = 1.0 - done_buf[t]
                nv = next_val.flatten() if t == config["num_steps"] - 1 else val_buf[t + 1]
                delta = rew_buf[t] + 0.99 * nv * nt - val_buf[t]
                adv[t] = lastgae = delta + 0.99 * 0.95 * nt * lastgae
            ret = adv + val_buf

        # --- 更新网络 ---
        agent.train()
        b_obs, b_logp, b_act, b_adv, b_ret = obs_buf.reshape(-1, 48), logp_buf.reshape(-1), act_buf.reshape(
            -1), adv.reshape(-1), ret.reshape(-1)
        indices = np.arange(config["num_steps"] * config["num_envs"] * 2)

        pg_losses, v_losses, ent_losses = [], [], []
        for _ in range(config["update_epochs"]):
            np.random.shuffle(indices)
            for s in range(0, len(indices), config["batch_size"]):
                mb = indices[s:s + config["batch_size"]]
                _, newlogp, ent, newv = agent.get_action_and_value(b_obs[mb], b_act[mb])

                ratio = (newlogp - b_logp[mb]).exp()
                # 统一使用 m_adv
                m_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)

                pg_loss1 = -m_adv * ratio
                pg_loss2 = -m_adv * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newv.flatten() - b_ret[mb]) ** 2).mean()
                entropy = ent.mean()

                loss = pg_loss - config["ent_coef"] * entropy + v_loss * config["vf_coef"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ent_losses.append(entropy.item())

        # --- 记录数据 ---
        writer.add_scalar("VS/P1_Mean_Reward", m_p1_rew, global_step)
        writer.add_scalar("VS/P2_Mean_Reward", m_p2_rew, global_step)
        writer.add_scalar("VS/P1_Win_Rate", p1_win_rate, global_step)
        writer.add_scalar("Loss/Policy", np.mean(pg_losses), global_step)
        writer.add_scalar("Loss/Value", np.mean(v_losses), global_step)

        print(f"步数: {global_step:7d} | P1奖励: {m_p1_rew:+.2f} | P2奖励: {m_p2_rew:+.2f} | P1胜率: {p1_win_rate:.2%}")
        torch.save(agent.state_dict(), config["save_path"])

    envs.close();
    writer.close()


if __name__ == "__main__": train()