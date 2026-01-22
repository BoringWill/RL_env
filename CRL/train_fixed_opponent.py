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

# --- 配置参数 ---
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "slime_ppo_vs_fixed.pth",  # 训练产出的模型保存路径 (P1)
    "opponent_path": "模型集/slime_ppo_gpu_v2.pth",  # 固定对手权重 (P2)
    "total_timesteps": 20000000,
    "num_envs": 32,
    "num_steps": 2048,
    "update_epochs": 10,
    "batch_size": 8192,
    "lr": 3e-4,
    "ent_coef": 0.05,
    "min_ent_coef": 0.01,
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
    # 确保保存目录存在
    checkpoint_dir = "模型集_opponent"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # 1. 初始化环境
    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(config["num_envs"])])

    # 2. 初始化模型
    agent = Agent().to(config["device"])  # 正在学习的模型 (P1)
    opponent = Agent().to(config["device"])  # 固定的最强权重 (P2)

    # --- 新增：加载 P1 之前的训练进度以实现续训 ---
    if os.path.exists(config["save_path"]):
        try:
            agent.load_state_dict(torch.load(config["save_path"], map_location=config["device"], weights_only=False))
            print(f">>> 成功加载 P1 续训权重: {config['save_path']}")
        except Exception as e:
            print(f">>> 加载 P1 权重失败，从随机初始化开始: {e}")
    else:
        print(">>> 未找到 P1 历史权重，从头开始训练。")

    # 加载 P2 权重
    if os.path.exists(config["opponent_path"]):
        opponent.load_state_dict(torch.load(config["opponent_path"], map_location=config["device"], weights_only=False))
        opponent.eval()  # 锁定 P2
        print(f">>> 成功加载固定对手: {config['opponent_path']}")
    else:
        print(f">>> 警告: 未找到 {config['opponent_path']}，请检查路径。")
        return

    optimizer = optim.Adam(agent.parameters(), lr=config["lr"])
    writer = SummaryWriter(f"runs/vs_fixed_{time.strftime('%Y%m%d-%H%M%S')}")

    # 3. 缓冲区 (只存储 P1 的数据)
    obs_buf = torch.zeros((config["num_steps"], config["num_envs"], 48)).to(config["device"])
    act_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    logp_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    rew_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    done_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])
    val_buf = torch.zeros((config["num_steps"], config["num_envs"])).to(config["device"])

    obs_p1, _ = envs.reset()
    # P2 的 FrameStack 逻辑
    p2_deques = [deque(maxlen=4) for _ in range(config["num_envs"])]
    temp_env = SlimeSelfPlayEnv()
    temp_env.reset()
    init_p2_raw = temp_env._get_obs(2)
    for d in p2_deques: [d.append(init_p2_raw) for _ in range(4)]
    obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

    global_step = 0
    total_games = 0
    p1_wins = 0
    recent_wins = deque(maxlen=10)  # 新增：记录最近10局的胜负 (1为胜，0为负)
    last_save_step = 0  # 用于记录上次保存权重的步数

    while global_step < config["total_timesteps"]:
        agent.eval()

        # 熵系数衰减
        frac = max(0.0, 1.0 - (global_step / config["total_timesteps"]))
        current_ent_coef = config["min_ent_coef"] + (config["ent_coef"] - config["min_ent_coef"]) * frac

        # --- 采样阶段 ---
        for step in range(config["num_steps"]):
            global_step += config["num_envs"]

            t_obs_p1 = torch.from_numpy(obs_p1).float().to(config["device"])
            t_obs_p2 = torch.from_numpy(obs_p2).float().to(config["device"])

            with torch.no_grad():
                # P1 采样 (学习中)
                actions_p1, logp_p1, _, values_p1 = agent.get_action_and_value(t_obs_p1)
                # P2 决策 (最强权重使用 argmax)
                actions_p2 = torch.argmax(opponent.actor(t_obs_p2), dim=1)

            # 推进环境
            n_obs_p1, reward, term, trunc, infos = envs.step(
                np.stack([actions_p1.cpu().numpy(), actions_p2.cpu().numpy()], axis=1))

            # 记录数据
            for i in range(config["num_envs"]):
                if term[i] or trunc[i]:
                    total_games += 1
                    is_win = 1 if infos["p1_score"][i] > infos["p2_score"][i] else 0
                    p1_wins += is_win
                    recent_wins.append(is_win)  # 更新最近比赛记录

                    # 记录每局步数
                    if "episode_steps" in infos:
                        writer.add_scalar("Game/Episode_Steps", infos["episode_steps"][i], total_games)

            # 更新 P2 观测
            for i in range(config["num_envs"]): p2_deques[i].append(infos["p2_raw_obs"][i])
            n_obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

            # 填充缓冲区
            obs_buf[step], act_buf[step], logp_buf[step], val_buf[
                step] = t_obs_p1, actions_p1, logp_p1, values_p1.flatten()
            rew_buf[step] = torch.from_numpy(reward).to(config["device"])
            done_buf[step] = torch.from_numpy((term | trunc).astype(np.float32)).to(config["device"])

            obs_p1, obs_p2 = n_obs_p1, n_obs_p2

        # --- 计算优势 (GAE) ---
        with torch.no_grad():
            next_obs = torch.from_numpy(obs_p1).float().to(config["device"])
            _, _, _, next_val = agent.get_action_and_value(next_obs)
            adv = torch.zeros_like(rew_buf).to(config["device"])
            lastgae = 0
            for t in reversed(range(config["num_steps"])):
                nt = 1.0 - done_buf[t]
                nv = next_val.flatten() if t == config["num_steps"] - 1 else val_buf[t + 1]
                delta = rew_buf[t] + 0.99 * nv * nt - val_buf[t]
                adv[t] = lastgae = delta + 0.99 * 0.95 * nt * lastgae
            ret = adv + val_buf

        # --- PPO 更新阶段 ---
        agent.train()
        b_obs = obs_buf.reshape(-1, 48)
        b_logp = logp_buf.reshape(-1)
        b_act = act_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)

        indices = np.arange(config["num_steps"] * config["num_envs"])
        for _ in range(config["update_epochs"]):
            np.random.shuffle(indices)
            for s in range(0, len(indices), config["batch_size"]):
                mb = indices[s:s + config["batch_size"]]
                _, newlogp, ent, newv = agent.get_action_and_value(b_obs[mb], b_act[mb])

                ratio = (newlogp - b_logp[mb]).exp()
                m_adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)

                pg_loss1 = -m_adv * ratio
                pg_loss2 = -m_adv * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss = 0.5 * ((newv.flatten() - b_ret[mb]) ** 2).mean()
                loss = pg_loss - current_ent_coef * ent.mean() + v_loss * config["vf_coef"]

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"])
                optimizer.step()

        # --- 记录日志与保存逻辑 ---
        total_win_rate = p1_wins / total_games if total_games > 0 else 0
        recent_win_rate = sum(recent_wins) / len(recent_wins) if len(recent_wins) > 0 else 0

        writer.add_scalar("Train/Total_Win_Rate", total_win_rate, global_step)
        writer.add_scalar("Train/Recent_Win_Rate", recent_win_rate, global_step)

        # 修改后的输出格式：包含总胜率和最近10局胜率
        print(
            f"总步数: {global_step:7d} | 总胜率: {total_win_rate:.2%} | 最近10局胜率: {recent_win_rate:.2%} | 对局数: {total_games}")

        # 1. 保存最新权重 (覆盖)
        torch.save(agent.state_dict(), config["save_path"])

        # 2. 每 1,000,000 步保存一个独立权重到文件夹
        if (global_step - last_save_step) >= 1000000:
            ckpt_name = f"slime_ppo_{global_step // 1000000}M.pth"
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
            torch.save(agent.state_dict(), ckpt_path)
            last_save_step = global_step
            print(f">>> 已保存阶段性权重: {ckpt_path}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()