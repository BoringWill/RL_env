import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from slime_env import SlimeSelfPlayEnv, FrameStack
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import time
import os  # 新增：用于处理文件夹和路径

# --- 配置参数 ---
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "slime_ppo_gpu.pth",
    "total_timesteps": 50000000,
    "num_envs": 32,
    "num_steps": 2048,
    "update_epochs": 10,
    "batch_size": 8192,
    "lr": 3e-4,
    "ent_coef": 0.05,  # 初始熵系数
    "min_ent_coef": 0.01,  # 最小熵系数目标
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "model_set_dir": "模型集_selfplay",  # 新增：模型保存文件夹
    "save_interval": 1000000,           # 新增：保存间隔步数
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
    # 新增：创建模型集文件夹
    if not os.path.exists(config["model_set_dir"]):
        os.makedirs(config["model_set_dir"])

    envs = gym.vector.AsyncVectorEnv([make_env() for _ in range(config["num_envs"])])
    agent = Agent().to(config["device"])

    try:
        agent.load_state_dict(torch.load(config["save_path"], map_location=config["device"]))
        print(">>> 已加载权重，继续训练...")
    except:
        print(">>> 从零启动...")

    optimizer = optim.Adam(agent.parameters(), lr=config["lr"])

    log_dir = f"runs/slime_ppo_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir)
    print(f">>> 日志将保存至: {log_dir}")

    # 缓冲区
    obs_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2, 48)).to(config["device"])
    act_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    logp_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    rew_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    done_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])
    val_buf = torch.zeros((config["num_steps"], config["num_envs"] * 2)).to(config["device"])

    obs_p1, _ = envs.reset()
    p2_deques = [deque(maxlen=4) for _ in range(config["num_envs"])]

    # 获取初始状态用于填充
    temp_env = SlimeSelfPlayEnv()
    temp_env.reset()
    init_p2_raw = temp_env._get_obs(2)
    for d in p2_deques: [d.append(init_p2_raw) for _ in range(4)]
    obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

    global_step = 0
    last_save_step = 0 # 新增：记录上次保存时的步数

    total_games_finished = 0
    total_p1_wins = 0
    total_p2_wins = 0

    last_game_p1_score = 0
    last_game_p2_score = 0

    # --- 新增：内部记录每个环境当前对局步数的计数器 ---
    env_steps = np.zeros(config["num_envs"])

    while global_step < config["total_timesteps"]:
        agent.eval()

        # 计算当前的熵系数（线性衰减）
        frac = 1.0 - (global_step / config["total_timesteps"])
        frac = max(0.0, frac)
        current_ent_coef = config["min_ent_coef"] + (config["ent_coef"] - config["min_ent_coef"]) * frac

        for step in range(config["num_steps"]):
            global_step += config["num_envs"]
            t_obs = torch.from_numpy(np.concatenate([obs_p1, obs_p2])).float().to(config["device"])
            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(t_obs)

            p1_acts, p2_acts = actions[:config["num_envs"]].cpu().numpy(), actions[config["num_envs"]:].cpu().numpy()
            n_obs_p1, reward, term, trunc, infos = envs.step(np.stack([p1_acts, p2_acts], axis=1))

            # --- 步数自增 ---
            env_steps += 1

            for i in range(config["num_envs"]):
                # term 代表比赛分出了胜负（10分），trunc 代表超时截断
                if term[i] or trunc[i]:
                    last_game_p1_score = infos["p1_score"][i]
                    last_game_p2_score = infos["p2_score"][i]

                    # --- 修改点：记录该对局步数到 TensorBoard，并重置计数器 ---
                    writer.add_scalar("Game/Episode_Length", env_steps[i], total_games_finished)
                    env_steps[i] = 0

                    total_games_finished += 1
                    writer.add_scalar("Game_Score/P1", last_game_p1_score, total_games_finished)
                    writer.add_scalar("Game_Score/P2", last_game_p2_score, total_games_finished)

                    if term[i]:  # 只有正常结束才计入胜场统计
                        if infos["p1_score"][i] > infos["p2_score"][i]:
                            total_p1_wins += 1
                        else:
                            total_p2_wins += 1

            for i in range(config["num_envs"]): p2_deques[i].append(infos["p2_raw_obs"][i])
            n_obs_p2 = np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

            obs_buf[step], act_buf[step], logp_buf[step], val_buf[step] = t_obs, actions, logprobs, values.flatten()
            rew_buf[step, :config["num_envs"]] = torch.from_numpy(reward).to(config["device"])
            rew_buf[step, config["num_envs"]:] = torch.from_numpy(-reward).to(config["device"])
            done_buf[step] = torch.from_numpy((term | trunc).astype(np.float32)).repeat(2).to(config["device"])

            obs_p1, obs_p2 = n_obs_p1, n_obs_p2

        global_win_rate = total_p1_wins / total_games_finished if total_games_finished > 0 else 0.5

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

        agent.train()
        b_obs, b_logp, b_act, b_adv, b_ret = obs_buf.reshape(-1, 48), logp_buf.reshape(-1), act_buf.reshape(
            -1), adv.reshape(-1), ret.reshape(-1)
        indices = np.arange(config["num_steps"] * config["num_envs"] * 2)

        pg_losses, v_losses = [], []
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
                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())

        writer.add_scalar("GameWins/Global_Win_Rate", global_win_rate, global_step)
        writer.add_scalar("Loss/Policy", np.mean(pg_losses), global_step)
        writer.add_scalar("Loss/Value", np.mean(v_losses), global_step)
        writer.add_scalar("Train/Entropy_Coef", current_ent_coef, global_step)

        print(
            f"步数: {global_step:7d} | 熵系数: {current_ent_coef:.4f} | 总局数: {total_games_finished} | 最近战报: P1 {last_game_p1_score}:{last_game_p2_score} P2 | "
            f"P1总胜场: {total_p1_wins} | 总胜率: {global_win_rate:.2%}")

        torch.save(agent.state_dict(), config["save_path"])

        # 新增：每 1000000 步保存一个阶段性模型
        if global_step - last_save_step >= config["save_interval"]:
            save_name = f"slime_ppo_{global_step // 1000000}M.pth"
            save_path = os.path.join(config["model_set_dir"], save_name)
            torch.save(agent.state_dict(), save_path)
            print(f">>> 阶段性模型已保存至: {save_path}")
            last_save_step = (global_step // config["save_interval"]) * config["save_interval"]

    envs.close()
    writer.close()


if __name__ == "__main__":
    train()