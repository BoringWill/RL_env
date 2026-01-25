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

# --- 配置参数 ---
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "slime_ppo_vs_fixed.pth",
    "p1_path": "模型集_opponent/train_20260125-013011/evolution_v5.pth",
    "p2_path": "模型集_opponent/train_20260125-013011/evolution_v5.pth",
    "resume_dir": "模型集_opponent/train_20260125-013011",
    "external_history_folder": "模型集_历代版本最强",  # 新增：外部对手文件夹
    "start_step": 0,
    "p2_epsilon": 0.05,
    "auto_replace_threshold": 0.80,
    "min_games_to_replace": 30,
    "total_timesteps": 30000000,
    "num_envs": 32,
    "num_steps": 128,
    "update_epochs": 4,
    "batch_size": 1024,
    "lr": 2.5e-4,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "openai_eta": 0.1,
    "historical_ratio": 0.2,
    "alpha_sampling": 0.1,  # 保底采样权重
    "save_every_n_evolutions": 10,
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

    # --- 初始化变量 ---
    global_step = config["start_step"]
    evolution_count = 0
    evolution_trigger_count = 0
    total_games = 0
    agent_wins = 0
    opponent_pool_paths = []
    q_scores = []
    recent_wins = deque(maxlen=10)

    # --- 1. 加载学生模型及统计数据 ---
    if os.path.exists(current_save_path):
        checkpoint = torch.load(current_save_path, map_location=config["device"])
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            agent.load_state_dict(checkpoint["model_state_dict"])
            total_games = checkpoint.get("total_games", 0)
            agent_wins = checkpoint.get("agent_wins", 0)
            evolution_trigger_count = checkpoint.get("evolution_trigger_count", 0)
            print(f">>>> 成功恢复进度: 已进行 {total_games} 局，进化触发 {evolution_trigger_count} 次")
        else:
            agent.load_state_dict(checkpoint)
        print(f">>>> 已从 {current_save_path} 加载权重")
    elif os.path.exists(config["p1_path"]):
        agent.load_state_dict(torch.load(config["p1_path"], map_location=config["device"]))

    # --- 2. 恢复考官和对手池 (合并文件夹逻辑) ---
    # A. 确保考官模型存在
    if not os.path.exists(opponent_model_path):
        if os.path.exists(config["p1_path"]):
            torch.save(torch.load(config["p1_path"], map_location=config["device"]), opponent_model_path)
        else:
            print("警告: 找不到初始模型路径 p1_path")

    # B. 加载考官到环境对手
    if os.path.exists(opponent_model_path):
        opp_checkpoint = torch.load(opponent_model_path, map_location=config["device"])
        opp_state = opp_checkpoint["model_state_dict"] if isinstance(opp_checkpoint,
                                                                     dict) and "model_state_dict" in opp_checkpoint else opp_checkpoint
        for opp in opponents:
            opp.load_state_dict(opp_state)
            opp.eval()

    # C. 扫描所有对手来源
    # 来源1: 外部历史文件夹
    if os.path.exists(config["external_history_folder"]):
        ext_files = glob.glob(os.path.join(config["external_history_folder"], "*.pth"))
        for f in ext_files:
            # 统一路径格式，避免重复
            abs_f = os.path.abspath(f)
            if abs_f not in [os.path.abspath(p) for p in opponent_pool_paths]:
                opponent_pool_paths.append(f)
        print(f">>>> 已从外部文件夹加载 {len(ext_files)} 个对手模型")

    # 来源2: 本地运行产生的进化模型
    history_files = glob.glob(os.path.join(current_run_dir, "evolution_v*.pth"))
    history_files.sort(key=lambda x: int(os.path.basename(x).replace('evolution_v', '').replace('.pth', '')))
    for h_path in history_files:
        abs_h = os.path.abspath(h_path)
        if abs_h not in [os.path.abspath(p) for p in opponent_pool_paths]:
            opponent_pool_paths.append(h_path)

    # 初始化质量分
    q_scores = [1.0] * len(opponent_pool_paths)
    evolution_count = len(history_files)

    # --- 3. TensorBoard 路径处理 ---
    log_name = f"vs_fixed_{timestamp}" + ("_resume" if is_resume else "")
    writer = SummaryWriter(f"runs/{log_name}")

    optimizer = optim.Adam(agent.parameters(), lr=config["lr"])
    current_opp_paths = [opponent_model_path for _ in range(config["num_envs"])]
    current_opp_indices = [-1 for _ in range(config["num_envs"])]

    # 缓冲区初始化
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
    side_swapped = np.random.rand(config["num_envs"]) > 0.5

    while global_step < config["total_timesteps"]:
        agent.eval()
        for step in range(config["num_steps"]):
            global_step += config["num_envs"]
            t_obs_agent = torch.zeros((config["num_envs"], 48)).to(config["device"])
            t_obs_opp = torch.zeros((config["num_envs"], 48)).to(config["device"])

            for i in range(config["num_envs"]):
                if not side_swapped[i]:
                    t_obs_agent[i], t_obs_opp[i] = torch.from_numpy(obs_p1[i]), torch.from_numpy(obs_p2[i])
                else:
                    t_obs_agent[i], t_obs_opp[i] = torch.from_numpy(obs_p2[i]), torch.from_numpy(obs_p1[i])

            with torch.no_grad():
                actions_agent, logp_agent, _, values_agent = agent.get_action_and_value(t_obs_agent)
                actions_opp = torch.zeros(config["num_envs"], device=config["device"])
                for i in range(config["num_envs"]):
                    logits_opp = opponents[i].actor(t_obs_opp[i:i + 1])
                    actions_opp[i] = torch.distributions.Categorical(logits=logits_opp).sample()

            env_actions = np.zeros((config["num_envs"], 2), dtype=np.int32)
            for i in range(config["num_envs"]):
                if not side_swapped[i]:
                    env_actions[i] = [actions_agent[i].item(), actions_opp[i].item()]
                else:
                    env_actions[i] = [actions_opp[i].item(), actions_agent[i].item()]

            n_obs_p1, reward, term, trunc, infos = envs.step(env_actions)

            for i in range(config["num_envs"]):
                rew_buf[step][i] = reward[i] if not side_swapped[i] else -reward[i]

                if term[i] or trunc[i]:
                    total_games += 1
                    is_agent_win = 1 if (not side_swapped[i] and infos["p1_score"][i] > infos["p2_score"][i]) or \
                                        (side_swapped[i] and infos["p2_score"][i] > infos["p1_score"][i]) else 0
                    agent_wins += is_agent_win
                    recent_wins.append(is_agent_win)

                    # 质量分逻辑
                    if current_opp_indices[i] != -1 and is_agent_win:
                        qs = np.array(q_scores)
                        raw_probs = np.exp(qs - np.max(qs)) / np.sum(np.exp(qs - np.max(qs)))
                        actual_prob = (1 - config["alpha_sampling"]) * raw_probs[current_opp_indices[i]] + \
                                      (config["alpha_sampling"] / len(opponent_pool_paths))
                        q_scores[current_opp_indices[i]] -= config["openai_eta"] / (
                                len(opponent_pool_paths) * actual_prob)

                    if len(opponent_pool_paths) > 0 and random.random() < config["historical_ratio"]:
                        qs = np.array(q_scores)
                        softmax_probs = np.exp(qs - np.max(qs)) / np.sum(np.exp(qs - np.max(qs)))
                        uniform_probs = np.ones_like(softmax_probs) / len(opponent_pool_paths)
                        final_probs = (1 - config["alpha_sampling"]) * softmax_probs + config[
                            "alpha_sampling"] * uniform_probs

                        idx = np.random.choice(len(opponent_pool_paths), p=final_probs)
                        path, current_opp_indices[i] = opponent_pool_paths[idx], idx
                    else:
                        path, current_opp_indices[i] = opponent_model_path, -1

                    # 加载对手
                    opp_ckpt = torch.load(path, map_location=config["device"])
                    opp_state = opp_ckpt["model_state_dict"] if isinstance(opp_ckpt,
                                                                           dict) and "model_state_dict" in opp_ckpt else opp_ckpt
                    opponents[i].load_state_dict(opp_state)
                    opponents[i].eval()

                    current_opp_paths[i], side_swapped[i] = path, np.random.rand() > 0.5
                    if "episode_steps" in infos: writer.add_scalar("Game/Episode_Steps", infos["episode_steps"][i],
                                                                   total_games)
                    p2_deques[i].clear()
                    for _ in range(4): p2_deques[i].append(infos["p2_raw_obs"][i])
                else:
                    p2_deques[i].append(infos["p2_raw_obs"][i])

            obs_buf[step], act_buf[step], logp_buf[step], val_buf[
                step] = t_obs_agent, actions_agent, logp_agent, values_agent.flatten()
            done_buf[step] = torch.from_numpy((term | trunc).astype(np.float32)).to(config["device"])
            obs_p1, obs_p2 = n_obs_p1, np.array([np.concatenate(list(d), axis=0) for d in p2_deques])

        # PPO 更新逻辑
        with torch.no_grad():
            t_next_obs = torch.zeros((config["num_envs"], 48)).to(config["device"])
            for i in range(config["num_envs"]): t_next_obs[i] = torch.from_numpy(
                obs_p2[i] if side_swapped[i] else obs_p1[i]).float()
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
                loss = pg_loss - config["ent_coef"] * ent.mean() + v_loss * config["vf_coef"]
                optimizer.zero_grad();
                loss.backward();
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5);
                optimizer.step()

        # 监控记录
        total_agent_win_rate = agent_wins / total_games if total_games > 0 else 0
        writer.add_scalar("Train/Total_Win_Rate", total_agent_win_rate, global_step)
        for idx, score in enumerate(q_scores): writer.add_scalar(f"Opponent_Scores/v{idx + 1}", score, global_step)

        # 持久化保存
        checkpoint_data = {
            "model_state_dict": agent.state_dict(),
            "total_games": total_games,
            "agent_wins": agent_wins,
            "evolution_trigger_count": evolution_trigger_count
        }

        if total_games >= config["min_games_to_replace"] and total_agent_win_rate >= config["auto_replace_threshold"]:
            evolution_trigger_count += 1
            print(f"\n[进化触发] 第 {evolution_trigger_count} 次胜率达标！")
            torch.save(checkpoint_data, opponent_model_path)
            if evolution_trigger_count % config["save_every_n_evolutions"] == 0:
                evolution_count += 1
                new_v_path = os.path.join(current_run_dir, f"evolution_v{evolution_count}.pth")
                torch.save(checkpoint_data, new_v_path)
                opponent_pool_paths.append(new_v_path)
                q_scores.append(max(q_scores) if q_scores else 1.0)
            total_games, agent_wins = 0, 0
            recent_wins.clear()

        torch.save(checkpoint_data, current_save_path)
        q_info = f" | 池分均值: {np.mean(q_scores):.2f}" if q_scores else ""
        print(
            f"步数: {global_step:7d} | 当前周期局数: {total_games:4d} | 当前总胜率: {total_agent_win_rate:.2%}{q_info} | 环境0的对手: {os.path.basename(current_opp_paths[0])}")

    envs.close();
    writer.close()


if __name__ == "__main__":
    train()