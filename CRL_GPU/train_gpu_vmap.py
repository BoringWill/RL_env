import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, glob, time, random
from torch.utils.tensorboard import SummaryWriter
from collections import deque
from slime_env_gpu import SlimeVolleyballGPU

# --- 1. 配置参数 ---
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "slime_ppo_vs_fixed_gpu.pth",
    "resume_dir": "模型集_opponent/1",
    "external_history_folder": "模型集_历代版本最强",

    # --- 初始考官设置 ---
    # 如果你想从某个高手模型开始练，请把路径写在这里
    "initial_opponent_path": "模型集_历代版本最强/12.pth",

    "total_timesteps": 500_000_000,
    "num_envs": 2048,
    "num_steps": 256,
    "update_epochs": 4,
    "batch_size": 32768,
    "lr": 1e-3,  # 提高学习率以适配大 Batch
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.05,  # 提高熵系数，强制增加探索
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    "historical_ratio": 0.2,
    "openai_eta": 0.1,
    "alpha_sampling": 0.05,
    "auto_replace_threshold": 0.80,
    "min_games_to_replace": 1000,
    "save_every_n_evolutions": 5,
    "save_interval_steps": 50_000_000,
}


# --- 2. 模型结构 ---
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


def read_state_dict_from_disk(path, device):
    if not os.path.exists(path): return None
    try:
        ckpt = torch.load(path, map_location=device)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        return sd
    except:
        return None


# --- 3. 训练程序 ---
def train():
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    os.makedirs(config["resume_dir"], exist_ok=True)
    if not os.path.exists(config["external_history_folder"]):
        os.makedirs(config["external_history_folder"], exist_ok=True)

    writer = SummaryWriter(log_dir=f"runs/vs_fixed_gpu_{timestamp}")

    current_save_path = os.path.join(config["resume_dir"], config["save_path"])
    opponent_model_path = os.path.join(config["resume_dir"], "fixed_opponent_current.pth")

    env = SlimeVolleyballGPU(config["num_envs"], config["device"])
    agent = Agent().to(config["device"])
    optimizer = optim.Adam(agent.parameters(), lr=config["lr"], eps=1e-5)
    opp_agent = Agent().to(config["device"])

    # 扫描对手池
    opponent_pool_paths = []
    local_files = sorted(glob.glob(os.path.join(config["resume_dir"], "evolution_v*.pth")))
    for f in local_files: opponent_pool_paths.append(os.path.abspath(f))
    ext_files = glob.glob(os.path.join(config["external_history_folder"], "*.pth"))
    for f in ext_files:
        abs_f = os.path.abspath(f)
        if abs_f not in opponent_pool_paths: opponent_pool_paths.append(abs_f)

    q_scores = [1.0] * len(opponent_pool_paths)
    current_opp_indices = np.full(config["num_envs"], -1)

    # --- A. 加载 P1 (学生) ---
    if os.path.exists(current_save_path):
        sd = read_state_dict_from_disk(current_save_path, config["device"])
        if sd is not None:
            agent.load_state_dict(sd)
            print(f">>>> 已恢复 P1 训练断点: {current_save_path}")

    # --- B. 核心：加载初始考官权重 ---
    if not os.path.exists(opponent_model_path):
        if os.path.exists(config["initial_opponent_path"]):
            sd_init = read_state_dict_from_disk(config["initial_opponent_path"], config["device"])
            if sd_init is not None:
                torch.save({"model_state_dict": sd_init}, opponent_model_path)
                print(f">>>> 初始考官已就位，来源: {config['initial_opponent_path']}")
            else:
                torch.save({"model_state_dict": agent.state_dict()}, opponent_model_path)
                print(">>>> 读取指定初始考官失败，使用 P1 随机权重作为考官")
        else:
            torch.save({"model_state_dict": agent.state_dict()}, opponent_model_path)
            print(">>>> 未找到指定初始考官，使用 P1 随机权重作为起点")

    # FrameStack
    obs_queue_p1 = deque([torch.zeros((config["num_envs"], 12), device=config["device"]) for _ in range(4)], maxlen=4)
    obs_queue_p2 = deque([torch.zeros((config["num_envs"], 12), device=config["device"]) for _ in range(4)], maxlen=4)
    obs_p1, obs_p2 = env.reset()
    for _ in range(4):
        obs_queue_p1.append(obs_p1);
        obs_queue_p2.append(obs_p2)

    global_step = 0
    last_periodic_save = 0
    agent_wins, total_games = 0, 0
    evolution_count = len(local_files)
    model_cache = {}

    print(f">>> 对手池已就绪，总计 {len(opponent_pool_paths)} 个模型。")

    while global_step < config["total_timesteps"]:
        # 1. 确定本轮对手分配
        is_history_mask = np.random.rand(config["num_envs"]) < config["historical_ratio"]
        if len(opponent_pool_paths) > 0:
            qs = np.array(q_scores)
            softmax_probs = np.exp(qs - np.max(qs)) / (np.sum(np.exp(qs - np.max(qs))) + 1e-8)
            final_probs = (1 - config["alpha_sampling"]) * softmax_probs + config["alpha_sampling"] / len(
                opponent_pool_paths)
            rand_indices = np.random.choice(len(opponent_pool_paths), size=config["num_envs"], p=final_probs)
            current_opp_indices = np.where(is_history_mask, rand_indices, -1)
        else:
            current_opp_indices.fill(-1)

        # 2. 预加载对手权重到 GPU 缓存
        unique_opps = np.unique(current_opp_indices)
        for idx in unique_opps:
            path = opponent_model_path if idx == -1 else opponent_pool_paths[idx]
            if path not in model_cache:
                sd = read_state_dict_from_disk(path, config["device"])
                model_cache[path] = sd if sd is not None else agent.state_dict()

        # Rollout 数据容器
        b_obs = torch.zeros((config["num_steps"], config["num_envs"], 48), device=config["device"])
        b_actions = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_logprobs = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_rewards = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_dones = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_values = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        swap_sides = torch.rand(config["num_envs"], device=config["device"]) > 0.5

        # 3. Rollout 循环
        for step in range(config["num_steps"]):
            agent.eval()
            curr_obs_p1 = torch.stack(list(obs_queue_p1), dim=1).flatten(1)
            curr_obs_p2 = torch.stack(list(obs_queue_p2), dim=1).flatten(1)
            agent_obs = torch.where(swap_sides.unsqueeze(1), curr_obs_p2, curr_obs_p1)
            opp_obs = torch.where(swap_sides.unsqueeze(1), curr_obs_p1, curr_obs_p2)

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(agent_obs)
                opp_action = torch.zeros(config["num_envs"], device=config["device"], dtype=torch.long)

                for opp_idx in unique_opps:
                    env_mask = torch.from_numpy(current_opp_indices == opp_idx).to(config["device"])
                    if not env_mask.any(): continue
                    path = opponent_model_path if opp_idx == -1 else opponent_pool_paths[opp_idx]
                    opp_agent.load_state_dict(model_cache[path])
                    logits_opp = opp_agent.actor(opp_obs[env_mask])
                    opp_action[env_mask] = torch.distributions.Categorical(logits=logits_opp).sample()

            p1_act, p2_act = torch.where(swap_sides, opp_action, action), torch.where(swap_sides, action, opp_action)
            next_obs_pair, rewards, dones, _ = env.step(torch.stack([p1_act, p2_act], dim=1).int())

            # 更新 FrameStack
            mask = (~dones).float().unsqueeze(1)
            for i in range(4):
                obs_queue_p1[i] *= mask;
                obs_queue_p2[i] *= mask
            obs_queue_p1.append(next_obs_pair[0]);
            obs_queue_p2.append(next_obs_pair[1])

            agent_rew = torch.where(swap_sides, -rewards, rewards)

            if dones.any():
                done_indices = torch.where(dones)[0].cpu().numpy()
                for i in done_indices:
                    total_games += 1
                    is_win = (agent_rew[i] > 0).item()
                    if is_win: agent_wins += 1
                    opp_idx = current_opp_indices[i]
                    if opp_idx != -1 and is_win and len(opponent_pool_paths) > 0:
                        p_idx = final_probs[opp_idx]
                        q_scores[opp_idx] -= config["openai_eta"] / (len(opponent_pool_paths) * p_idx + 1e-6)

            b_obs[step], b_actions[step], b_logprobs[step] = agent_obs, action, logprob
            b_rewards[step], b_dones[step], b_values[step] = agent_rew, dones.float(), value.flatten()

        # 4. PPO 更新
        agent.train()
        with torch.no_grad():
            last_val = agent.critic(agent_obs).reshape(1, -1)
            advantages = torch.zeros_like(b_rewards)
            lastgaelam = 0
            for t in reversed(range(config["num_steps"])):
                nt = 1.0 - b_dones[t]
                nv = last_val if t == config["num_steps"] - 1 else b_values[t + 1]
                delta = b_rewards[t] + config["gamma"] * nv * nt - b_values[t]
                advantages[t] = lastgaelam = delta + config["gamma"] * config["gae_lambda"] * nt * lastgaelam
            returns = advantages + b_values

        f_obs, f_logp, f_act, f_adv, f_ret = b_obs.reshape(-1, 48), b_logprobs.reshape(-1), b_actions.reshape(
            -1), advantages.reshape(-1), returns.reshape(-1)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        inds = np.arange(f_obs.shape[0])
        for _ in range(config["update_epochs"]):
            np.random.shuffle(inds)
            for s in range(0, f_obs.shape[0], 4096):
                mb = inds[s:s + 4096]
                _, new_lp, ent, new_v = agent.get_action_and_value(f_obs[mb], f_act[mb])
                ratio = (new_lp - f_logp[mb]).exp()
                pg_loss = torch.max(-f_adv[mb] * ratio, -f_adv[mb] * torch.clamp(ratio, 0.8, 1.2)).mean()
                v_loss = 0.5 * ((new_v.flatten() - f_ret[mb]) ** 2).mean()
                loss = pg_loss - config["ent_coef"] * ent.mean() + v_loss * config["vf_coef"]
                optimizer.zero_grad();
                loss.backward();
                nn.utils.clip_grad_norm_(agent.parameters(), config["max_grad_norm"]);
                optimizer.step()

        # 5. 监控与保存
        global_step += config["num_envs"] * config["num_steps"]
        win_rate = agent_wins / total_games if total_games > 0 else 0
        writer.add_scalar("Train/Win_Rate", win_rate, global_step)
        for idx, score in enumerate(q_scores):
            label = os.path.basename(opponent_pool_paths[idx]).replace('.pth', '')
            writer.add_scalar(f"Opponent_Scores/{label}", score, global_step)

        # 进化逻辑
        if total_games >= config["min_games_to_replace"] and win_rate >= config["auto_replace_threshold"]:
            print(f">>> [进化] 胜率 {win_rate:.2%}, 更新考官为当前最强 Agent")
            torch.save({"model_state_dict": agent.state_dict()}, opponent_model_path)
            if opponent_model_path in model_cache: del model_cache[opponent_model_path]

            if evolution_count % config["save_every_n_evolutions"] == 0:
                v_path = os.path.join(config["resume_dir"], f"evolution_v{evolution_count}.pth")
                torch.save({"model_state_dict": agent.state_dict()}, v_path)
                abs_v = os.path.abspath(v_path)
                if abs_v not in opponent_pool_paths:
                    opponent_pool_paths.append(abs_v);
                    q_scores.append(max(q_scores) if q_scores else 1.0)
            evolution_count += 1
            agent_wins, total_games = 0, 0

        if global_step // config["save_interval_steps"] > last_periodic_save:
            periodic_path = os.path.join(config["resume_dir"], f"model_step_{global_step // 1000000}M.pth")
            torch.save({"model_state_dict": agent.state_dict()}, periodic_path)
            last_periodic_save = global_step // config["save_interval_steps"]

        torch.save({"model_state_dict": agent.state_dict()}, current_save_path)
        print(f"步数: {global_step:8d} | 胜率: {win_rate:.2%} | 池总数: {len(opponent_pool_paths)}")


if __name__ == "__main__":
    train()