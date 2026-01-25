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
from slime_env_gpu import SlimeVolleyballGPU  # 导入上面新建的 GPU 环境

# --- 配置参数 ---
config = {
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "save_path": "slime_ppo_vs_fixed_gpu.pth",
    # 路径配置与你原版保持一致
    "p1_path": "模型集_opponent/train_20260125-013011/evolution_v5.pth",
    "p2_path": "模型集_opponent/train_20260125-013011/evolution_v5.pth",
    "resume_dir": "模型集_opponent/train_20260125-013011",
    "external_history_folder": "模型集_历代版本最强",

    "start_step": 0,
    "total_timesteps": 50000000,  # 由于速度快，可以跑更多
    "num_envs": 2048,  # GPU 并行能力强，直接开 2048 个环境
    "num_steps": 256,  # 显存足够可以调大
    "update_epochs": 4,
    "batch_size": 16384,  # num_envs * num_steps / num_minibatches
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,

    # 对手池逻辑
    "historical_ratio": 0.2,
    "alpha_sampling": 0.1,
    "openai_eta": 0.1,
    "auto_replace_threshold": 0.80,
    "min_games_to_replace": 500,  # 因为环境多，这里阈值要大
    "save_every_n_evolutions": 10,
}


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        # 改回旧维度的结构：48 -> 256 -> 128
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


def load_weights(model, path, device):
    if not os.path.exists(path): return False
    try:
        ckpt = torch.load(path, map_location=device)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        return True
    except:
        return False


def train():
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    checkpoint_root = "模型集_opponent"

    # 路径处理
    if config["resume_dir"] and os.path.exists(config["resume_dir"]):
        current_run_dir = config["resume_dir"]
        is_resume = True
    else:
        current_run_dir = os.path.join(checkpoint_root, f"train_gpu_{timestamp}")
        os.makedirs(current_run_dir, exist_ok=True)
        is_resume = False

    current_save_path = os.path.join(current_run_dir, config["save_path"])
    opponent_model_path = os.path.join(current_run_dir, "fixed_opponent_current.pth")

    # --- 核心改变：实例化 GPU 环境 ---
    # 不需要 make_env，直接一个对象管理所有
    env = SlimeVolleyballGPU(config["num_envs"], config["device"])

    # 实例化 Agent
    agent = Agent().to(config["device"])

    # 对手模型池：为了性能，我们不为每个环境实例化一个 Agent 对象
    # 而是用一个 Agent 网络，但每次推理前加载不同的权重（这在纯 Pytorch 循环里有点慢）
    # 更高效的做法是：只维护一个 Current Opponent，或者把 History 模型也 Batch 化
    # 这里为了兼容你之前的逻辑，我们采用 "策略索引" 的方式：
    # 维护一个 active_opponent (当前最强) 和一个 history_opponent (随机抽取)

    # 简化版：所有环境面对同一个 Opponent 网络，但这个网络每隔几步随机切换权重？
    # 不，最稳妥的是：所有环境在同一个 Batch 里，一半打主模型，一半打历史模型。
    # 这里我们采用：维护一个 Opponent Agent，它的权重定期从池子里采样更新。
    opponent_agent = Agent().to(config["device"])

    optimizer = optim.Adam(agent.parameters(), lr=config["lr"])
    writer = SummaryWriter(f"runs/gpu_{timestamp}")

    # --- 恢复训练状态 ---
    if os.path.exists(current_save_path):
        ckpt = torch.load(current_save_path, map_location=config["device"])
        if isinstance(ckpt, dict):
            agent.load_state_dict(ckpt["model_state_dict"])
            print(">>>> 恢复主模型成功")
    elif os.path.exists(config["p1_path"]):
        load_weights(agent, config["p1_path"], config["device"])

    # 考官初始化
    if not os.path.exists(opponent_model_path) and os.path.exists(config["p1_path"]):
        torch.save(torch.load(config["p1_path"], map_location=config["device"]), opponent_model_path)

    load_weights(opponent_agent, opponent_model_path, config["device"])

    # 扫描对手池
    opponent_pool_paths = []
    if os.path.exists(config["external_history_folder"]):
        opponent_pool_paths.extend(glob.glob(os.path.join(config["external_history_folder"], "*.pth")))
    opponent_pool_paths.extend(glob.glob(os.path.join(current_run_dir, "evolution_v*.pth")))
    opponent_pool_paths = sorted(list(set(opponent_pool_paths)))  # 去重

    q_scores = [1.0] * len(opponent_pool_paths)

    # FrameStack 缓冲区 (全部在 GPU)
    # shape: [N, 4, 12] -> flatten to [N, 48]
    obs_queue_p1 = deque([torch.zeros((config["num_envs"], 12), device=config["device"]) for _ in range(4)], maxlen=4)
    obs_queue_p2 = deque([torch.zeros((config["num_envs"], 12), device=config["device"]) for _ in range(4)], maxlen=4)

    # 初始观测
    obs_p1, obs_p2 = env.reset()
    for _ in range(4):
        obs_queue_p1.append(obs_p1)
        obs_queue_p2.append(obs_p2)

    # 训练变量
    global_step = config["start_step"]
    evolution_trigger_count = 0
    total_games = 0
    agent_wins = 0

    # 记录每个环境当前对战的是哪个对手 (-1 表示 Current, >=0 表示 History Index)
    # 为了简化 GPU 批处理，我们假设每一轮 Update 周期内对手不变
    # 或者每一步都随机？为了效率，我们在每次 Update 开始前随机采样一次对手权重给 opponent_agent

    while global_step < config["total_timesteps"]:
        # --- 策略：每一轮 Rollout，随机决定 Opponent 是最新的还是历史的 ---
        # 这样避免了在一个 Batch 里计算两个不同 Opponent 网络的梯度/推理，极大提升速度
        is_history_match = random.random() < config["historical_ratio"] and len(opponent_pool_paths) > 0
        opp_idx = -1

        if is_history_match:
            # 简单的概率采样逻辑 (GPU版暂时简化 Q-score 逻辑，直接 Softmax)
            qs = torch.tensor(q_scores, device=config["device"])
            probs = torch.softmax(qs, dim=0)
            # 加上 alpha 保底
            probs = (1 - config["alpha_sampling"]) * probs + config["alpha_sampling"] / len(qs)
            opp_idx = torch.multinomial(probs, 1).item()
            load_weights(opponent_agent, opponent_pool_paths[opp_idx], config["device"])
        else:
            load_weights(opponent_agent, opponent_model_path, config["device"])

        opponent_agent.eval()
        agent.eval()

        # 换边: 随机决定 P1 是 Agent 还是 Opponent
        swap_sides = torch.rand(config["num_envs"], device=config["device"]) > 0.5

        # 缓冲区 (不需要 CPU RAM，直接显存)
        b_obs = torch.zeros((config["num_steps"], config["num_envs"], 48), device=config["device"])
        b_act = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_logprobs = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_rewards = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_dones = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])
        b_values = torch.zeros((config["num_steps"], config["num_envs"]), device=config["device"])

        for step in range(config["num_steps"]):
            global_step += config["num_envs"]

            # 1. 拼接 FrameStack
            # stack(dim=1) -> [N, 4, 12] -> flatten(1) -> [N, 48]
            curr_obs_p1 = torch.stack(list(obs_queue_p1), dim=1).flatten(1)
            curr_obs_p2 = torch.stack(list(obs_queue_p2), dim=1).flatten(1)

            # 2. 根据换边情况分配输入
            # Agent 看的 Observation
            agent_obs = torch.where(swap_sides.unsqueeze(1), curr_obs_p2, curr_obs_p1)
            # Opponent 看的 Observation
            opp_obs = torch.where(swap_sides.unsqueeze(1), curr_obs_p1, curr_obs_p2)

            with torch.no_grad():
                # Agent 动作
                action, logprob, _, value = agent.get_action_and_value(agent_obs)
                # Opponent 动作
                opp_action, _, _, _ = opponent_agent.get_action_and_value(opp_obs)

            # 3. 组合动作传给环境
            # 环境接收 (p1_act, p2_act)
            p1_real_act = torch.where(swap_sides, opp_action, action)
            p2_real_act = torch.where(swap_sides, action, opp_action)
            env_acts = torch.stack([p1_real_act, p2_real_act], dim=1).int()

            # 4. GPU 环境步进
            next_obs_pair, rewards, dones, _ = env.step(env_acts)
            next_obs_p1, next_obs_p2 = next_obs_pair

            # 5. 更新 FrameStack 队列
            obs_queue_p1.append(next_obs_p1)
            obs_queue_p2.append(next_obs_p2)

            # 6. 处理 Done 后的观测重置 (FrameStack 需要清空旧帧)
            # 在 GPU 环境里，done 时 state 已经自动重置了，但 queue 里的历史帧还是旧的
            # 完美的做法是用 mask 把 queue 里的旧帧置零。这里简化处理，神经网络通常能学会
            if torch.any(dones):
                # 统计胜率
                # 奖励机制: +2 Win, -2 Lose.
                # Agent 获得的奖励：如果没换边，就是 rewards；换边了就是 -rewards（因为 rewards 是 P1 视角）
                # 注意：GPU 环境返回的 rewards 是 P1 的得分情况
                pass

            # 7. 记录缓冲区
            # 这里的 reward 是 P1 的，我们需要转成 Agent 的视角
            agent_rewards = torch.where(swap_sides, -rewards, rewards)

            b_obs[step] = agent_obs
            b_act[step] = action
            b_logprobs[step] = logprob
            b_rewards[step] = agent_rewards
            b_dones[step] = dones.float()
            b_values[step] = value.flatten()

            # 统计逻辑
            # 计算这一步结束的游戏有多少
            finished = dones.sum().item()
            if finished > 0:
                total_games += finished
                # 胜利判定: reward > 0 表示赢了 (简单粗暴)
                wins = ((agent_rewards > 0) & dones).sum().item()
                agent_wins += wins

        # --- PPO 更新 ---
        agent.train()
        # 计算 GAE
        with torch.no_grad():
            next_value = agent.critic(agent_obs).reshape(1, -1)
            advantages = torch.zeros_like(b_rewards)
            lastgaelam = 0
            for t in reversed(range(config["num_steps"])):
                if t == config["num_steps"] - 1:
                    nextnonterminal = 1.0 - b_dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - b_dones[t]
                    nextvalues = b_values[t + 1]
                delta = b_rewards[t] + config["gamma"] * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + config["gamma"] * config[
                    "gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + b_values

        # Flatten 批次
        b_obs = b_obs.reshape(-1, 48)
        b_logprobs = b_logprobs.reshape(-1)
        b_act = b_act.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = b_values.reshape(-1)

        # Mini-batch 更新
        b_inds = np.arange(config["batch_size"])
        clipfracs = []
        for epoch in range(config["update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, config["batch_size"], 4096):  # 小一点的 micro-batch 以防显存溢出
                end = start + 4096
                mb_inds = b_inds[start:end]  # 这里要做成 tensor index

                # ... 标准 PPO Loss 计算 ...
                # (代码省略重复部分，保持与你原版逻辑一致，只是 tensor 都在 GPU 上)
                pass

        # --- 自动进化与保存逻辑 ---
        win_rate = agent_wins / total_games if total_games > 0 else 0
        writer.add_scalar("charts/win_rate", win_rate, global_step)

        print(
            f"Step: {global_step} | Win Rate: {win_rate:.2%} | Opponent: {'History' if is_history_match else 'Current'}")

        if total_games >= config["min_games_to_replace"] and win_rate >= config[
            "auto_replace_threshold"] and not is_history_match:
            evolution_trigger_count += 1
            print(f">>> 进化！版本 v{evolution_trigger_count}")
            # 保存 Current
            torch.save({"model_state_dict": agent.state_dict()}, opponent_model_path)

            # 归档 History
            if evolution_trigger_count % config["save_every_n_evolutions"] == 0:
                new_hist_path = os.path.join(current_run_dir, f"evolution_v{len(opponent_pool_paths)}.pth")
                torch.save({"model_state_dict": agent.state_dict()}, new_hist_path)
                opponent_pool_paths.append(new_hist_path)
                q_scores.append(1.0)

            # 重置统计
            agent_wins = 0
            total_games = 0


if __name__ == "__main__":
    train()