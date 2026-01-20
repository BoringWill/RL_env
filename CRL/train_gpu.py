import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pygame
from slime_env import SlimeSelfPlayEnv
from torch.utils.tensorboard import SummaryWriter
import os

# --- 1. 神经网络模型定义 ---
class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4)
        )

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

# --- 2. 超参数配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-3
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_COEF = 0.2
ENT_COEF = 0.05
VF_COEF = 0.5
BATCH_SIZE = 256
UPDATE_EPOCHS = 4
NUM_STEPS = 2048
TOTAL_TIMESTEPS = 1000000
SAVE_PATH = "slime_ppo_gpu.pth"
# --- 新增：日志保存路径 ---
LOG_DIR = "runs/slime_ppo_experiment"

# --- 3. 训练函数 ---
def train():
    env = SlimeSelfPlayEnv(render_mode=None)
    agent = Agent().to(DEVICE)
    optimizer = optim.Adam(agent.parameters(), lr=LEARNING_RATE, eps=1e-5)

    # --- 新增：初始化 SummaryWriter ---
    writer = SummaryWriter(LOG_DIR)

    obs_buf = torch.zeros((NUM_STEPS * 2, 12)).to(DEVICE)
    act_buf = torch.zeros(NUM_STEPS * 2).to(DEVICE)
    logp_buf = torch.zeros(NUM_STEPS * 2).to(DEVICE)
    rew_buf = torch.zeros(NUM_STEPS * 2).to(DEVICE)
    done_buf = torch.zeros(NUM_STEPS * 2).to(DEVICE)
    val_buf = torch.zeros(NUM_STEPS * 2).to(DEVICE)

    obs_p1, _ = env.reset()
    obs_p2 = env._get_obs(2)

    global_step = 0
    episode_rewards = []
    current_ep_reward = 0

    print(f">>> 开始训练。设备: {DEVICE} | 目标步数: {TOTAL_TIMESTEPS}")

    while global_step < TOTAL_TIMESTEPS:
        agent.eval()
        for step in range(NUM_STEPS):
            global_step += 1
            t_obs = torch.from_numpy(np.stack([obs_p1, obs_p2])).float().to(DEVICE)

            with torch.no_grad():
                actions, logprobs, _, values = agent.get_action_and_value(t_obs)

            n_obs_p1, n_obs_p2, reward, done, _ = env.step(actions[0].item(), actions[1].item())
            current_ep_reward += reward

            idx1, idx2 = step, step + NUM_STEPS
            obs_buf[idx1], obs_buf[idx2] = t_obs[0], t_obs[1]
            act_buf[idx1], act_buf[idx2] = actions[0], actions[1]
            logp_buf[idx1], logp_buf[idx2] = logprobs[0], logprobs[1]
            val_buf[idx1], val_buf[idx2] = values[0].flatten(), values[1].flatten()
            rew_buf[idx1], rew_buf[idx2] = torch.tensor(reward).to(DEVICE), torch.tensor(-reward).to(DEVICE)
            done_buf[idx1], done_buf[idx2] = torch.tensor(done).to(DEVICE), torch.tensor(done).to(DEVICE)

            obs_p1, obs_p2 = n_obs_p1, n_obs_p2

            if done:
                episode_rewards.append(current_ep_reward)
                # --- 新增：记录单局奖励到 TensorBoard ---
                writer.add_scalar("charts/episodic_reward", current_ep_reward, global_step)
                current_ep_reward = 0
                obs_p1, _ = env.reset()
                obs_p2 = env._get_obs(2)

        with torch.no_grad():
            advantages = torch.zeros_like(rew_buf).to(DEVICE)
            for offset in [0, NUM_STEPS]:
                lastgaelam = 0
                for t in reversed(range(NUM_STEPS)):
                    idx = offset + t
                    next_non_terminal = 1.0 - done_buf[idx]
                    next_val = val_buf[idx + 1] if (t < NUM_STEPS - 1) else 0
                    delta = rew_buf[idx] + GAMMA * next_val * next_non_terminal - val_buf[idx]
                    advantages[idx] = lastgaelam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * lastgaelam
            returns = advantages + val_buf

        agent.train()
        b_inds = np.arange(NUM_STEPS * 2)
        pg_losses, v_losses, ents = [], [], []

        for epoch in range(UPDATE_EPOCHS):
            np.random.shuffle(b_inds)
            for start in range(0, NUM_STEPS * 2, BATCH_SIZE):
                end = start + BATCH_SIZE
                mb_idx = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(obs_buf[mb_idx], act_buf[mb_idx])
                ratio = (newlogprob - logp_buf[mb_idx]).exp()
                mb_adv = advantages[mb_idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)).mean()
                v_loss = 0.5 * ((newvalue.flatten() - returns[mb_idx]) ** 2).mean()
                loss = pg_loss - ENT_COEF * entropy.mean() + v_loss * VF_COEF

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                pg_losses.append(pg_loss.item())
                v_losses.append(v_loss.item())
                ents.append(entropy.mean().item())

        # --- 新增：记录训练损失和熵到 TensorBoard ---
        writer.add_scalar("losses/policy_loss", np.mean(pg_losses), global_step)
        writer.add_scalar("losses/value_loss", np.mean(v_losses), global_step)
        writer.add_scalar("losses/entropy", np.mean(ents), global_step)
        writer.add_scalar("charts/learning_rate", LEARNING_RATE, global_step)

        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) > 0 else 0
        print(f"步数: {global_step:7d} | 奖励: {avg_reward:6.2f} | P_Loss: {np.mean(pg_losses):.4f} | V_Loss: {np.mean(v_losses):.4f} | Ent: {np.mean(ents):.4f} | 进度: {100 * global_step / TOTAL_TIMESTEPS:3.1f}%")

        torch.save(agent.state_dict(), SAVE_PATH)

    # --- 新增：训练结束关闭 writer ---
    writer.close()

if __name__ == "__main__":
    train()