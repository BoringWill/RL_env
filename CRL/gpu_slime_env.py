import torch
from constants import *


class SlimeVolleyballGPU:
    def __init__(self, num_envs, device="cuda"):
        self.num_envs = num_envs
        self.device = device

        # 状态矩阵 (num_envs, 12): [p1_x, p1_y, p1_vx, p1_vy, p2_x, p2_y, p2_vx, p2_vy, b_x, b_y, b_vx, b_vy]
        self.state = torch.zeros((num_envs, 12), device=device)
        self.last_touch = torch.zeros(num_envs, device=device)  # 1: P1, -1: P2

        self.WIDTH_T = torch.tensor(WIDTH, device=device).float()
        self.HALF_WIDTH = self.WIDTH_T / 2
        self.GROUND_Y_T = torch.tensor(GROUND_Y, device=device).float()

    def reset(self, indices=None):
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)

        self.state[indices] = 0.0
        self.state[indices, 0] = 200.0
        self.state[indices, 1] = GROUND_Y - SLIME_RADIUS
        self.state[indices, 4] = 800.0
        self.state[indices, 5] = GROUND_Y - SLIME_RADIUS
        self.state[indices, 8] = torch.where(torch.rand(len(indices), device=self.device) > 0.5, 200.0, 800.0)
        self.state[indices, 9] = 200.0
        self.last_touch[indices] = 0
        return self.get_obs(indices)

    def step(self, act_p1, act_p2):
        # 1. 物理模拟
        self._apply_player_physics(0, act_p1)
        self._apply_player_physics(4, act_p2)
        self.state[:, 8:10] += self.state[:, 10:12]  # ball pos += vel
        self.state[:, 11] += GRAVITY  # ball gravity

        # 2. 碰撞判定与得分
        rewards, dones = self._handle_collisions()
        obs_p1, obs_p2 = self.get_obs()
        return (obs_p1, obs_p2), rewards, dones

    def _apply_player_physics(self, offset, actions):
        px, py, pvx, pvy = self.state[:, offset], self.state[:, offset + 1], self.state[:, offset + 2], self.state[:,
                                                                                                        offset + 3]
        target_vx = torch.zeros_like(pvx)
        target_vx = torch.where(actions == 1, torch.tensor(-PLAYER_SPEED, device=self.device), target_vx)
        target_vx = torch.where(actions == 2, torch.tensor(PLAYER_SPEED, device=self.device), target_vx)

        on_ground = py >= (GROUND_Y - SLIME_RADIUS - 1.0)
        pvy = torch.where(on_ground & (actions == 3), torch.tensor(JUMP_POWER, device=self.device), pvy)
        pvy += GRAVITY
        px += target_vx
        py += pvy

        if offset == 0:
            px = torch.clamp(px, SLIME_RADIUS, self.HALF_WIDTH - SLIME_RADIUS)
        else:
            px = torch.clamp(px, self.HALF_WIDTH + SLIME_RADIUS, self.WIDTH_T - SLIME_RADIUS)

        touch_g = py >= (GROUND_Y - SLIME_RADIUS)
        py = torch.where(touch_g, self.GROUND_Y_T - SLIME_RADIUS, py)
        pvy = torch.where(touch_g, torch.tensor(0.0, device=self.device), pvy)
        self.state[:, offset:offset + 4] = torch.stack([px, py, target_vx, pvy], dim=1)

    def _handle_collisions(self):
        bx, by = self.state[:, 8], self.state[:, 9]
        # 墙壁反弹
        hit_wall = (bx <= BALL_RADIUS) | (bx >= WIDTH - BALL_RADIUS)
        self.state[hit_wall, 10] *= -1.0

        # 史莱姆碰撞
        for offset in [0, 4]:
            dx, dy = bx - self.state[:, offset], by - self.state[:, offset + 1]
            dist = torch.sqrt(dx ** 2 + dy ** 2)
            coll = (dist < (BALL_RADIUS + SLIME_RADIUS)) & (by < self.state[:, offset + 1] + 10)

            if coll.any():
                # --- 关键修改：只对碰撞的环境进行计算 ---
                # 提取发生碰撞的子集
                coll_dx = dx[coll]
                coll_dy = dy[coll]
                coll_pvx = self.state[coll, offset + 2]

                angle = torch.atan2(coll_dy, coll_dx)

                # 更新状态：等号左边和右边现在都是 (N_coll,) 维度
                self.state[coll, 10] = torch.cos(angle) * 5.0 + coll_pvx
                self.state[coll, 11] = torch.sin(angle) * 5.0 - 6.5
                self.last_touch[coll] = 1.0 if offset == 0 else -1.0

        # 得分判定
        dones = (by >= GROUND_Y - BALL_RADIUS)
        rewards = torch.zeros(self.num_envs, device=self.device)
        if dones.any():
            p1_win = bx > self.HALF_WIDTH
            # 这里也需要用 dones 掩码或者直接赋值
            rewards = torch.where(p1_win, 1.0, -1.0)
            # 注意：在 train 逻辑里，我们要的是 dones 时刻的 reward
            # 如果没结束，reward 应该是 0
            rewards = torch.where(dones, rewards, 0.0)

        return rewards, dones

    def get_obs(self, indices=None):
        if indices is None:
            s = self.state
            lt = self.last_touch
        else:
            s = self.state[indices]
            lt = self.last_touch[indices]

        obs_p1 = torch.cat([s, lt.unsqueeze(1)], dim=1)

        s2 = s.clone()
        s2[:, [0, 4, 8]] = self.WIDTH_T - s2[:, [0, 4, 8]]  # 镜像 X
        s2[:, [2, 6, 10]] *= -1.0  # 镜像 VX
        s2_swapped = s2[:, [4, 5, 6, 7, 0, 1, 2, 3, 8, 9, 10, 11]]
        obs_p2 = torch.cat([s2_swapped, -lt.unsqueeze(1)], dim=1)
        return obs_p1, obs_p2