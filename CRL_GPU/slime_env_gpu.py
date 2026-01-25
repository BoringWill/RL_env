import torch
import math


class SlimeVolleyballGPU:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device

        # --- 常量定义 (与 constants.py 保持一致) ---
        self.WIDTH = 1000.0
        self.HEIGHT = 500.0
        self.GRAVITY = 0.4
        self.PLAYER_SPEED = 6.0
        self.JUMP_POWER = -8.0
        self.BALL_MAX_SPEED = 13.0
        self.BALL_RADIUS = 10.0
        self.SLIME_RADIUS = 30.0
        self.NET_WIDTH = 10.0
        self.NET_HEIGHT = 100.0
        self.NET_X = self.WIDTH / 2.0
        self.NET_Y = self.HEIGHT - 50.0 - self.NET_HEIGHT
        self.GROUND_Y = self.HEIGHT - 50.0
        self.WIN_SCORE = 10
        self.MAX_STEPS = 3000  # 防止死循环

    def reset(self, seed=None):
        """完全并行的环境重置"""
        # 状态张量: [N, 14]
        # 0: p1_x, 1: p1_y, 2: p1_vx, 3: p1_vy
        # 4: p2_x, 5: p2_y, 6: p2_vx, 7: p2_vy
        # 8: b_x,  9: b_y,  10: b_vx, 11: b_vy
        # 12: p1_score, 13: p2_score

        self.states = torch.zeros((self.num_envs, 14), device=self.device, dtype=torch.float32)
        self.env_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)

        # 初始化分数
        self.states[:, 12] = 0
        self.states[:, 13] = 0

        # 内部重置球和位置
        self._reset_positions(torch.arange(self.num_envs, device=self.device))

        return self._get_obs()

    def _reset_positions(self, env_indices):
        """重置指定环境的物理位置 (发球)"""
        if len(env_indices) == 0: return

        # P1 归位
        self.states[env_indices, 0] = 200.0
        self.states[env_indices, 1] = self.GROUND_Y
        self.states[env_indices, 2:4] = 0.0

        # P2 归位
        self.states[env_indices, 4] = 800.0
        self.states[env_indices, 5] = self.GROUND_Y
        self.states[env_indices, 6:8] = 0.0

        # 球的初始化 (轮流发球逻辑简化: 随机发球或者固定发球)
        # 这里简单起见，根据总分偶数左边发，奇数右边发
        total_score = self.states[env_indices, 12] + self.states[env_indices, 13]
        spawn_left = (total_score % 2 == 0)

        self.states[env_indices, 8] = torch.where(spawn_left, torch.tensor(200.0, device=self.device),
                                                  torch.tensor(800.0, device=self.device))
        self.states[env_indices, 9] = 150.0
        self.states[env_indices, 10] = 0.0
        self.states[env_indices, 11] = 1.0  # 初始轻微下落/上升

    def step(self, actions):
        """
        actions: [N, 2] int tensor (p1_action, p2_action)
        """
        self.env_steps += 1

        # 解包状态以便操作 (都是 view，不消耗内存)
        p1_x, p1_y = self.states[:, 0], self.states[:, 1]
        p1_vx, p1_vy = self.states[:, 2], self.states[:, 3]
        p2_x, p2_y = self.states[:, 4], self.states[:, 5]
        p2_vx, p2_vy = self.states[:, 6], self.states[:, 7]
        bx, by = self.states[:, 8], self.states[:, 9]
        bvx, bvy = self.states[:, 10], self.states[:, 11]

        act1, act2 = actions[:, 0], actions[:, 1]

        # --- 1. 玩家物理 ---
        # P1 动作
        p1_vx[:] = 0.0
        p1_vx = torch.where(act1 == 1, -self.PLAYER_SPEED, p1_vx)
        p1_vx = torch.where(act1 == 2, self.PLAYER_SPEED, p1_vx)
        # 跳跃 (仅当在地面时)
        can_jump1 = (p1_y >= self.GROUND_Y - self.SLIME_RADIUS) & (p1_vy == 0)
        p1_vy = torch.where((act1 == 3) & can_jump1, self.JUMP_POWER, p1_vy)

        # P2 动作 (同理)
        p2_vx[:] = 0.0
        p2_vx = torch.where(act2 == 1, -self.PLAYER_SPEED, p2_vx)
        p2_vx = torch.where(act2 == 2, self.PLAYER_SPEED, p2_vx)
        can_jump2 = (p2_y >= self.GROUND_Y - self.SLIME_RADIUS) & (p2_vy == 0)
        p2_vy = torch.where((act2 == 3) & can_jump2, self.JUMP_POWER, p2_vy)

        # 应用重力和移动
        p1_x += p1_vx
        p1_y += p1_vy
        p2_x += p2_vx
        p2_y += p2_vy

        # 玩家重力与地面碰撞
        p1_vy += self.GRAVITY
        p2_vy += self.GRAVITY

        # 地面限制
        on_ground1 = p1_y + self.SLIME_RADIUS >= self.GROUND_Y
        p1_y = torch.where(on_ground1, self.GROUND_Y - self.SLIME_RADIUS, p1_y)
        p1_vy = torch.where(on_ground1, 0.0, p1_vy)

        on_ground2 = p2_y + self.SLIME_RADIUS >= self.GROUND_Y
        p2_y = torch.where(on_ground2, self.GROUND_Y - self.SLIME_RADIUS, p2_y)
        p2_vy = torch.where(on_ground2, 0.0, p2_vy)

        # 玩家网和墙壁碰撞限制
        # P1: [0, NET_X - NET_W/2]
        p1_x = torch.clamp(p1_x, self.SLIME_RADIUS, self.NET_X - self.NET_WIDTH / 2 - self.SLIME_RADIUS)
        # P2: [NET_X + NET_W/2, WIDTH]
        p2_x = torch.clamp(p2_x, self.NET_X + self.NET_WIDTH / 2 + self.SLIME_RADIUS, self.WIDTH - self.SLIME_RADIUS)

        # --- 2. 球物理 ---
        bx += bvx
        by += bvy
        bvy += self.GRAVITY

        # 墙壁反弹
        hit_wall_left = bx - self.BALL_RADIUS < 0
        hit_wall_right = bx + self.BALL_RADIUS > self.WIDTH

        bx = torch.where(hit_wall_left, self.BALL_RADIUS, bx)
        bvx = torch.where(hit_wall_left, torch.abs(bvx) * 0.8, bvx)

        bx = torch.where(hit_wall_right, self.WIDTH - self.BALL_RADIUS, bx)
        bvx = torch.where(hit_wall_right, -torch.abs(bvx) * 0.8, bvx)

        # 网的碰撞检测 (简化版: 顶部反弹 + 侧面反弹)
        # 顶部
        in_net_x = (bx > self.NET_X - self.NET_WIDTH / 2) & (bx < self.NET_X + self.NET_WIDTH / 2)
        hit_net_top = in_net_x & (by + self.BALL_RADIUS >= self.NET_Y) & (by - bvy < self.NET_Y)

        bvy = torch.where(hit_net_top, -torch.abs(bvy) * 0.8, bvy)
        by = torch.where(hit_net_top, self.NET_Y - self.BALL_RADIUS, by)

        # 侧面
        below_net_top = by > self.NET_Y
        hit_net_left = below_net_top & (bx + self.BALL_RADIUS > self.NET_X - self.NET_WIDTH / 2) & (bx < self.NET_X)
        hit_net_right = below_net_top & (bx - self.BALL_RADIUS < self.NET_X + self.NET_WIDTH / 2) & (bx > self.NET_X)

        bvx = torch.where(hit_net_left, -torch.abs(bvx) * 0.8, bvx)
        bx = torch.where(hit_net_left, self.NET_X - self.NET_WIDTH / 2 - self.BALL_RADIUS, bx)

        bvx = torch.where(hit_net_right, torch.abs(bvx) * 0.8, bvx)
        bx = torch.where(hit_net_right, self.NET_X + self.NET_WIDTH / 2 + self.BALL_RADIUS, bx)

        # --- 3. 球与玩家碰撞 (核心逻辑) ---
        for (px, py, pvx, pvy) in [(p1_x, p1_y, p1_vx, p1_vy), (p2_x, p2_y, p2_vx, p2_vy)]:
            dx = bx - px
            dy = by - py
            dist_sq = dx ** 2 + dy ** 2
            min_dist = self.BALL_RADIUS + self.SLIME_RADIUS

            is_hit = (dist_sq < min_dist ** 2) & (by < py)  # 且球在上方

            # 计算碰撞响应 (仅对 is_hit 为真的环境生效)
            if torch.any(is_hit):
                dist = torch.sqrt(dist_sq)
                angle = torch.atan2(dy, dx)

                # 速度更新
                base_speed = 4.0  # 基础弹力
                upward_force = -7.0

                # 这里必须小心处理 tensor 维度
                new_vx = torch.cos(angle) * base_speed + pvx * 2.0
                new_vy = torch.sin(angle) * base_speed + upward_force + pvy * 2.0

                # 限速
                spd = torch.sqrt(new_vx ** 2 + new_vy ** 2)
                scale = torch.clamp(self.BALL_MAX_SPEED / (spd + 1e-6), max=1.0)
                new_vx *= scale
                new_vy *= scale

                bvx = torch.where(is_hit, new_vx, bvx)
                bvy = torch.where(is_hit, new_vy, bvy)

                # 防止重叠
                overlap = min_dist - dist
                bx = torch.where(is_hit, bx + torch.cos(angle) * overlap, bx)
                by = torch.where(is_hit, by + torch.sin(angle) * overlap, by)

        # --- 4. 胜负判定 ---
        # 触地
        touch_ground = by >= self.GROUND_Y - self.BALL_RADIUS

        # 奖励计算 (P1视角)
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # 如果触地
        p2_score_inc = touch_ground & (bx < self.WIDTH / 2)
        p1_score_inc = touch_ground & (bx >= self.WIDTH / 2)

        self.states[:, 12] += p1_score_inc.int()
        self.states[:, 13] += p2_score_inc.int()

        rewards = torch.where(p1_score_inc, 2.0, rewards)
        rewards = torch.where(p2_score_inc, -2.0, rewards)

        # 小分重置 (Ball reset)
        reset_indices = torch.nonzero(touch_ground).squeeze(-1)
        self._reset_positions(reset_indices)

        # 游戏结束判定 (Winning Score)
        game_over = (self.states[:, 12] >= self.WIN_SCORE) | (self.states[:, 13] >= self.WIN_SCORE) | (
                    self.env_steps >= self.MAX_STEPS)
        dones = game_over

        # 游戏彻底结束重置
        full_reset_indices = torch.nonzero(game_over).squeeze(-1)
        if len(full_reset_indices) > 0:
            self.states[full_reset_indices, 12] = 0
            self.states[full_reset_indices, 13] = 0
            self.env_steps[full_reset_indices] = 0
            self._reset_positions(full_reset_indices)

        # 回写状态 (因为之前为了方便用了临时变量)
        self.states[:, 0], self.states[:, 1] = p1_x, p1_y
        self.states[:, 2], self.states[:, 3] = p1_vx, p1_vy
        self.states[:, 4], self.states[:, 5] = p2_x, p2_y
        self.states[:, 6], self.states[:, 7] = p2_vx, p2_vy
        self.states[:, 8], self.states[:, 9] = bx, by
        self.states[:, 10], self.states[:, 11] = bvx, bvy

        return self._get_obs(), rewards, dones, {}  # Info 暂略

    def _get_obs(self):
        """生成 P1 和 P2 的观测 (批量)"""
        # obs shape: [2, N, 12] -> 返回 (obs_p1, obs_p2)

        p1_x, p1_y, p1_vx, p1_vy = self.states[:, 0], self.states[:, 1], self.states[:, 2], self.states[:, 3]
        p2_x, p2_y, p2_vx, p2_vy = self.states[:, 4], self.states[:, 5], self.states[:, 6], self.states[:, 7]
        bx, by, bvx, bvy = self.states[:, 8], self.states[:, 9], self.states[:, 10], self.states[:, 11]

        # 归一化因子
        W, H = self.WIDTH, self.HEIGHT

        # P1 视角
        obs1 = torch.stack([
            p1_x / W, p1_y / H, p1_vx / 15.0, p1_vy / 15.0,
            p2_x / W, p2_y / H, p2_vx / 15.0, p2_vy / 15.0,
            bx / W, by / H, bvx / 15.0, bvy / 15.0
        ], dim=1)
        obs1 = obs1 * 2.0 - 1.0

        # P2 视角 (镜像)
        obs2 = torch.stack([
            (W - p2_x) / W, p2_y / H, -p2_vx / 15.0, p2_vy / 15.0,
            (W - p1_x) / W, p1_y / H, -p1_vx / 15.0, p1_vy / 15.0,
            (W - bx) / W, by / H, -bvx / 15.0, bvy / 15.0
        ], dim=1)
        obs2 = obs2 * 2.0 - 1.0

        return obs1, obs2