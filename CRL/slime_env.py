import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
from entities import Entity, SlimeBall
from constants import *


class SlimeSelfPlayEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        self.screen = None
        self.clock = None
        # 初始标记：None 代表第一局在中间发球
        self.last_winner = None

    def _get_obs(self, player_id):
        p1, p2, b = self.p1, self.p2, self.ball
        if player_id == 1:
            obs = [p1.x, p1.y, p1.vx, p1.vy, p2.x, p2.y, p2.vx, p2.vy, b.x, b.y, b.vx, b.vy]
        else:
            obs = [WIDTH - p2.x, p2.y, -p2.vx, p2.vy, WIDTH - p1.x, p1.y, -p1.vx, p1.vy, WIDTH - b.x, b.y, -b.vx, b.vy]

        obs = np.array(obs, dtype=np.float32)
        obs[[0, 4, 8]] /= WIDTH
        obs[[1, 5, 9]] /= HEIGHT
        obs[[2, 3, 6, 7, 10, 11]] /= 15.0
        return (obs * 2.0) - 1.0

    def reset(self, seed=None):
        super().reset(seed=seed)
        # 初始化玩家位置
        self.p1 = Entity(200, GROUND_Y, SLIME_RADIUS, COLOR_P1)
        self.p2 = Entity(800, GROUND_Y, SLIME_RADIUS, COLOR_P2)

        # --- 逻辑修改：首局中间，之后赢家头上发球 ---
        if self.last_winner is None:
            ball_x = WIDTH // 2
        else:
            ball_x = 200 if self.last_winner == 1 else 800

        self.ball = SlimeBall(ball_x, 100, BALL_RADIUS, COLOR_BALL)
        self.ball.vx = 0.0
        self.ball.vy = 2.0  # 垂直下落

        self.global_step_in_episode = 0
        return self._get_obs(1), {}

    def step(self, action_p1, action_p2):
        self.global_step_in_episode += 1
        prev_ball_x = self.ball.x

        # --- 核心修改：完全同步 test.py 的移动逻辑 ---
        for p, a in [(self.p1, action_p1), (self.p2, action_p2)]:
            p.vx = 0  # 对应 test.py 中的 p.vx = 0 (每帧先归零)

            if a == 1:  # A 或 LEFT
                p.vx = -PLAYER_SPEED
            elif a == 2:  # D 或 RIGHT
                p.vx = PLAYER_SPEED

            # 跳跃逻辑 (只有在地面 vy==0 时允许起跳)
            if a == 3 and p.vy == 0:
                p.vy = JUMP_POWER

        # 物理更新
        self.p1.apply_physics()
        self.p2.apply_physics()
        self.ball.update()

        # 玩家限制：不能穿过网，也不能出屏幕 (同步 test.py)
        self.p1.x = max(self.p1.radius, min(NET_X - NET_WIDTH / 2 - self.p1.radius, self.p1.x))
        self.p2.x = max(NET_X + NET_WIDTH / 2 + self.p2.radius, min(WIDTH - self.p2.radius, self.p2.x))

        # 碰撞检测
        self._custom_net_collision()  # 调用你要求的弹网逻辑
        self.ball.check_player_collision(self.p1)
        self.ball.check_player_collision(self.p2)

        # 奖励计算
        reward_p1 = 0.0
        if prev_ball_x < NET_X and self.ball.x >= NET_X:
            reward_p1 += 10.0
        if prev_ball_x > NET_X and self.ball.x <= NET_X:
            reward_p1 -= 5.0

        # 终局判定与发球位更新
        done = False
        if self.ball.y >= GROUND_Y - self.ball.radius:
            done = True
            if self.ball.x < WIDTH / 2:  # P1 丢分
                reward_p1 -= 20.0
                self.last_winner = 2  # 赢家是 P2
            else:  # P1 得分
                reward_p1 += 20.0
                self.last_winner = 1  # 赢家是 P1

        if self.global_step_in_episode > 2000:
            done = True

        if self.render_mode == "human":
            self.render()

        return self._get_obs(1), self._get_obs(2), reward_p1, done, {}

    def _custom_net_collision(self):
        """同步 test.py 中的弹网效果，并保留你要求的弹起逻辑"""
        b = self.ball
        net_left = NET_X - NET_WIDTH / 2
        net_right = NET_X + NET_WIDTH / 2

        # 撞击网顶弹起
        if b.y + b.radius >= NET_Y and b.y < NET_Y and net_left < b.x < net_right:
            b.vy = -abs(b.vy) * 0.8
            b.y = NET_Y - b.radius
        # 撞击网侧反弹
        elif net_left - b.radius < b.x < net_right + b.radius and b.y > NET_Y:
            b.vx *= -1.0
            if b.x < NET_X:
                b.x = net_left - b.radius
            else:
                b.x = net_right + b.radius

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            self.clock = pygame.time.Clock()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit()
        self.screen.fill(COLOR_BG)
        pygame.draw.rect(self.screen, COLOR_GROUND, (0, GROUND_Y, WIDTH, 50))
        pygame.draw.rect(self.screen, COLOR_NET, (NET_X - NET_WIDTH / 2, NET_Y, NET_WIDTH, NET_HEIGHT))
        self.p1.draw_slime(self.screen)
        self.p2.draw_slime(self.screen)
        self.ball.draw_ball(self.screen)
        pygame.display.flip()
        self.clock.tick(60)