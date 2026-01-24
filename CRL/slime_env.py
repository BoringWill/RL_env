import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
from entities import Entity, SlimeBall
from constants import *
from collections import deque


class SlimeSelfPlayEnv(gym.Env):
    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)

        self.screen = None
        self.clock = None
        self.font = None

        self.p1_score = 0
        self.p2_score = 0
        self.win_score = 10
        # self.t_limit = 10000
        self.global_step_in_episode = 0

        # 球速控制 - 保持固定为 1.0
        self.ball_speed_multiplier = 1.0

    def _get_obs(self, player_id):
        p1, p2, b = self.p1, self.p2, self.ball
        if player_id == 1:
            obs = [p1.x, p1.y, p1.vx, p1.vy, p2.x, p2.y, p2.vx, p2.vy, b.x, b.y, b.vx, b.vy]
        else:
            # P2 视角：水平镜像处理
            obs = [WIDTH - p2.x, p2.y, -p2.vx, p2.vy, WIDTH - p1.x, p1.y, -p1.vx, p1.vy, WIDTH - b.x, b.y, -b.vx, b.vy]
        obs = np.array(obs, dtype=np.float32)
        # 归一化到 [-1, 1]
        obs[[0, 4, 8]] /= WIDTH
        obs[[1, 5, 9]] /= HEIGHT
        obs[[2, 3, 6, 7, 10, 11]] /= 15.0
        return (obs * 2.0) - 1.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p1_score = 0
        self.p2_score = 0
        self.global_step_in_episode = 0
        self._internal_point_reset(full_reset=True)
        return self._get_obs(1), {}

    def _internal_point_reset(self, full_reset=False):
        # 每一球重置时保持步数计时器重置，球速恒定
        self.global_step_in_episode = 0

        # 史莱姆回到各自半场的中心位置
        self.p1 = Entity(200, GROUND_Y, SLIME_RADIUS, COLOR_P1)
        self.p2 = Entity(800, GROUND_Y, SLIME_RADIUS, COLOR_P2)

        # 轮流发球逻辑
        total_points = self.p1_score + self.p2_score
        spawn_x = 200 if total_points % 2 == 0 else 800

        self.ball = SlimeBall(spawn_x, 150, BALL_RADIUS, COLOR_BALL)

        # 将固定倍率同步给球对象
        self.ball.speed_multiplier = self.ball_speed_multiplier

        # 发球速度应用固定倍率
        self.ball.vx = 0
        self.ball.vy = 1.0 * self.ball_speed_multiplier

    def step(self, actions):
        action_p1, action_p2 = actions
        self.global_step_in_episode += 1
        reward_p1 = 0.0
        terminated = False
        truncated = False

        # 执行动作
        for p, a in [(self.p1, action_p1), (self.p2, action_p2)]:
            p.vx = 0
            if a == 1:
                p.vx = -PLAYER_SPEED
            elif a == 2:
                p.vx = PLAYER_SPEED
            if a == 3 and p.vy == 0: p.vy = JUMP_POWER

        # 物理更新
        self.p1.apply_physics()
        self.p2.apply_physics()
        self.ball.update()
        self._custom_net_collision()

        # 碰撞检测
        self.ball.check_player_collision(self.p1)
        self.ball.check_player_collision(self.p2)

        # 限制史莱姆移动范围
        self.p1.x = max(self.p1.radius, min(NET_X - NET_WIDTH / 2 - self.p1.radius, self.p1.x))
        self.p2.x = max(NET_X + NET_WIDTH / 2 + self.p2.radius, min(WIDTH - self.p2.radius, self.p2.x))

        # 得分检测
        if self.ball.y >= GROUND_Y - self.ball.radius:
            if self.ball.x < WIDTH / 2:
                reward_p1 = -2.0
                self.p2_score += 1
            else:
                reward_p1 = 2.0
                self.p1_score += 1

            if self.p1_score >= self.win_score or self.p2_score >= self.win_score:
                terminated = True
            else:
                self._internal_point_reset(full_reset=False)

        return (self._get_obs(1),
                self._get_obs(2),
                reward_p1,
                terminated,
                truncated,
                {
                    "p2_raw_obs": self._get_obs(2),
                    "p1_score": self.p1_score,
                    "p2_score": self.p2_score,
                    "episode_steps": self.global_step_in_episode
                })

    def _custom_net_collision(self):
        b = self.ball
        nl, nr = NET_X - NET_WIDTH / 2, NET_X + NET_WIDTH / 2

        # 1. 球网顶部碰撞（y轴向下为正，NET_Y 是网顶的 y 坐标）
        # 判定条件：球的底部超过网顶，且上一时刻球在网顶上方，且球在网的宽度范围内
        if b.y + b.radius >= NET_Y and b.y - b.vy < NET_Y:
            if nl < b.x < nr:
                b.vy = -abs(b.vy) * 0.8
                b.y = NET_Y - b.radius  # 强制回弹到网顶上方
                return

        # 2. 球网侧面碰撞
        # 判定条件：球的 y 坐标在球网的高度区间内（NET_Y 到 GROUND_Y 之间）
        if b.y >= NET_Y:
            # 从左侧撞网
            if nl - b.radius < b.x < NET_X and b.vx > 0:
                b.vx = -abs(b.vx)
                b.x = nl - b.radius
            # 从右侧撞网
            elif NET_X < b.x < nr + b.radius and b.vx < 0:
                b.vx = abs(b.vx)
                b.x = nr + b.radius

    def render(self):
        if self.render_mode != "human": return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Slime Volleyball RL")
            self.font = pygame.font.SysFont("Arial", 24)
            self.clock = pygame.time.Clock()

        self.screen.fill(COLOR_BG)
        pygame.draw.rect(self.screen, COLOR_GROUND, (0, GROUND_Y, WIDTH, 50))
        pygame.draw.rect(self.screen, COLOR_NET, (NET_X - NET_WIDTH / 2, NET_Y, NET_WIDTH, NET_HEIGHT))

        score_txt = self.font.render(
            f"P1: {self.p1_score} | P2: {self.p2_score} | Speed: x{self.ball_speed_multiplier:.1f}", True, (0, 0, 0))
        self.screen.blit(score_txt, (WIDTH // 2 - 120, 20))

        self.p1.draw_slime(self.screen)
        self.p2.draw_slime(self.screen)
        self.ball.draw_ball(self.screen)

        pygame.display.flip()
        self.clock.tick(60)


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        self.observation_space = spaces.Box(
            low=np.tile(env.observation_space.low, n_frames),
            high=np.tile(env.observation_space.high, n_frames),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.frames.clear()
        for _ in range(self.n_frames): self.frames.append(obs)
        return np.concatenate(list(self.frames), axis=0), info

    def step(self, actions):
        obs_p1, obs_p2, reward, term, trunc, info = self.env.step(actions)
        self.frames.append(obs_p1)
        info["p2_raw_obs"] = obs_p2
        return np.concatenate(list(self.frames), axis=0), reward, term, trunc, info