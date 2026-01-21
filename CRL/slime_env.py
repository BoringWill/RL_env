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
        self.last_winner = 1
        self.t_limit = 10000
        self.global_step_in_episode = 0

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p1_score = 0
        self.p2_score = 0
        self.global_step_in_episode = 0
        self._internal_point_reset(full_reset=True)
        return self._get_obs(1), {}

    def _internal_point_reset(self, full_reset=False):
        # 史莱姆回到各自初始位置
        self.p1 = Entity(200, GROUND_Y, SLIME_RADIUS, COLOR_P1)
        self.p2 = Entity(800, GROUND_Y, SLIME_RADIUS, COLOR_P2)

        # 修改点：谁赢球谁发球
        # full_reset 为 True 时默认 P1 发球，否则根据 last_winner 决定
        if full_reset:
            spawn_x = 200
        else:
            spawn_x = 200 if self.last_winner == 1 else 800

        self.ball = SlimeBall(spawn_x, 150, BALL_RADIUS, COLOR_BALL)
        self.ball.vx, self.ball.vy = 0, 1.0

    def step(self, actions):
        action_p1, action_p2 = actions
        self.global_step_in_episode += 1
        reward_p1 = 0.0
        terminated = False
        truncated = False

        for p, a in [(self.p1, action_p1), (self.p2, action_p2)]:
            p.vx = 0
            if a == 1:
                p.vx = -PLAYER_SPEED
            elif a == 2:
                p.vx = PLAYER_SPEED
            if a == 3 and p.vy == 0: p.vy = JUMP_POWER

        self.p1.apply_physics()
        self.p2.apply_physics()
        self.ball.update()
        self._custom_net_collision()

        self.ball.check_player_collision(self.p1)
        self.ball.check_player_collision(self.p2)

        self.p1.x = max(self.p1.radius, min(NET_X - NET_WIDTH / 2 - self.p1.radius, self.p1.x))
        self.p2.x = max(NET_X + NET_WIDTH / 2 + self.p2.radius, min(WIDTH - self.p2.radius, self.p2.x))

        if self.ball.y >= GROUND_Y - self.ball.radius:
            if self.ball.x < WIDTH / 2:
                # 修改点：P1 输球扣 2 分
                reward_p1 = -2.0
                self.p2_score += 1
                self.last_winner = 2
            else:
                # P1 赢球得 1 分
                reward_p1 = 2.0
                self.p1_score += 1
                self.last_winner = 1

            if self.p1_score >= self.win_score or self.p2_score >= self.win_score:
                terminated = True
            else:
                # 触发得分后重置（包含发球位置逻辑）
                self._internal_point_reset(full_reset=False)

        if self.global_step_in_episode >= self.t_limit:
            truncated = True

        return self._get_obs(1), self._get_obs(2), reward_p1, terminated, truncated, {"p2_raw_obs": self._get_obs(2)}

    def _custom_net_collision(self):
        b = self.ball
        nl, nr = NET_X - NET_WIDTH / 2, NET_X + NET_WIDTH / 2
        if b.y + b.radius >= NET_Y and b.y < NET_Y and nl < b.x < nr:
            b.vy = -abs(b.vy) * 0.8;
            b.y = NET_Y - b.radius
        elif nl - b.radius < b.x < nr + b.radius and b.y > NET_Y:
            b.vx *= -1.0;
            b.x = nl - b.radius if b.x < NET_X else nr + b.radius

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

        score_txt = self.font.render(f"P1: {self.p1_score} | P2: {self.p2_score}", True, (0, 0, 0))
        self.screen.blit(score_txt, (WIDTH // 2 - 80, 20))

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