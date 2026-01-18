# entities.py
import pygame
import math
from constants import *


class Entity:
    def __init__(self, x, y, radius, color):
        self.x, self.y = x, y
        self.radius = radius
        self.color = color
        self.vx, self.vy = 0, 0

    def apply_physics(self):
        self.x += self.vx
        self.y += self.vy
        if self.y + self.radius < GROUND_Y:
            self.vy += GRAVITY
        else:
            self.y = GROUND_Y - self.radius
            self.vy = 0

    def draw_slime(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.rect(screen, COLOR_BG, (self.x - self.radius, self.y, self.radius * 2, self.radius))


class SlimeBall(Entity):
    def update(self):
        self.apply_physics()
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = abs(self.vx) * 0.8
        elif self.x + self.radius > WIDTH:
            self.x = WIDTH - self.radius
            self.vx = -abs(self.vx) * 0.8

    def draw_ball(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (0, 0, 0), (int(self.x), int(self.y)), self.radius, 2)

    def check_player_collision(self, slime):
        dx = self.x - slime.x
        dy = self.y - slime.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        if dist < (self.radius + slime.radius) and self.y < slime.y:
            angle = math.atan2(dy, dx)

            # 1. 基础反弹速度（确保球会向上弹）
            base_speed = 6
            self.vx = math.cos(angle) * base_speed
            self.vy = math.sin(angle) * base_speed - 10

            # 2. 注入两倍玩家速度 (核心要求)
            # 球的新速度 = 基础速度 + 玩家速度 * 2
            self.vx += slime.vx * 2.0
            self.vy += slime.vy * 2.0

            # 3. 速度上限平滑处理 (防止球飞出宇宙)
            current_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            if current_speed > BALL_MAX_SPEED:
                scale = BALL_MAX_SPEED / current_speed
                self.vx *= scale
                self.vy *= scale

            # 4. 位置修正 (防止粘连)
            overlap = (self.radius + slime.radius) - dist
            self.x += math.cos(angle) * overlap
            self.y += math.sin(angle) * overlap

    def check_net_collision(self):
        if abs(self.x - NET_X) < (self.radius + NET_WIDTH / 2) and self.y > NET_Y:
            if self.y < NET_Y + 15:  # 撞顶
                self.y = NET_Y - self.radius
                self.vy = -abs(self.vy) * 0.8
            else:  # 撞侧面
                if self.x < NET_X:
                    self.x = NET_X - NET_WIDTH / 2 - self.radius
                    self.vx = -abs(self.vx) * 0.8
                else:
                    self.x = NET_X + NET_WIDTH / 2 + self.radius
                    self.vx = abs(self.vx) * 0.8