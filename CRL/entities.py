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
        # 视觉上依然保持半圆绘制
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.rect(screen, COLOR_BG, (self.x - self.radius, self.y, self.radius * 2, self.radius))


class SlimeBall(Entity):
    def __init__(self, x, y, radius, color):
        super().__init__(x, y, radius, color)
        self.speed_multiplier = 1.0

    def draw_ball(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

    def update(self):
        self.apply_physics()
        if self.x - self.radius < 0:
            self.x = self.radius
            self.vx = abs(self.vx) * 0.8
        elif self.x + self.radius > WIDTH:
            self.x = WIDTH - self.radius
            self.vx = -abs(self.vx) * 0.8

    def check_player_collision(self, slime):
        dx = self.x - slime.x
        dy = self.y - slime.y
        dist = math.sqrt(dx ** 2 + dy ** 2)

        # 修改核心：判定范围改为整个圆，不再限制 self.y < slime.y
        if dist < (self.radius + slime.radius):
            # 计算碰撞法线角度
            angle = math.atan2(dy, dx)

            # 统一的反弹逻辑：基于碰撞点角度
            # 如果球在玩家下方(dy > 0)，sin(angle)为正，vy自然向下(扣球)
            # 如果球在玩家上方(dy < 0)，sin(angle)为负，vy自然向上(垫球)
            speed = 15.0 * self.speed_multiplier

            # 玩家速度对球的贡献
            self.vx = math.cos(angle) * speed + (slime.vx * 0.5)
            self.vy = math.sin(angle) * speed + (slime.vy * 0.5)

            # 速度上限限制
            current_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)
            dynamic_max_speed = BALL_MAX_SPEED * self.speed_multiplier
            if current_speed > dynamic_max_speed:
                scale = dynamic_max_speed / current_speed
                self.vx *= scale
                self.vy *= scale

            # --- 防止穿透：硬位移修正 ---
            overlap = (self.radius + slime.radius) - dist
            self.x += math.cos(angle) * overlap
            self.y += math.sin(angle) * overlap
            return True
        return False

    def check_net_collision(self):
        if abs(self.x - NET_X) < (self.radius + NET_WIDTH / 2) and self.y > NET_Y:
            if self.x < NET_X:
                self.x = NET_X - NET_WIDTH / 2 - self.radius
                self.vx = -abs(self.vx) * 0.8
            else:
                self.x = NET_X + NET_WIDTH / 2 + self.radius
                self.vx = abs(self.vx) * 0.8