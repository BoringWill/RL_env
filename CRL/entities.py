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
    # --- 修改点 1: 初始化倍率 ---
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

        if dist < (self.radius + slime.radius) and self.y < slime.y:
            angle = math.atan2(dy, dx)

            # --- 修改点 2: 基础速度乘以倍率 ---
            base_speed = 4 * self.speed_multiplier
            # 垂直向上的冲力也应该随倍率提升，否则球会感觉“飘”不起来
            upward_force = -7 * self.speed_multiplier

            self.vx = math.cos(angle) * base_speed + slime.vx * 2.0
            self.vy = math.sin(angle) * base_speed + upward_force + slime.vy * 2.0

            current_speed = math.sqrt(self.vx ** 2 + self.vy ** 2)

            # --- 修改点 3: 允许的最大速度上限也必须随倍率提升 ---
            dynamic_max_speed = BALL_MAX_SPEED * self.speed_multiplier
            if current_speed > dynamic_max_speed:
                scale = dynamic_max_speed / current_speed
                self.vx *= scale
                self.vy *= scale

            overlap = (self.radius + slime.radius) - dist
            self.x += math.cos(angle) * overlap
            self.y += math.sin(angle) * overlap
            return True  # 击球成功
        return False

    def check_net_collision(self):
        if abs(self.x - NET_X) < (self.radius + NET_WIDTH / 2) and self.y > NET_Y:
            if self.x < NET_X:
                self.x = NET_X - NET_WIDTH / 2 - self.radius
                self.vx = -abs(self.vx) * 0.8
            else:
                self.x = NET_X + NET_WIDTH / 2 + self.radius
                self.vx = abs(self.vx) * 0.8