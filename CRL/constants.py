# constants.py

# 窗口设置
WIDTH, HEIGHT = 1000, 500
FPS = 60

# --- 物理参数优化 ---
GRAVITY = 0.4         # 低重力，让球弹得更高更久
PLAYER_SPEED = 6        # 玩家左右移动速度
JUMP_POWER = -8        # 玩家跳跃力度
BALL_MAX_SPEED = 13      # 限制最高球速，防止太快

# --- 尺寸设置 ---
BALL_RADIUS = 10        # <--- 在这里修改球的大小 (圆形半径)
SLIME_RADIUS = 30       # 史莱姆的大小 (保持半圆形状)

# --- 网的设置 ---
NET_WIDTH = 10
NET_HEIGHT = 100        # 网高
NET_X = WIDTH // 2
NET_Y = HEIGHT - 50 - NET_HEIGHT # 网的顶部 Y 坐标

# --- 颜色 ---
COLOR_P1 = (255, 100, 100)
COLOR_P2 = (100, 100, 255)
COLOR_BALL = (255, 255, 0)
COLOR_GROUND = (120, 120, 120)
COLOR_NET = (200, 200, 200)
COLOR_BG = (30, 30, 30)
GROUND_Y = HEIGHT - 50