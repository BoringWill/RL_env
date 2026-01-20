# test.py
import pygame
import sys
from constants import *
from entities import Entity, SlimeBall


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("史莱姆排球对战 - WAD vs 方向键")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 40)

    # 初始化对象，使用常量中的半径
    p1 = Entity(200, GROUND_Y, SLIME_RADIUS, COLOR_P1)
    p2 = Entity(800, GROUND_Y, SLIME_RADIUS, COLOR_P2)
    ball = SlimeBall(500, 100, BALL_RADIUS, COLOR_BALL)

    score = [0, 0]

    while True:
        # 1. 刷新事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit();
                sys.exit()

        # 2. 纯人工按键监听 (不依赖任何 manualMode)
        keys = pygame.key.get_pressed()

        # P1 控制 (WDA)
        p1.vx = 0
        if keys[pygame.K_a]: p1.vx = -PLAYER_SPEED
        if keys[pygame.K_d]: p1.vx = PLAYER_SPEED
        if keys[pygame.K_w] and p1.vy == 0: p1.vy = JUMP_POWER

        # P2 控制 (方向键)
        p2.vx = 0
        if keys[pygame.K_LEFT]: p2.vx = -PLAYER_SPEED
        if keys[pygame.K_RIGHT]: p2.vx = PLAYER_SPEED
        if keys[pygame.K_UP] and p2.vy == 0: p2.vy = JUMP_POWER

        # 3. 物理更新
        p1.apply_physics()
        p2.apply_physics()
        ball.update()

        # 玩家不能穿过网，也不能出屏幕
        p1.x = max(p1.radius, min(NET_X - NET_WIDTH / 2 - p1.radius, p1.x))
        p2.x = max(NET_X + NET_WIDTH / 2 + p2.radius, min(WIDTH - p2.radius, p2.x))

        # 4. 碰撞检测
        ball.check_player_collision(p1)
        ball.check_player_collision(p2)
        ball.check_net_collision()

        # 5. 得分判定 (球落地)
        if ball.y >= GROUND_Y - ball.radius:
            if ball.x < WIDTH / 2:
                score[1] += 1
            else:
                score[0] += 1
            # 重置球到得分方的对面
            ball.x, ball.y, ball.vx, ball.vy = (200 if ball.x > WIDTH / 2 else 800), 100, 0, 0
            p1.x, p2.x = 200, 800

        # 6. 渲染
        screen.fill(COLOR_BG)

        # 画地面和网
        pygame.draw.rect(screen, COLOR_GROUND, (0, GROUND_Y, WIDTH, 50))
        pygame.draw.rect(screen, COLOR_NET, (NET_X - NET_WIDTH / 2, NET_Y, NET_WIDTH, NET_HEIGHT))

        # 调用各自的绘制方法
        p1.draw_slime(screen)
        p2.draw_slime(screen)
        ball.draw_ball(screen)

        # 显示比分
        score_text = font.render(f"{score[0]} : {score[1]}", True, (255, 255, 255))
        screen.blit(score_text, (WIDTH // 2 - 40, 20))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()