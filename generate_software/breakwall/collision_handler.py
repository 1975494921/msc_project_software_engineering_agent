# collision_handler.py

from ball import Ball
from paddle import Paddle
from brick import Brick

class CollisionHandler:
    @staticmethod
    def check_ball_paddle_collision(ball, paddle):
        if ball.rect.colliderect(paddle.rect):
            ball.dy = -ball.dy
            # Adjust ball's dx based on where it hits the paddle
            relative_intersect_x = (paddle.rect.x + paddle.rect.width / 2) - ball.rect.x
            normalized_intersect_x = relative_intersect_x / (paddle.rect.width / 2)
            ball.dx = normalized_intersect_x * ball.speed

    @staticmethod
    def check_ball_brick_collision(ball, bricks):
        for brick in bricks:
            if not brick.is_destroyed and ball.rect.colliderect(brick.rect):
                brick.is_destroyed = True
                ball.dy = -ball.dy
                break  # Assuming one collision per frame

if __name__ == "__main__":
    import pygame
    from constants import *
    from level_manager import LevelManager

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    ball = Ball(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, BALL_SIZE)
    paddle = Paddle(SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2, SCREEN_HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)
    level_manager = LevelManager()
    collision_handler = CollisionHandler()

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        screen.fill(BLACK)

        paddle.update(keys)
        ball.update()
        level_manager.update_bricks()

        collision_handler.check_ball_paddle_collision(ball, paddle)
        collision_handler.check_ball_brick_collision(ball, level_manager.bricks)

        paddle.draw(screen)
        ball.draw(screen)
        level_manager.draw_bricks(screen)

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
