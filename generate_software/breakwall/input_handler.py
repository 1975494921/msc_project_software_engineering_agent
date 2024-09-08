# input_handler.py

import pygame

class InputHandler:
    def __init__(self, paddle, speed_controller):
        self.paddle = paddle
        self.speed_controller = speed_controller

    def handle_input(self):
        keys = pygame.key.get_pressed()
        self.paddle.update(keys)

        if keys[pygame.K_UP]:
            self.speed_controller.increase_speed()
        if keys[pygame.K_DOWN]:
            self.speed_controller.decrease_speed()

if __name__ == "__main__":
    pygame.init()
    from paddle import Paddle
    from speed_controller import SpeedController
    from constants import *

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    paddle = Paddle(SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2, SCREEN_HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)
    speed_controller = SpeedController()
    input_handler = InputHandler(paddle, speed_controller)

    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        input_handler.handle_input()

        screen.fill(BLACK)
        paddle.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
