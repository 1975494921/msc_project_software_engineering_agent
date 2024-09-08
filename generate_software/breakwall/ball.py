# ball.py

import pygame
from game_object import GameObject
from constants import *

class Ball(GameObject):
    def __init__(self, x, y, size, color=WHITE, speed=INITIAL_BALL_SPEED):
        super().__init__(x, y, size, size, color)
        self.speed = speed
        self.dx = speed
        self.dy = -speed

    def update(self):
        self.rect.x += self.dx
        self.rect.y += self.dy

        # Bounce off the walls
        if self.rect.left < 0 or self.rect.right > SCREEN_WIDTH:
            self.dx = -self.dx
        if self.rect.top < 0:
            self.dy = -self.dy

    def reset(self, x, y):
        self.rect.x = x
        self.rect.y = y
        self.dx = self.speed
        self.dy = -self.speed

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    ball = Ball(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, BALL_SIZE)
    running = True
    clock = pygame.time.Clock()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(BLACK)
        ball.update()
        ball.draw(screen)
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()
