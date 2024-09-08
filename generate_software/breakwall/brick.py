# brick.py

import pygame
from game_object import GameObject
from constants import *

class Brick(GameObject):
    def __init__(self, x, y, width, height, color=GREEN):
        super().__init__(x, y, width, height, color)
        self.is_destroyed = False

    def update(self):
        if self.is_destroyed:
            self.rect.x = -100  # Move off-screen
            self.rect.y = -100

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    brick = Brick(100, 100, BRICK_WIDTH, BRICK_HEIGHT)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(BLACK)
        brick.draw(screen)
        pygame.display.flip()
    pygame.quit()
