# level_manager.py

from brick import Brick
from constants import *

class LevelManager:
    def __init__(self):
        self.bricks = []
        self.create_level()

    def create_level(self):
        self.bricks.clear()
        for row in range(BRICK_ROWS):
            for col in range(BRICK_COLUMNS):
                x = col * BRICK_WIDTH
                y = row * BRICK_HEIGHT + 50  # Offset to avoid overlap with top of the screen
                brick = Brick(x, y, BRICK_WIDTH, BRICK_HEIGHT)
                self.bricks.append(brick)

    def draw_bricks(self, screen):
        for brick in self.bricks:
            if not brick.is_destroyed:
                brick.draw(screen)

    def update_bricks(self):
        for brick in self.bricks:
            brick.update()

if __name__ == "__main__":
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    level_manager = LevelManager()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        screen.fill(BLACK)
        level_manager.draw_bricks(screen)
        pygame.display.flip()
    pygame.quit()
