# main.py

import pygame
from constants import *
from ball import Ball
from paddle import Paddle
from level_manager import LevelManager
from collision_handler import CollisionHandler
from game_state import GameState
from speed_controller import SpeedController
from input_handler import InputHandler

class BreakwallGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()

        self.ball = Ball(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, BALL_SIZE)
        self.paddle = Paddle(SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2, SCREEN_HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.level_manager = LevelManager()
        self.collision_handler = CollisionHandler()
        self.game_state = GameState()
        self.speed_controller = SpeedController()
        self.input_handler = InputHandler(self.paddle, self.speed_controller)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.input_handler.handle_input()

            self.screen.fill(BLACK)

            self.paddle.draw(self.screen)
            self.ball.update()
            self.ball.draw(self.screen)
            self.level_manager.update_bricks()
            self.level_manager.draw_bricks(self.screen)

            self.collision_handler.check_ball_paddle_collision(self.ball, self.paddle)
            self.collision_handler.check_ball_brick_collision(self.ball, self.level_manager.bricks)

            if self.ball.rect.bottom > SCREEN_HEIGHT:
                self.game_state.lose_life()
                if self.game_state.state == GAME_OVER:
                    running = False
                else:
                    self.ball.reset(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)

            self.game_state.check_win(self.level_manager.bricks)
            if self.game_state.state == GAME_WON:
                running = False

            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    game = BreakwallGame()
    game.run()
