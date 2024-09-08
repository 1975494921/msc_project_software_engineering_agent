# game_state.py

from constants import *

class GameState:
    def __init__(self):
        self.lives = INITIAL_LIVES
        self.score = 0
        self.state = GAME_RUNNING

    def lose_life(self):
        self.lives -= 1
        if self.lives <= 0:
            self.state = GAME_OVER

    def add_score(self, points):
        self.score += points

    def check_win(self, bricks):
        if all(brick.is_destroyed for brick in bricks):
            self.state = GAME_WON

    def reset(self):
        self.lives = INITIAL_LIVES
        self.score = 0
        self.state = GAME_RUNNING

if __name__ == "__main__":
    from level_manager import LevelManager
    game_state = GameState()
    level_manager = LevelManager()

    # Simulate losing a life
    game_state.lose_life()
    print(f"Lives after losing one: {game_state.lives}")

    # Simulate adding score
    game_state.add_score(10)
    print(f"Score after adding 10 points: {game_state.score}")

    # Simulate checking for win condition
    game_state.check_win(level_manager.bricks)
    print(f"Game state after checking win condition: {game_state.state}")

    # Reset game state
    game_state.reset()
    print(f"Game state after reset: Lives={game_state.lives}, Score={game_state.score}, State={game_state.state}")
