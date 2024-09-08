# speed_controller.py

from constants import INITIAL_BALL_SPEED

class SpeedController:
    def __init__(self, initial_speed=INITIAL_BALL_SPEED):
        self.speed = initial_speed

    def increase_speed(self, increment=1):
        self.speed += increment

    def decrease_speed(self, decrement=1):
        if self.speed - decrement > 0:
            self.speed -= decrement

    def get_speed(self):
        return self.speed

if __name__ == "__main__":
    speed_controller = SpeedController()
    print(f"Initial speed: {speed_controller.get_speed()}")

    speed_controller.increase_speed()
    print(f"Speed after increase: {speed_controller.get_speed()}")

    speed_controller.decrease_speed()
    print(f"Speed after decrease: {speed_controller.get_speed()}")
