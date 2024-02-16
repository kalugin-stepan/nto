import numpy as np

def clamp(x: float) -> float:
    while x > 180:
        x -= 360
    while x < -180:
        x += 360
    return x

class PID:
    e0 = 0
    Se = 0
    def __init__(self, P: float, I: float, D: float):
        self.P = P
        self.I = I
        self.D = D
    def process(self, e: float, dt: float) -> float:
        de = e - self.e0
        self.e0 = e
        self.Se += e*dt
        return self.P*e + self.I*self.Se + self.D*de/dt
    def reset(self):
        self.e0 = 0
        self.Se = 0