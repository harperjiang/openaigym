from collections import deque
from statistics import mean

class TerminateChecker:

    def __init__(self, threshold):
        self.threshold = threshold
        self.container = deque()
        self.maxSize = 100

    def record(self, value):
        self.container.append(value)
        while len(self.container) > self.maxSize:
            self.container.popleft()

    def success(self):
        if len(self.container)< self.maxSize:
            return False
        return mean(self.container) >= self.threshold
