import numpy as np


class PHM08(object):
    """PHM08
    algorithm for PHM 2008 challenge
    """
    def __init__(self):
        self.unit = 0  # unit number
        self.time = []  # time, in cycles
        self.settings = [[], [], []]  # 3 operational settings
        self.sensors = [[], [], [], [], [], [], [],
                        [], [], [], [], [], [], [],
                        [], [], [], [], [], [], []]

