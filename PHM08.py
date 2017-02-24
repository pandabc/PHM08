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

    def generate_data(self):
        n_cycles = len(self.time)
        settings = np.array(self.settings)
        sensors = np.array(self.sensors)
        features = np.row_stack((settings, sensors))

        x_train = np.zeros((24, 60))
        x_train[:, 0:30] = features[:, 0:30]
        x_train[:, 30:] = features[:, (n_cycles-30):n_cycles]
        x_train = x_train.transpose()

        y_train = np.column_stack((np.zeros((1, 30), dtype=int), np.ones((1, 30), dtype=int))).transpose()
        # y_train = np.zeros((60, 2))
        # y_train[0:30, 0] = 1
        # y_train[30:, 1] = 1

        return x_train, y_train


