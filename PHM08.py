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

    def generate_data_for_classification(self):
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

    def generate_data_for_regression(self):
        n_cycles = len(self.time)
        settings = np.array(self.settings)
        sensors = np.array(self.sensors)
        X_train = np.row_stack((settings, sensors))

        y_train = np.arange(n_cycles)
        y_train = y_train[::-1]

        X_train = np.array(X_train).transpose()
        y_train = np.array(y_train).reshape((n_cycles, 1))

        y_train[np.where(y_train > 125)] = 125

        return X_train, y_train, n_cycles
