import matplotlib.pyplot as plt


class TrainTracker:
    def __init__(self, data_containers, train_thread, smooth_alphas=1):
        if(not hasattr(data_containers, '__len__')):
            data_containers = [data_containers]
        self.n = len(data_containers)

        # Smoothing
        if(not hasattr(smooth_alphas, '__len__')):
            smooth_alphas = [smooth_alphas] * self.n
        self.smooth_alphas = smooth_alphas
        self.smooth_active = [None]*self.n
        for i in range(self.n):
            self.smooth_active[i] = True if abs(smooth_alphas[i] - 1) > 1e-6 else False

        self.data_containers = data_containers
        self.train_thread = train_thread
        self.cursors = [0] * self.n
        self.previous_end_data = [None] * self.n
        self.previous_smooth_data = [None] * self.n

    def initialize(self):
        self.fig, self.axes = plt.subplots(self.n, 1)
        return self.axes

    def format(self, id=0, xlabel=None, ylabel=None, xlim=None, ylim=None):
        ax = self.axes[id]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if(xlim is not None):
            ax.set_xlim(xlim)
        if(ylim is not None):
            ax.set_ylim(ylim)

    def start(self, update_interval=0.5):
        while(self.train_thread.is_alive()):
            self._update()
            plt.pause(update_interval)
        self._update()
        plt.show()

    def _update(self):
        for i in range(self.n):
            data = self.data_containers[i].extract_data()
            count = len(data)
            if(count == 0):
                continue

            xs = [self.cursors[i] - 1] + list(range(self.cursors[i], self.cursors[i] + count))
            ys = [self.previous_end_data[i]] + data
            self.previous_end_data[i] = data[-1]
            self.axes[i].plot(xs, ys, color='gray', lw=0.25)

            if(self.smooth_active[i]):
                smooth_data = self._exponential_smoothing(self.previous_smooth_data[i], data, self.smooth_alphas[i])
                if(smooth_data is not None):
                    xs = [self.cursors[i] - 1] + list(range(self.cursors[i], self.cursors[i] + count))
                    ys = [self.previous_smooth_data[i]] + smooth_data
                    self.axes[i].plot(xs, ys, color='g', lw=1)
                    self.previous_smooth_data[i] = smooth_data[-1]

            self.cursors[i] += count

    def _exponential_smoothing(self, start, data, alpha):
        if(data is None):
            return None
        
        if(not hasattr(data, '__len__')):
            data = [data]

        current = start if start is not None else data[0]
        for i in range(len(data)):
            data[i] = alpha*data[i] + (1-alpha)*current
            current = data[i]
        
        return data




# def visualize_training(dataContainers, threadTrain):
#     n = len(dataContainers)
#     cursors = [0] * n
#     lastPoints = [None] * n
#     lastSmooth = [None] * n

#     smooth = [True] * n

#     fig, axes = plt.subplots(n, 1)
#     while(threadTrain.is_alive()):

#         for i in range(n):
#             data = dataContainers[i].extract_data()
#             n2 = len(data)
#             smooth = _exponential_smoothing()

#             axes[i].plot([cursors[i]-1] + list(range(cursors[i], cursors[i] + n2)), [lastPoints[i]] + data, color='g', lw=0.25)
#             cursors[i] = cursors[i] + n2
#             if(n2 > 0):
#                 lastPoints[i] = data[-1]

#         plt.pause(0.5)


