import matplotlib.pyplot as plt
import os


class TrainTracker:
    def __init__(self, data_containers, train_thread, smooth_alphas=1, out_filepaths=None):
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
        self.cursors = [1] * self.n
        self.previous_end_data = [0] * self.n
        self.previous_smooth_data = [None] * self.n
        self.visualize = False

        self.out_files = [None] * self.n
        if(out_filepaths is not None):
            if(not hasattr(out_filepaths, '__len__')):
                out_filepaths = [out_filepaths]
            for i in range(min(self.n, len(out_filepaths))):
                os.makedirs(os.path.dirname(out_filepaths[i]), exist_ok=True)
                self.out_files[i] = open(out_filepaths[i], 'w')

    def initialize(self):
        self.visualize = True
        self.fig, self.axes = plt.subplots(self.n, 1)
        return self.axes

    def format(self, id=0, xlabel=None, ylabel=None, xlim=None, ylim=None, logy=False):
        ax = self.axes[id]
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if(xlim is not None):
            ax.set_xlim(xlim)
        if(ylim is not None):
            ax.set_ylim(ylim)
        if(logy):
            ax.set_yscale('log')

    def start(self, update_interval=0.5):
        while(self.train_thread.is_alive()):
            self._update()
            plt.pause(update_interval)

        self._update()
        for i in range(self.n):
            if(self.out_files[i] is not None):
                self.out_files[i].close()
        plt.show()

    def _update(self):
        for i in range(self.n):
            # A) extract recent data
            data = self.data_containers[i].extract_data()
            count = len(data)
            if(count == 0):
                continue

            # B) write to output files
            if(self.out_files[i] is not None):
                for j in range(count):
                    self.out_files[i].write(str(data[j]) + "\n")

            if(self.visualize):
                # C) plot raw data
                xs = [self.cursors[i] - 1] + list(range(self.cursors[i], self.cursors[i] + count))
                ys = [self.previous_end_data[i]] + data
                self.previous_end_data[i] = data[-1]
                self.axes[i].plot(xs, ys, color='gray', lw=0.25)

                # D) plot smooth data
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
