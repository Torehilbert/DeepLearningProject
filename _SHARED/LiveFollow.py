import matplotlib.pyplot as plt
import argparse
import numpy as np

from Data import DataUDPReceiver as DataReceiver


class Buffer:
    def __init__(self, buffersize):
        self.buffersize = buffersize
        self.xbuffer = [None] * buffersize
        self.ybuffer = [None] * buffersize
        self.cursor = 0
        self.xlast = -1
        self.dirty_flag = False
        self.full = False

    def add(self, y):
        self.xlast += 1
        self.xbuffer[self.cursor] = self.xlast
        self.ybuffer[self.cursor] = y
        self.cursor += 1
        if(self.cursor == self.buffersize):
            self.cursor = 0
            self.full = True
        self.dirty_flag = True

    def add_range(self, ys):
        for i in range(len(ys)):
            self.add(ys[i])

    def get(self):
        self.dirty_flag = False

        indx = np.argsort(self.xbuffer) if self.full else np.argsort(self.xbuffer[0:self.cursor])
        if(self.full):
            return np.array(self.xbuffer)[indx], np.array(self.ybuffer)[indx]
        else:
            xs = np.array(self.xbuffer[0:self.cursor])
            ys = np.array(self.ybuffer[0:self.cursor])
            return xs[indx], ys[indx]

    def has_changed(self):
        return self.dirty_flag


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', '--ch', '--c', '-c', nargs="+", default=[0], required=False, type=int)
    parser.add_argument('--names', nargs="+", required=False, type=str)
    parser.add_argument('--buffersize', '--size', default=100, required=False, type=int)
    args = parser.parse_args()

    channels = args.channels
    names = args.names

    buffersize = args.buffersize
    rec = DataReceiver(channels=channels)

    # Setup plot
    plt.figure()

    ch_to_ax = {}
    buffers = {}

    for i in range(len(channels)):
        ch = channels[i]

        ax = plt.subplot(len(channels), 1, i + 1)
        if(names is not None and len(names) > i):
            ax.legend([names[i]], loc='upper left')

        ch_to_ax[ch] = ax
        buffers[ch] = Buffer(buffersize)

    while(1):
        # Fetch new data
        all_data = rec.fetch_all()
        print(all_data)

        # Maintain rolling data buffers
        for i in range(len(all_data)):
            data = all_data[i]
            if(len(data) > 1 and data[0] in ch_to_ax):
                ch = int(data[0])
                buffers[ch].add_range(data[1:])

        # Update plots
        for ch in buffers:
            buffer = buffers[ch]
            if(not buffer.has_changed()):
                continue

            ax = ch_to_ax[ch]
            ax.clear()
            print(buffer.dirty_flag)
            xs, ys = buffer.get()
            ax.plot(xs, ys, color='gray', lw='0.25')

        plt.pause(0.01)
