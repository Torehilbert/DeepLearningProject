import matplotlib.pyplot as plt
import time

from Data import DataUDPReceiver as DataReceiver


if __name__ == "__main__":
    names = [
        'Training reward',
        'Validation reward',
        'Loss critic',
        'Loss policy',
        'Loss entropy'
    ]

    n = 5
    rec = DataReceiver(n=n)

    # Create axes
    plt.figure()
    axs = []
    for i in range(n):
        axs.append(plt.subplot(n, 1, i + 1))

    has_legend = [False] * n
    x_old = [0] * n
    y_old = [None] * n

    # Update axes
    while(True):
        all_data = rec.fetch_all()
        for i in range(n):
            data = all_data[i]
            if(len(data) > 0):
                ax = axs[i]
                xs = list(range(x_old[i], x_old[i] + len(data) + 1))
                ax.plot(xs, [y_old[i]] + data, color='gray')
                x_old[i] = xs[-1]
                y_old[i] = data[-1]
                if(not has_legend[i]):
                    ax.legend([names[i]], loc='upper left')
                    has_legend[i] = True
        
        plt.pause(0.1)