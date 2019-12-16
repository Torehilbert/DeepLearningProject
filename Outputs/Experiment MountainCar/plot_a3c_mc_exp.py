import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy.interpolate import interp1d


if __name__ == "__main__":
    colors = [(0, 0, 0), (197 / 255, 225 / 255, 111 / 255), (41 / 255, 126 / 255, 124 / 255), (29 / 255, 52 / 255, 78 / 255)]
    linestyles = ['--', '-', '-', '-']
    parent_path = r"C:\Source\DeepLearningProject\Outputs\Experiment MountainCar"
    groups_idx = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [12, 13, 14]]
    x = np.linspace(0, 50000, 1001)
    x_name = 'Episodes'
    subfolders = os.listdir(parent_path)

    xs = []
    ys = []
    idx = 0
    dfs = []
    for sub in subfolders:
        if(not os.path.isdir(os.path.join(parent_path, sub))):
            continue

        data_path = os.path.join(parent_path, sub, 'log.csv')
        df = pd.read_csv(data_path)
        df['Index'] = idx
        f = interp1d(df[x_name], df['Reward'])
        df_smooth = pd.DataFrame()
        df_smooth[x_name] = x
        df_smooth['Index'] = idx
        df_smooth['Reward'] = f(x)
        df_smooth['Reward Smooth'] = df_smooth['Reward'].ewm(alpha=0.05).mean()
        dfs.append(df_smooth)
        idx += 1
    
    df = pd.concat(dfs)
    #df.to_csv('test.csv')
    print(df)
    counter = 0
    df_plot = pd.DataFrame()
    for idx in groups_idx:
        counter += 1
        group_indicies = df['Index'].isin(idx)
        group = df[group_indicies].groupby(x_name)
        df_mean = group.mean()
        print(df_mean)
        df_min = group.min()
        df_max = group.max()
        df_plot[x_name] = x
        df_plot['Group %d mean' % counter] = df_mean['Reward Smooth'].values
        df_plot['Group %d min' % counter] = df_min['Reward Smooth'].values
        df_plot['Group %d max' % counter] = df_max['Reward Smooth'].values
    
    print(df_plot)
    ax = plt.subplot(1, 1, 1)
    for i in range(len(groups_idx)):
        ax.fill_between(df_plot[x_name], df_plot['Group %d min' % (i + 1)], df_plot['Group %d max' % (i+1)], alpha=0.3, color=colors[i])
        plt.plot(df_plot[x_name], df_plot['Group %d mean' % (i + 1)], color=colors[i], linestyle=linestyles[i])
    plt.legend(['0.01', '1.00', '2.00', 'Dynamic'])
    plt.ylabel('Validation reward')
    plt.xlabel('Episodes')
    plt.title('Entropy weight in MountainCar-v0')
    plt.tight_layout()
    plt.show()