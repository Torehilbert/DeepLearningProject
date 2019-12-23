import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy.interpolate import interp1d


if __name__ == "__main__":
    colors = [(197 / 255, 225 / 255, 111 / 255), (41 / 255, 126 / 255, 124 / 255), (29 / 255, 52 / 255, 78 / 255), (1,0,0), (0,1,0)]
    parent_path = r"C:\Source\DeepLearningProject\Outputs\Experiment A3C Cores"
    parent_path = r"C:\Users\ToreH\Desktop\LunarLander Results"
    groups_idx = [[0, 1, 2, 3, 4, 5], [9, 10, 11, 12, 13, 14]]
    groups_idx = [list(range(0,15)), list(range(15,30)), list(range(30,45)), list(range(45,60))]
    x = np.linspace(1, 8000, 1001)
    #x = np.linspace(0, 20000, 1001)
    x_name = 'Time'  # 'Episodes'
    #x_name = 'Episodes'
    PLOT_LABEL_X = 'Wallclock time (s)'
    #PLOT_LABEL_X = x_name


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
        xsub = x[x < np.max(df[x_name].values)]
        f = interp1d(df[x_name], df['Reward'])
        df_smooth = pd.DataFrame()
        df_smooth[x_name] = xsub
        df_smooth['Index'] = idx
        df_smooth['Reward'] = f(xsub)
        df_smooth['Reward Smooth'] = df_smooth['Reward'].ewm(alpha=0.05).mean()
        dfs.append(df_smooth)
        idx += 1
    
    df = pd.concat(dfs)
    df.to_csv('test.csv')
    counter = 0

    ax = plt.subplot(1, 1, 1)
    max_smooth_reward = -10000000
    for idx in groups_idx:
        counter += 1
        group_indicies = df['Index'].isin(idx)
        dfsub = df[group_indicies]
        group = dfsub.groupby(x_name)
        xsub = list(group.groups.keys())

        df_mean = group.mean()
        df_min = group.min()
        df_max = group.max()
        if(np.max(df_mean['Reward Smooth']) > max_smooth_reward):
            max_smooth_reward = np.max(df_mean['Reward Smooth'])
        ax.fill_between(xsub, df_min['Reward Smooth'].values, df_max['Reward Smooth'].values, alpha=0.3, color=colors[counter - 1])
        plt.plot(xsub, df_mean['Reward Smooth'].values, color=colors[counter - 1])
    plt.axhline(y=max_smooth_reward, color='black', alpha=0.3, linestyle='--', lw=1)

    plt.title('Number of processes in LunarLander-v2')
    plt.legend(['1', '6', '12', '23'])
    plt.ylabel('Validation reward')
    plt.xlabel(PLOT_LABEL_X)
    plt.xlim([np.min(x), np.max(x)])
    plt.xlim([48, np.max(x)])
    plt.ylim([0, 280])
    plt.tight_layout()
    ax.set_xscale('log')
    plt.show()
