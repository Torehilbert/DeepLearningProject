import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
from scipy.interpolate import interp1d


if __name__ == "__main__":
    parent_path = r"C:\Source\DeepLearningProject\Outputs\Experiment A3C Cores"
    groups_idx = [[0, 1, 2, 3, 4, 5], [9,10,11,12,13,14]]
    x = np.linspace(0, 1500, 1001)
    x_name = 'Time'  # 'Episodes'
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
        ax.fill_between(df_plot[x_name], df_plot['Group %d min' % (i+1)], df_plot['Group %d max' % (i+1)], alpha=0.3)
        plt.plot(df_plot[x_name], df_plot['Group %d mean' % (i+1)])
    plt.legend(['1', '6'])
    plt.show()



    # plt.figure()
    # for idx in groups_idx:
    #     plt.plot(df['Episodes'].values[df['Index'].isin(idx)], 
    #             df['Reward Smooth'].values[df['Index'].isin(idx)])
    # plt.show()


    # x = np.linspace(0, minxmax, 1000)
    # alpha = 0.95
    # xsf = []
    # ysf = []
    # ysfe = []
    # for i in range(len(xs)):
    #     f = interp1d(xs[i], ys[i])
    #     xsf.append(x)
    #     ysf.append(f(x))
    #     exsmooth = np.empty_like(ysf[-1])
    #     for j in range(len(ysf[-1])):
    #         if(j==0):
    #             val = ysf[-1][j]
    #         else:
    #             val = alpha * val + (1 - alpha) * ysf[-1][j]
    #         exsmooth[j] = val
    #     ysfe.append(exsmooth)

    # ysfe = np.array(ysfe)

    # # Plot
    # colors = [(197 / 255, 225 / 255, 111 / 255), (41 / 255, 126 / 255, 124 / 255)] #, (29 / 255, 52 / 255, 78 / 255)]
    # groups_idx = [[0,1,2], [3,4,5]]
    
    # ax = plt.subplot(1,1,1)
    # for idx, color in zip(groups_idx, colors):
    #     print(idx)
    #     mean = np.mean(ysfe[idx, :], axis=0)
    #     lower = np.min(ysfe[idx, :], axis=0)
    #     upper = np.max(ysfe[idx, :], axis=0)
    #     #print(mean)
    #     print(x.shape)
    #     ax.fill_between(x, lower, upper, color=color)
    #     plt.plot(x, mean, color=color)
    
    # plt.show()

    # plt.figure()
    # mean_series = np.empty_like(ysfe[-1])
    # upper = np.empty_like(ysfe[-1])
    # lower = np.empty_like(ysfe[-1])
    
    # for i in range(len(idx_g1)):
    #     mean_series


    # for i in range(len(xsf)):
    #     plt.plot(xsf[i], ysfe[i])
    # plt.show()