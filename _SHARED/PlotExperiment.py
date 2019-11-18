import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


PARAMETER_NAME = 'parameter'
REP_NAME = 'rep'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', required=False, type=str, default=os.path.dirname(__file__))
    parser.add_argument('--trackname', required=False, type=str, default='validation_reward.csv')
    parser.add_argument('--esmooth', required=False, type=float, default=0.6)

    parser.add_argument('--xmin', required=False, type=int, default=0)
    parser.add_argument('--xmax', required=False, type=int, default=100000000)
    parser.add_argument('--ylower', required=False, type=float, default=None)
    parser.add_argument('--yupper', required=False, type=float, default=None)
    parser.add_argument('--title', required=False, type=str, default=None)
    parser.add_argument('--xlabel', required=False, type=str, default=None)
    parser.add_argument('--ylabel', required=False, type=str, default=None)
    parser.add_argument('--xmult', required=False, type=int, default=1)

    args = parser.parse_args()

    # Load all data
    conts = os.listdir(args.datapath)
    dfs = []
    for cont in conts:
        path_folder = os.path.join(args.datapath, cont)
        path_tracks = os.path.join(path_folder, 'tracks')
        if(not os.path.isdir(path_tracks)):
            print('Ignoring "%s", it is not an output folder.' % path_folder)
            continue

        path_file = os.path.join(path_tracks, args.trackname)
        if(not os.path.isfile(path_file)):
            print('ERROR: There was not file at "%s". Aborting.' % path_file)
            sys.exit(1)

        df = pd.read_csv(path_file, header=None)
        df.columns = ['raw']

        alpha = args.esmooth
        filtered = np.array(df.values, copy=True)
        for i in range(len(filtered)):
            if(i == 0):
                filtered[i] = 0
            else:
                filtered[i] = alpha * filtered[i - 1] + (1 - alpha) * filtered[i]
        df['smooth'] = filtered

        splits = cont.split('_')
        arr_value = np.ones(shape=(df.shape[0],)) * float(splits[1]) / 100
        arr_rep = np.ones(shape=(df.shape[0],), dtype=int) * int(splits[2])
        df[PARAMETER_NAME] = arr_value
        df[REP_NAME] = arr_rep

        dfs.append(df)
    df = pd.concat(dfs)
    df['index1'] = df.index

    # Plot
    colors = [(197 / 255, 225 / 255, 111 / 255), (41 / 255, 126 / 255, 124 / 255), (29 / 255, 52 / 255, 78 / 255)]
    pgroup = df.groupby([PARAMETER_NAME, 'index1'])
    what = pgroup.agg(['mean', 'median', 'std', 'min', 'max', lambda x: np.percentile(x, q=20), lambda x: np.percentile(x, q=80)])
    what.reset_index(0, inplace=True)
    unqs = what[PARAMETER_NAME].unique()
    plt.figure()
    ax = plt.subplot(1, 1, 1)
    legends = []
    for i in range(len(unqs)):
        val = unqs[i]
        dfsub = what[what[PARAMETER_NAME] == val]
        if(dfsub.shape[0] > args.xmax):
            endpoint = args.xmax + 1
            ys = dfsub[('smooth', 'median')].values[args.xmin:endpoint]
            xs = np.arange(args.xmin, endpoint, 1)
            zs = dfsub[('smooth', 'std')].iloc[args.xmin:endpoint, ]
            lowers = dfsub[('smooth', '<lambda_0>')].iloc[args.xmin:endpoint, ]
            uppers = dfsub[('smooth', '<lambda_1>')].iloc[args.xmin:endpoint, ]
        else:
            ys = dfsub[('smooth', 'median')].values[args.xmin:]
            xs = np.arange(args.xmin, dfsub.shape[0], 1)
            zs = dfsub[('smooth', 'std')].iloc[args.xmin:, ]
            lowers = dfsub[('smooth', '<lambda_0>')].iloc[args.xmin:, ]
            uppers = dfsub[('smooth', '<lambda_1>')].iloc[args.xmin:, ]
        ax.plot(args.xmult * xs, ys, color=colors[i])
        ax.fill_between(args.xmult * xs, lowers, uppers, alpha=0.3, color=colors[i])
        legends.append(str(val))

    ax.grid()
    ax.set_axisbelow(True)
    plt.legend(legends)
    plt.ylim([args.ylower, args.yupper])
    plt.title(args.title)
    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.show()
