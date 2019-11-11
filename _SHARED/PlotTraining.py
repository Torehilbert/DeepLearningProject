import pandas as pd
import matplotlib.pyplot as plt
import os


def PlotReward(series, title='Rewards'):
    plt.figure()
    plt.plot(series, color='gray', lw='0.25')
    plt.axhline(y=200, ls=':', lw=1, color=[1, 0, 0, 0.4])
    plt.axhline(y=0, ls='-', lw=0.25, color='black')
    plt.title('Rewards')
    plt.xlabel('Iteration')
    plt.ylabel('Total reward (episode)')


def PlotLosses(array_of_series, loss_names):
    plt.figure(figsize=(4, 10))
    for i in range(len(array_of_series)):
        plt.subplot(len(array_of_series), 1, i + 1)
        plt.title(loss_names[i])
        plt.plot(array_of_series[i], color='gray', lw='0.25')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
    plt.tight_layout()


def PlotGradientNorm(series, clip_size):
    plt.figure()
    plt.plot(series, color='gray', lw='0.25')
    plt.axhline(y=clip_size, ls=':', lw=1, color=[1, 0, 0, 0.4])
    plt.legend(['Gradient L2 norm', 'Clip boundary'])


def LoadCSVSeries(folder, name, extension='.csv'):
    path = os.path.join(folder, name + extension)
    series = pd.read_csv(path, header=None).values
    return series


if __name__ == "__main__":
    folder = r"C:\Source\DeepLearningProject\Outputs\output DEBUG\tracks"

    reward_train = LoadCSVSeries(folder, 'train_reward')
    loss_total = LoadCSVSeries(folder, 'loss_total')
    loss_actor = LoadCSVSeries(folder, 'loss_actor')
    loss_critic = LoadCSVSeries(folder, 'loss_critic')
    loss_entropy = LoadCSVSeries(folder, 'loss_entropy')

    PlotReward(reward_train, title='Total training reward')
    PlotLosses([
        loss_total,
        loss_actor,
        loss_critic,
        loss_entropy],
        ['Loss (Total)',
         'Loss (Actor)',
         'Loss (Critic)',
         'Loss (Entropy)']
    )
    plt.show()

    # PlotReward(LoadCSVSeries(folder, 'train_reward'), show=False, hbars=True)
    # PlotReward(LoadCSVSeries(folder, 'loss_total'), show=False)
    # PlotReward(LoadCSVSeries(folder, 'loss_actor'), show=False)
    # PlotReward(LoadCSVSeries(folder, 'loss_critic'))
