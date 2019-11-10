import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df_false = pd.read_csv('false.csv', header=None)
    df_true =  pd.read_csv('true.csv', header=None)

    plt.figure()
    plt.plot(df_true.values)
    plt.show()