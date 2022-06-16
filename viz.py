import pandas as pd
import matplotlib as plt


def hist_chart(train):
    columns = ['bathroomcnt','bedroomcnt','calculatedfinishedsquarefeet','lotsizesquarefeet','yearbuilt','taxvaluedollarcnt','logerror']
    for col in columns:
        plt.figure(figsize=(4,2))
        plt.hist(train[col])
        plt.title(col)
        plt.show()