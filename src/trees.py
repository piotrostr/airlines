import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np
import argparse

from data import get_data_split, get_XY
from sklearn.metrics import mean_squared_error
from xgboost import plot_tree


def plot_predictions(preds: np.ndarray, actual: np.ndarray):
    plt.figure()
    length = range(len(preds))
    plt.scatter(length, preds)
    plt.scatter(length, actual)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-plot', action='store_true')
    parser.add_argument('--show-tree', action='store_true')
    args = parser.parse_args()

    df_train, df_test = get_data_split()
    X_train, Y_train = get_XY(df_train)
    X_test, Y_test = get_XY(df_test)

    model = xgb.XGBRegressor()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)
    print("mse: ", mean_squared_error(Y_pred, Y_test))
    if args.show_plot:
        plot_predictions(Y_pred[:50], Y_test.values[:50])
        plt.show()
    if args.show_tree:
        plot_tree(model, rankdir='LR')
        plt.show()
