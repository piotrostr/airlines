import statsmodels.api as sm

from data import get_data_split, get_XY
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    df_train, df_test = get_data_split()
    X_train, Y_train = get_XY(df_train)
    X_test, Y_test = get_XY(df_test)
    model = sm.OLS(Y_train, X_train)
    res = model.fit()
    print(res.summary())
    Y_preds = res.predict(X_test)
    print("mse: ", mean_squared_error(Y_preds, Y_test))
