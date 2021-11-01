from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


def fit_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def assess_model_performance(trained_model, X_test, y_test):
    return mean_absolute_error(y_test, trained_model.predict(X_test))
