import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

def load_data(file_path):
    df = pd.read_csv(file_path)
    y = df["y"]
    X = df.drop(columns=["y"])
    return X, y

def build_model(X, y):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model

def filter_factors(X, y, model, significance_level=0.05):
    while True:
        p_values = model.pvalues
        if p_values.max() > significance_level:
            max_p_factor = p_values.idxmax()
            response = input(f"Удалить фактор {max_p_factor} (p-value = {p_values.max()}): (y/n) ")
            if response.lower() == 'y':
                X = X.drop(columns=[max_p_factor])
                model = build_model(X, y)
            else:
                break
        else:
            break
    return X, model

def check_correlation(X, y):
    correlation_matrix = X.corr()
    print("Корреляция между факторами:")
    print(correlation_matrix)
    print("\nКорреляция факторов с откликом y:")
    print(y.corr(X))

def compute_error(y, predictions):
    error = np.mean(np.abs((y - predictions) / y))
    return error

def evaluate_model(model, y):
    predictions = model.predict()
    r_squared = model.rsquared
    rmse = np.sqrt(mean_squared_error(y, predictions))
    f_statistic = model.fvalue
    f_p_value = model.f_pvalue
    error = compute_error(y, predictions)

    print(f"R^2: {r_squared}")
    print(f"RMSE: {rmse}")
    print(f"Ошибка E: {error}")
    print(f"F-статистика: {f_statistic}")
    print(f"p-значение для F-статистики: {f_p_value}")

    return r_squared, rmse, f_statistic, f_p_value, error

def make_predictions(model, new_factors):
    predictions = model.predict(sm.add_constant(new_factors))
    return predictions

if __name__ == "__main__":
    input_file = "input.csv"
    prediction_file = "new_factors.csv"
    output_file = "results.txt"

    X, y = load_data(input_file)
    model = build_model(X, y)
    print(model.summary())

    X_filtered, model_filtered = filter_factors(X, y, model)
    print(model_filtered.summary())

    r_squared, rmse, f_statistic, f_p_value, error = evaluate_model(model_filtered, y)

    predictions = None
    if os.path.exists(prediction_file):
        print(f"Файл с новыми факторами найден: {prediction_file}")
        new_factors = pd.read_csv(prediction_file)
        print(f"Новые факторы: {new_factors.head()}")
        predictions = make_predictions(model_filtered, new_factors)
        print("Предсказания:", predictions.tolist())
    else:
        print(f"Файл с новыми факторами не найден: {prediction_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(model_filtered.summary().as_text())
        f.write(f"\nR^2: {r_squared}\nRMSE: {rmse}\n")
        f.write(f"Ошибка E: {error}\n")
        f.write(f"F-статистика: {f_statistic}\n")
        f.write(f"p-значение для F-статистики: {f_p_value}\n")
        if predictions is not None:
            f.write(f"Предсказания: {predictions.tolist()}\n")