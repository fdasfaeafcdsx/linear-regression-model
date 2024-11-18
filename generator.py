import pandas as pd
import numpy as np

def generate_input_data(filename):
    num_rows = 100
    num_factors = 5
    np.random.seed(42)
    factors = np.random.rand(num_rows, num_factors) * 10
    y = 3 * factors[:, 0] + 2 * factors[:, 1] + 1.5 * factors[:, 2] + 0.5 * factors[:, 3] + 0.7 * factors[:, 4] + np.random.randn(num_rows) * 2
    df = pd.DataFrame(factors, columns=[f"x{i+1}" for i in range(num_factors)])
    df["y"] = y
    df.to_csv(filename, index=False)
    print(f"Файл {filename} успешно сгенерирован.")
def generate_new_factors(filename):
    num_rows = 5
    num_factors = 5
    np.random.seed(99)
    new_factors = np.random.rand(num_rows, num_factors) * 10
    df = pd.DataFrame(new_factors, columns=[f"x{i+1}" for i in range(num_factors)])
    df.to_csv(filename, index=False)
    print(f"Файл {filename} успешно сгенерирован.")
input_file = "input.csv"
new_factors_file = "new_factors.csv"
generate_input_data(input_file)
generate_new_factors(new_factors_file)
