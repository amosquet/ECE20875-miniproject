import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def regression(x, y, degree_list):
    mse_list = []
    for degree in degree_list:
        poly_model = PolynomialFeatures(degree=degree)
        poly_x = poly_model.fit_transform(x)
        model = LinearRegression()
        model.fit(poly_x, y)
        y_pred = model.predict(poly_x)
        mse_list.append(root_mean_squared_error(y, y_pred))
    return mse_list

def best_regression(x, y, degree):
    poly_model = PolynomialFeatures(degree=degree)
    poly_x = poly_model.fit_transform(x)
    model = LinearRegression()
    model.fit(poly_x, y)
    y_pred = model.predict(poly_x)
    mse = root_mean_squared_error(y, y_pred)
    return y_pred, model

def split_dow(dataset):
    return [dataset[dataset.index % 7 == i] for i in range(7)]

def create_x_values(data):
    return np.arange(1, len(data) + 1).reshape(-1, 1)

def prepare_data(dataset, bridges, weather):
    bridge_data = {bridge: pd.to_numeric(dataset[bridge].replace(',', '', regex=True)) for bridge in bridges}
    weather_data = dataset[weather].apply(lambda x: pd.to_numeric(x.replace(',', '', regex=True)))
    return bridge_data, weather_data

def find_best_bridge_combination(bridge_data, total_traffic, bridges):
    best_combination = None
    best_mse = float('inf')
    for i in range(len(bridges)):
        for j in range(i + 1, len(bridges)):
            for k in range(j + 1, len(bridges)):
                selected_bridges = [bridges[i], bridges[j], bridges[k]]
                x = np.column_stack([bridge_data[bridge] for bridge in selected_bridges])
                y = total_traffic
                mse = regression(x, y, [1])[0]
                if mse < best_mse:
                    best_mse = mse
                    best_combination = selected_bridges
    return best_combination, best_mse

def plot_mse_vs_degree(degree_list, mse_list):
    plt.scatter(degree_list, mse_list, color="black")
    plt.plot(degree_list, mse_list, color="red")
    plt.title("Prediction Model Using Weather")
    plt.xlabel("Degree")
    plt.ylabel("MSE")
    plt.show()

def plot_dow_trends(dow_data, day_list):
    for i, day_data in enumerate(dow_data):
        x = create_x_values(day_data)
        y = day_data.values
        plt.scatter(x, y, color="black", label="Data Points")
        plt.plot(x, y, color="red", label="Trend")
        plt.ylabel("Number of Bicyclists")
        plt.xlabel("Day")
        plt.title(f"Number of Bicyclists on {day_list[i]}")
        plt.legend()
        plt.show()

# Main script
dataset = pd.read_csv('NYC_Bicycle_Counts_2016.csv')
bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']
weather = ['High Temp', 'Low Temp', 'Precipitation']

bridge_data, weather_data = prepare_data(dataset, bridges, weather)
total_traffic = sum(bridge_data.values())

best_combination, best_mse = find_best_bridge_combination(bridge_data, total_traffic, bridges)
print(f"Best combination of bridges: {best_combination} with MSE: {best_mse}")

x = weather_data.values
y = total_traffic
degree_list = [1, 2, 3, 4, 5]
mse_list = regression(x, y, degree_list)
best_degree = degree_list[mse_list.index(min(mse_list))]
predicted, best_model = best_regression(x, y, best_degree)

plot_mse_vs_degree(degree_list, mse_list)
print(f"Best degree for weather prediction: {best_degree}")

dow_data = split_dow(total_traffic)
day_list = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
plot_dow_trends(dow_data, day_list)
