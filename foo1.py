import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def regression(x, y, degreeList):
    """
    Perform polynomial regression for each degree specified in degreeList and calculate the mean squared error.

    Parameters:
    x (array-like): The input data.
    y (array-like): The target values.
    degreeList (list of int): A list of polynomial degrees to fit.

    Returns:
    list: A list of mean squared errors for each polynomial degree.
    """
    pltMSE = []
    for deg in degreeList:
        polyModel = PolynomialFeatures(degree=deg)
        polyXVal = polyModel.fit_transform(x)
        regressionModel = LinearRegression()
        regressionModel.fit(polyXVal, y)
        yPredict = regressionModel.predict(polyXVal)
        pltMSE.append(root_mean_squared_error(y, yPredict))
    return pltMSE

def bestRegression(x, y, deg):
    polyModel = PolynomialFeatures(degree=deg)
    polyXVal = polyModel.fit_transform(x)
    regressionModel = LinearRegression()
    regressionModel.fit(polyXVal, y)
    yPredict = regressionModel.predict(polyXVal)
    pltMSE = root_mean_squared_error(y, yPredict)
    return yPredict, regressionModel

def splitDOW(dataset):
    dow = [dataset[dataset.index % 7 == i] for i in range(7)]
    return dow

def createXValues(data):
    x = np.arange(1, len(data) + 1).reshape(-1, 1)
    return x

# Importing Data and Creating Initial Variables
dataset_2 = pd.read_csv('NYC_Bicycle_Counts_2016.csv')
bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']
weather = ['High Temp', 'Low Temp', 'Precipitation']

# Prepare data
bridge_data = {bridge: pd.to_numeric(dataset_2[bridge].replace(',', '', regex=True)) for bridge in bridges}
weather_data = dataset_2[weather].apply(lambda x: pd.to_numeric(x.replace(',', '', regex=True)))

# Step 1: Identify the three bridges to install sensors on
total_traffic = sum(bridge_data.values())
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

print(f"Best combination of bridges: {best_combination} with MSE: {best_mse}")

# Step 2: Predict the total number of bicyclists based on the next day's weather forecast
x = weather_data.values
y = total_traffic
degreeList = [1, 2, 3, 4, 5]
pltMSE = regression(x, y, degreeList)
best_degree = degreeList[pltMSE.index(min(pltMSE))]
predicted, best_model = bestRegression(x, y, best_degree)

# Collect plots to display at the end
figures = []

fig, ax = plt.subplots()
ax.scatter(degreeList, pltMSE, color="black")
ax.plot(degreeList, pltMSE, color="red")
ax.set_title("Prediction Model Using Weather")
ax.set_xlabel("Degree")
ax.set_ylabel("MSE")
figures.append(fig)

print(f"Best degree for weather prediction: {best_degree}")
print(f"Mean Squared Error: {min(pltMSE)}")

# Step 3: Analyze and visualize the data to identify patterns or trends associated with specific days of the week
dow_data = splitDOW(total_traffic)
dayList = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

for i, day_data in enumerate(dow_data):
    x = createXValues(day_data)
    y = day_data.values
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="black", label="Data Points")
    ax.plot(x, y, color="red", label="Trend")
    ax.set_ylabel("Number of Bicyclists")
    ax.set_xlabel("Day")
    ax.set_title(f"Number of Bicyclists on {dayList[i]}")
    ax.legend()
    figures.append(fig)

# Display all collected plots at the end
for fig in figures:
    fig.show()

# keep plots open
plt.show()