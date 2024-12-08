import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import OrderedDict


def sampleSTDE(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))

def find_bridge_to_add_sensors(data, bridges):

    brookData = data['Brooklyn Bridge'].to_numpy()
    manData = data['Manhattan Bridge'].to_numpy()
    queenData = data['Queensboro Bridge'].to_numpy()
    willData = data['Williamsburg Bridge'].to_numpy()

    brookSTDE = sampleSTDE(brookData)
    manSTDE = sampleSTDE(manData)
    queenSTDE = sampleSTDE(queenData)
    willSTDE = sampleSTDE(willData)

    STDEList = [brookSTDE, manSTDE, queenSTDE, willSTDE]
    if max(STDEList) == brookSTDE:
        return 'Brooklyn Bridge'
    elif max(STDEList) == manSTDE:
        return 'Manhattan Bridge'
    elif max(STDEList) == queenSTDE:
        return 'Queensboro Bridge'
    elif max(STDEList) == willSTDE:
        return 'Williamsburg Bridge'
    else:
        return "Something went wrong"



def regression(data, bridges):

    x = data[bridges].values
    y = data['Total'].values

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(xTrain, yTrain)

    pred = model.score(xTest, yTest)
    return model, pred



def main():

    data = pd.read_csv('NYC_Bicycle_Counts_2016.csv')
    for bridge in ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge', 'Total']:
        data[bridge] = pd.to_numeric(data[bridge].replace(',', '', regex=True))

    data['Date'] = pd.to_datetime(data['Date'] + '-2016', format='%d-%b-%Y')
    data['Day of Week'] = data['Date'].dt.day_name()
    bridges = ['Brooklyn Bridge', 'Manhattan Bridge', 'Queensboro Bridge', 'Williamsburg Bridge']

    weatherData = data[['Low Temp', 'High Temp', 'Precipitation']]

    ############################################################################
    #Part 1: Finding Which Bridges to Add Sensors To
    ############################################################################
    print("Part 1")

    not_bridge = find_bridge_to_add_sensors(data, bridges)
    print(f"Put sensors on every bridge but the {not_bridge}")


    ############################################################################
    #Part 2: Can Weather Predict Bicyclist Numbers?
    ############################################################################
    print("\Part 2")

    bikerData = data['Total']

    weatherTrain, weatherTest, bikerTrain, bikerTest = train_test_split(weatherData, bikerData, test_size = 0.2, random_state = 2)

    model = LinearRegression()
    model.fit(weatherTrain, bikerTrain)
    bikerPrediction = model.predict(weatherTest)

    # r2_score = model.score(weatherTest, bikerTest)
    # print(f"R^2 score: {r2_score}")
    #R^2 correlation no worky, sadge. So we'll use something else instead

    correlation = np.corrcoef(bikerPrediction, bikerTest)[0, 1]
    print(f"Correlation: {correlation}")

    #graphing problem 2 data to visualize
    # plt.figure(figsize=(14, 6))
    # plt.subplot(1, 2, 1)
    # plt.scatter(weatherTest['High Temp'], bikerTest, color = 'blue', label = "High Temp")
    # plt.xlabel("High Temp")
    # plt.ylabel("Total Bikers")
    # plt.title("Correlation Between High Temp and Number of Bikers")
    # plt.legend()
    
    # plt.subplot(1, 2, 2)
    # plt.scatter(weatherTest['Low Temp'], bikerTest, color = 'blue', label = "Low Temp")
    # plt.xlabel("Low Temp")
    # plt.ylabel("Total Bikers")
    # plt.title("Correlation Between Low Temp and Number of Bikers")
    # plt.legend()
    # plt.show()


    ############################################################################
    # Part 3: Can you predict the day of the week based on the number of bicyclists?
    ############################################################################
    print("\nPart 3")

    #Define the order of the days of the week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    #Create an ordered dictionary where the keys are days of the week and values are lists with all of the totals with the corresponding day of the week
    dayDict = OrderedDict()
    for day in days_order:
        dayDict[day] = data[data['Day of Week'] == day]['Total'].values

    #Calculate the average number of bikers for each day of the week
    avgDict = OrderedDict()
    for day in dayDict:
        avgDict[day] = np.mean(dayDict[day])

    # rint the average number of bikers for each day of the week, easily see if there is an obvious pattern
    for day in avgDict:
        print(f"{day}: {avgDict[day]}")
    #from the data provided, it seems that the number of bikers is highest on Wednesday and lowest on Sunday.
    #there does seem to be a trend, but it is not very strong. This will likely not be a very accurate model.
    #some of the data has less than a 1000 biker difference between days, which is not a lot when the total number of bikers is in the tens of thousands.

    regression(data['Date'], data['Total'])

    correlation = np.corrcoef(bikerPrediction, bikerTest)[0, 1]
    print(f"Correlation: {correlation}")
    



if __name__ == "__main__":
    main()