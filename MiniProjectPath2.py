import pandas
import numpy as np

''' 
The following is the starting code for path2 for data reading to make your first step easier.
'dataset_2' is the clean data for path1.
'''
dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv') #Extracting CSV Data
dataset_2['Brooklyn Bridge'] = pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True)) #These 5 lines allow us to isolate and access bridge specific data
dataset_2['Manhattan Bridge'] = pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True))
dataset_2['Queensboro Bridge'] = pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge'] = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
dataset_2['Williamsburg Bridge'] = pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True))
# print(dataset_2.to_string()) #This line will print out your data

#Dataset 2 is an Array showing the data from the CSV with 214 rows of data and columns based on the CSV



############################################################################
#Part 1: Finding Which Bridges to Add Sensors To
############################################################################

# Create individual numpy arrays for each bridge's data
brookData = dataset_2['Brooklyn Bridge'].to_numpy()
manData = dataset_2['Manhattan Bridge'].to_numpy()
queenData = dataset_2['Queensboro Bridge'].to_numpy()
willData = dataset_2['Williamsburg Bridge'].to_numpy()

#Find the standard error of each bridge's dataset
def sampleSTDE(data):
    return np.std(data, ddof=1) / np.sqrt(len(data))

brookSTDE = sampleSTDE(brookData)
manSTDE = sampleSTDE(manData)
queenSTDE = sampleSTDE(queenData)
willSTDE = sampleSTDE(willData)

#Find the bridge with the highest STDE
STDEList = [brookSTDE, manSTDE, queenSTDE, willSTDE]
if max(STDEList) == brookSTDE:
    print('Put sensors on every bridge but the Brooklyn Bridge')
elif max(STDEList) == manSTDE:
    print('Put sensors on every bridge but the Manhattan Bridge')
elif max(STDEList) == queenSTDE:
    print('Put sensors on every bridge but the Queensboro Bridge')
elif max(STDEList) == willSTDE:
    print('Put sensors on every bridge but the Williamsburg Bridge')
else:
    print("Something went wrong")


############################################################################
#Part 2: Can Weather Predict Bicyclist Numbers?
############################################################################
# The city administration is cracking down on helmet laws, and wants to deploy police officers on days with high traffic to hand out citations.
# Can they use the next day's weather forecast(low/high temperature and precipitation) to predict the total number of bicyclists that day? 

#create a linear regression model to predict the total number of bicyclists based on the next day's weather forecast

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#Extracting the data from the CSV
dataset_2['High Temp'] = pandas.to_numeric(dataset_2['High Temp'].replace(',','', regex=True))
dataset_2['Low Temp'] = pandas.to_numeric(dataset_2['Low Temp'].replace(',','', regex=True))
dataset_2['Precipitation'] = pandas.to_numeric(dataset_2['Precipitation'].replace(',','', regex=True))
dataset_2['Total'] = pandas.to_numeric(dataset_2['Total'].replace(',','', regex=True))

weatherData = dataset_2[['High Temp', 'Low Temp', 'Precipitation']]
bikerData = dataset_2['Total']

#Splitting the data into training and testing data
weatherTrain, weatherTest, bikerTrain, bikerTest = train_test_split(weatherData, bikerData, test_size=0.2)

#Creating the model
model = LinearRegression()
model.fit(weatherTrain, bikerTrain)

#Predicting the data
bikerPrediction = model.predict(weatherTest)

#Printing the model's accuracy, r^2
r2 = model.score(weatherTest, bikerTest)

print("coefficient of determination: " + r2)
if r2 > 0.5:
    print('The model is accurate')
else:
    print('The model is not accurate')
