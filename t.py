import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import locale

def regression(x,y,degreeList,df):
    degree = 3
    polyModel = PolynomialFeatures(degree = degree)
    polyXVal = polyModel.fit_transform(x)

    polyModel.fit(polyXVal,y)
    regressionModel = LinearRegression()
    regressionModel.fit(polyXVal,y)
    yPredict = regressionModel.predict(polyXVal)
    regressionModel.coef_
    mean_squared_error(y,yPredict,squared=False)

    pltMSE = []
    for deg in degreeList:
        polyModel = PolynomialFeatures(degree = deg)
        polyXVal = polyModel.fit_transform(x)
        polyModel.fit(polyXVal,df)
   
        regressionModel = LinearRegression()
        regressionModel.fit(polyXVal,y)
        yPredict = regressionModel.predict(polyXVal)

        pltMSE.append(mean_squared_error(y,yPredict,squared=False))

    return pltMSE

def bestRegression(x,y,df,deg):
    polyModel = PolynomialFeatures(degree = deg)
    polyXVal = polyModel.fit_transform(x)

    polyModel.fit(polyXVal,y)
    regressionModel = LinearRegression()
    regressionModel.fit(polyXVal,y)
    yPredict = regressionModel.predict(polyXVal)
    regressionModel.coef_
    pltMSE = mean_squared_error(y,yPredict,squared=False)

    polyModel = PolynomialFeatures(degree = deg)
    polyXVal = polyModel.fit_transform(x)
    polyModel.fit(polyXVal,df)
   
    regressionModel = LinearRegression()
    regressionModel.fit(polyXVal,y)
    yPredict = regressionModel.predict(polyXVal)

    pltMSE=(mean_squared_error(y,yPredict,squared=False))
    return yPredict, regressionModel

def splitDOW(dataset):
    count = 6
    sun = []
    mon = []
    tues = []
    wed = []
    thur = []
    fri = []
    sat = []
    for d in dataset:
        if type(d) == str:
            print(d)
            continue
        if count == 1:
            sun.append(d)
        elif count == 2:
            mon.append(d)
        elif count == 3:
            tues.append(d)
        elif count == 4:
            wed.append(d)
        elif count == 5:
            thur.append(d)
        elif count == 6:
            fri.append(d)
        elif count == 7:
            sat.append(d)
            count = 0
        count +=1
   
    dow = [pandas.to_numeric(sun),pandas.to_numeric(mon),pandas.to_numeric(tues),pandas.to_numeric(wed),pandas.to_numeric(thur),pandas.to_numeric(fri),pandas.to_numeric(sat)]
    return dow

def dowDFFilter(df):
    friFilter = (df.index+1)%7==0
    satFilter = (df.index+1)%7==1
    sunFilter = (df.index+1)%7==2
    monFilter = (df.index+1)%7==3
    tuesFilter = (df.index+1)%7==4
    wedFilter = (df.index+1)%7==5
    thursFilter = (df.index+1)%7==6

    friDF = df[friFilter]
    satDF = df[satFilter]
    sunDF = df[sunFilter]
    monDF = df[monFilter]
    tuesDF = df[tuesFilter]
    wedDF = df[wedFilter]
    thursDF = df[thursFilter]

    dowDF = [sunDF,monDF,tuesDF,wedDF,thursDF,friDF,satDF]

    return dowDF

def createXValues(data):
    x = data
    for i in range(0,len(data)):
        x[i] = i+1
    x = np.reshape(x,(-1,1))
    return x

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Importing Data and Creating Initial Variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

dataset_2 = pandas.read_csv('NYC_Bicycle_Counts_2016.csv')
brooklyn = np.array(pandas.to_numeric(dataset_2['Brooklyn Bridge'].replace(',','', regex=True)))
manhattan = np.array(pandas.to_numeric(dataset_2['Manhattan Bridge'].replace(',','', regex=True)))
queensboro = np.array(pandas.to_numeric(dataset_2['Queensboro Bridge'].replace(',','', regex=True)))
williamsburg = np.array(pandas.to_numeric(dataset_2['Williamsburg Bridge'].replace(',','', regex=True)))
lowTemp = np.array(pandas.to_numeric(dataset_2['High Temp'].replace(',','', regex=True)))
highTemp = np.array(pandas.to_numeric(dataset_2['Low Temp'].replace(',','', regex=True)))
precip = np.array(pandas.to_numeric(dataset_2['Precipitation'].replace(',','', regex=True)))
day = np.array(dataset_2['Day'].replace(',','', regex=True))

brkDF = dataset_2[['Brooklyn Bridge','High Temp','Low Temp','Precipitation']].dropna()
manDF = dataset_2[['Manhattan Bridge','High Temp','Low Temp','Precipitation']].dropna()
queenDF = dataset_2[['Queensboro Bridge','High Temp','Low Temp','Precipitation']].dropna()
willDF = dataset_2[['Williamsburg Bridge','High Temp','Low Temp','Precipitation']].dropna()

xValues = brkDF[['High Temp','Low Temp','Precipitation']].values

brkData = dataset_2[['Brooklyn Bridge']].dropna()
manData = dataset_2[['Manhattan Bridge']].dropna()
queenData = dataset_2[['Queensboro Bridge']].dropna()
willData = dataset_2[['Williamsburg Bridge']].dropna()
dowBrkDf = dowDFFilter(brkData)
dowManDf = dowDFFilter(manData)
dowQueenDf = dowDFFilter(queenData)
dowWillDf = dowDFFilter(willData)


yValuesBrk = pandas.to_numeric(brkDF['Brooklyn Bridge'].replace(',','', regex=True)).values
yValuesMan = pandas.to_numeric(manDF['Manhattan Bridge'].replace(',','', regex=True)).values
yValuesQueen = pandas.to_numeric(queenDF['Queensboro Bridge'].replace(',','', regex=True)).values
yValuesWill = pandas.to_numeric(willDF['Williamsburg Bridge'].replace(',','', regex=True)).values

dayList = ["Sundays","Mondays","Tuesdays","Wednesdays","Thursdays","Fridays","Saturdays"]

degreeList = [1,2,3,4,5]
xAxis = np.linspace(1,len(williamsburg),len(williamsburg))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Running and Predicting Brooklyn Bridge Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pltMSEBrk = regression(xValues,yValuesBrk,degreeList,brkDF)
bestMSEBrk = pltMSEBrk.index(min(pltMSEBrk))
#print(bestMSEBrk)
plt.scatter(degreeList,pltMSEBrk,color="black")
plt.plot(degreeList,pltMSEBrk,color="red")
plt.title("Brooklyn Prediction Model Using Weather")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.show()

predictedBrk, bestModelBrk = bestRegression(xValues,yValuesBrk,brkDF,bestMSEBrk)
coeffBrk = bestModelBrk.coef_
interBrk = bestModelBrk.intercept_
print('Brooklyn Best Model Coefficients: ',coeffBrk)
print('Brooklyn Best Model Intercept: ',interBrk)
plt.scatter(xAxis,brooklyn,color="black",label="Data Points")
plt.plot(xAxis,predictedBrk,color="red",label="Predicted Model")
plt.ylabel("Number of Bicylists")
plt.xlabel("Day")
plt.title("Predicted Vs Actual Bikers on Brooklyn Bridge")
plt.legend()
plt.show()

yValDowBrk = splitDOW(yValuesBrk)
print(yValDowBrk[0])
for i in range(0,7):
    index = i+1
    if index>6:
        index = 0
    dayStr = dayList[i]
    data = yValDowBrk[i]
    df = dowBrkDf[index]
    dowBrk = pandas.to_numeric(df['Brooklyn Bridge'].replace(',','', regex=True)).values
    dowXValues = createXValues(dowBrk)
    # print(dayStr)
    # print("data")
    # print(data)
    # print("df")
    # print(df)
    # print("Lengths y,x")
    # print(len(data))
    # print(len(dowXValues))
    # print("=-=-=-=-=")
    predictBrkSun, bestBrkSun = bestRegression(dowXValues,data,df,bestMSEBrk)
    plt.scatter(dowXValues,data,color="black",label="Data Points")
    plt.plot(dowXValues,predictBrkSun,color="red",label="Predicted Model")
    plt.ylabel("Number of Bicylists")
    plt.xlabel("Day")
    plt.title("Predicted Vs Actual Bikers on Brooklyn Bridge on "+dayStr)
    plt.legend()
    plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Running and Manhattan Brooklyn Bridge Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pltMSEMan = regression(xValues,yValuesMan,degreeList,manDF)
bestMSEMan = pltMSEMan.index(min(pltMSEMan))
print(bestMSEMan)
plt.scatter(degreeList,pltMSEMan,color="black")
plt.plot(degreeList,pltMSEMan,color="red")
plt.title("Manhattan Prediction Model Using Weather")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.show()
predictedMan, bestModelMan = bestRegression(xValues,yValuesMan,manDF,bestMSEMan)
coeffMan = bestModelMan.coef_
interMan = bestModelMan.intercept_
print('Manhattan Best Model Coefficients: ',coeffMan)
print('Manhattan Best Model Intercept: ',interMan)
plt.scatter(xAxis,manhattan,color="black",label="Data Points")
plt.plot(xAxis,predictedMan,color="red",label="Predicted Model")
plt.ylabel("Number of Bicylists")
plt.xlabel("Day")
plt.title("Predicted Vs Actual Bikers on Manhattan Bridge")
plt.legend()
plt.show()

yValDowMan = splitDOW(yValuesMan)
#print(yValDowMan[0])
for i in range(0,7):
    index = i+1
    if index>6:
        index = 0
    dayStr = dayList[i]
    data = yValDowMan[i]
    df = dowManDf[index]
    dowMan = pandas.to_numeric(df['Manhattan Bridge'].replace(',','', regex=True)).values
    dowXValues = createXValues(dowMan)
    predictManDow, bestManDow = bestRegression(dowXValues,data,df,bestMSEMan)
    plt.scatter(dowXValues,data,color="black",label="Data Points")
    plt.plot(dowXValues,predictManDow,color="red",label="Predicted Model")
    plt.ylabel("Number of Bicylists")
    plt.xlabel("Day")
    plt.title("Predicted Vs Actual Bikers on Manhattan Bridge on "+dayStr)
    plt.legend()
    plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Running and Predicting Queensboro Bridge Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pltMSEQueen = regression(xValues,yValuesQueen,degreeList,queenDF)
bestMSEQueen = pltMSEQueen.index(min(pltMSEQueen))
print(bestMSEQueen)
plt.scatter(degreeList,pltMSEQueen,color="black")
plt.plot(degreeList,pltMSEQueen,color="red")
plt.title("Queensboro Prediction Model Using Weather")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.show()
predictedQueen, bestModelQueen = bestRegression(xValues,yValuesQueen,queenDF,bestMSEQueen)
coeffQueen = bestModelQueen.coef_
interQueen = bestModelQueen.intercept_
print('Queensboro Best Model Coefficients: ',coeffQueen)
print('Queensboro Best Model Intercept: ',interQueen)
plt.scatter(xAxis,queensboro,color="black",label="Data Points")
plt.plot(xAxis,predictedQueen,color="red",label="Predicted Model")
plt.ylabel("Number of Bicylists")
plt.xlabel("Day")
plt.title("Predicted Vs Actual Bikers on Queensboro Bridge")
plt.legend()
plt.show()

yValDowQueen = splitDOW(yValuesQueen)
#print(yValDowMan[0])
for i in range(0,7):
    index = i+1
    if index>6:
        index = 0
    dayStr = dayList[i]
    data = yValDowQueen[i]
    df = dowQueenDf[index]
    dowQueen = pandas.to_numeric(df['Queensboro Bridge'].replace(',','', regex=True)).values
    dowXValues = createXValues(dowQueen)
    predictQueenDow, bestQueenDow = bestRegression(dowXValues,data,df,bestMSEQueen)
    plt.scatter(dowXValues,data,color="black",label="Data Points")
    plt.plot(dowXValues,predictQueenDow,color="red",label="Predicted Model")
    plt.ylabel("Number of Bicylists")
    plt.xlabel("Day")
    plt.title("Predicted Vs Actual Bikers on Queensboro Bridge on "+dayStr)
    plt.legend()
    plt.show()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Running and Predicting Williamsburg Bridge Data
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pltMSEWill = regression(xValues,yValuesWill,degreeList,willDF)
bestMSEWill = pltMSEWill.index(min(pltMSEWill))
print(bestMSEWill)
plt.scatter(degreeList,pltMSEWill,color="black")
plt.plot(degreeList,pltMSEWill,color="red")
plt.title("Williamsburg Prediction Model Using Weather")
plt.xlabel("Degree")
plt.ylabel("MSE")
plt.show()
predictedWill, bestModelWill = bestRegression(xValues,yValuesWill,willDF,bestMSEWill)
coeffWill = bestModelWill.coef_
interWill = bestModelWill.intercept_
print('Williamsburg Best Model Coefficients: ',coeffWill)
print('Williamsburg Best Model Intercept: ',interWill)
plt.scatter(xAxis,williamsburg,color="black",label="Data Points")
plt.plot(xAxis,predictedWill,color="red",label="Predicted Model")
plt.ylabel("Number of Bicylists")
plt.xlabel("Day")
plt.title("Predicted Vs Actual Bikers on Williamsburg Bridge")
plt.legend()
plt.show()

yValDowWill = splitDOW(yValuesWill)
#print(yValDowMan[0])
for i in range(0,7):
    index = i+1
    if index>6:
        index = 0
    dayStr = dayList[i]
    data = yValDowWill[i]
    df = dowWillDf[index]
    dowWill = pandas.to_numeric(df['Williamsburg Bridge'].replace(',','', regex=True)).values
    dowXValues = createXValues(dowWill)
    predictWillDow, bestWillDow = bestRegression(dowXValues,data,df,bestMSEWill)
    plt.scatter(dowXValues,data,color="black",label="Data Points")
    plt.plot(dowXValues,predictWillDow,color="red",label="Predicted Model")
    plt.ylabel("Number of Bicylists")
    plt.xlabel("Day")
    plt.title("Predicted Vs Actual Bikers on Williamsburg Bridge on "+dayStr)
    plt.legend()
    plt.show()