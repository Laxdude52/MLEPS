# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:26:38 2023

@author: every
"""

'''
Necessary Features

Get:
    Future Model Setting:
        1. plant levels
            a. Max ramp up/down
            b. other associated data
        2. elec demand pred
        3. solar output pred
    Current Logic Setting:
        1. plant levels
            a. Max ramp up/down
            b. other associated data
        2. real demand:
            real elec demand - real solar output (future) - other unit output (ex. battery, water)
 
Do:
    Future:
        Set future plant levels
    Logic:
        Set current plant levels

'''


import data as ud
import defaultSimulation as sim
import tensorflow as tf
import time
import logSimulation as los
import copy
import random
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

tf.saved_model.LoadOptions(experimental_io_device = '/job:localhost')
MAXTIME = 44316
solarModelInfo = ['default', 'Solar']

#Get current plant levels + max ramp up/down
def getAllPlantInformation():
    return ud.plantInformation

def getSpecPlantInformation(plant):
    return ud.plantInformation[plant]

#Get solar pred 
def getSolarPred(currentTime):
    #print(currentTime)
    return ud.livePredict(currentTime, 'Solar', 'Future', 'default', numPast=5)

def getCurrentSolar(currentTime):
    try:
        data = ud.dataModels['Solar']['Simple']['default']['initData'].y_train.iloc[[currentTime]]
    except:
        data = ud.dataModels['Solar']['Simple']['default']['initData'].y_test.iloc[[currentTime]]
    return data

#Get elecDemandPRed
def fgetElecDemand(currentTime):
    #if(currentTime <= 5):
        return ud.livePredict(currentTime, 'Demand', 'Future', 'default', numPast=5)
    #else:
    #    return ud.livePredict(currentTime-5, 'Demand', 'Future', 'default')

#Get real elec demand
def rgetElecDemand(currentTime):
    try:
        data = ud.dataModels['Demand']['Simple']['default']['initData'].y_train.iloc[[currentTime]]
    except:
        data = ud.dataModels['Demand']['Simple']['default']['initData'].y_test.iloc[[currentTime]]
    return data

def getCurrentProduction():
    tmpCurProd = 0
    for plant in sim.plantList:
        tmpPlantInfo = getSpecPlantInformation(plant)
        tmpCurProd = tmpCurProd + tmpPlantInfo["CurrentLevel"]
    return tmpCurProd

def resetSimulation():
    for plant in sim.plantList:
        ud.setPlantLevel(plant, ud.plantInformation[plant]["Min"])
    
#sim.initDefaultSimulation()
sim.initDefaultSimulation(load=True)

def getMostEfficent(rampDir, plant):
    mostEfficentVal = 90000000
    mostEfficentSpeed = 90000000
    i=0
    data = plant["efficencyData"].iloc[:,1] 
    #print("CurLevel" + str(plant["CurrentLevel"]))
    currentLevel = plant["CurrentLevel"]
    #print(plant)
    #print("CurLevel" + str(currentLevel))
    maxOut = plant["Max"]-1 
    if(maxOut < currentLevel+plant["MaxRamp"]):
        upperBound = maxOut
    else:
        upperBound = currentLevel+plant["MaxRamp"]
    if((plant['Min']+1) > currentLevel-plant["MaxRamp"]):
        lowerBound = plant['Min']+1
    else:
        lowerBound = currentLevel-plant["MaxRamp"]
    if (rampDir == 1):
        #print("Cur: " + str(currentLevel) + "\nUppB: " + str(upperBound))
        for i in range(currentLevel, upperBound):
            if(data[i] < mostEfficentVal):
                mostEfficentVal = data[i]
                mostEfficentSpeed = i
        #print("E level" + str(mostEfficentSpeed))
    elif (rampDir == -1):
        #print("Down")
        #print("CurLevel: (Loop) " + str(currentLevel))
        #print("lowBound: (Loop) " + str(lowerBound))

        for i in range(lowerBound, currentLevel):
            if(data[i] < mostEfficentVal):
                mostEfficentVal = data[lowerBound]
                mostEfficentSpeed = lowerBound
    else:
        print(rampDir)
        mostEfficentSpeed = currentLevel
        #raise Exception("No dir")
    if(mostEfficentSpeed >= plant["Max"]):
        mostEfficentSpeed = plant["Max"]-1
    elif(mostEfficentSpeed <= plant["Min"]):
        mostEfficentSpeed = plant["Min"]+1
    #print("Plant new level: " + str(mostEfficentSpeed) + " Plant max: " + str(plant['Max']))
    heatRate = plant["efficencyData"].iloc[:,0][mostEfficentSpeed]
    return mostEfficentSpeed, heatRate, mostEfficentVal
    
def futureAlgorithm(currentTime):
    random.shuffle(sim.plantList)
    tmpElecDemandPred = int(fgetElecDemand(currentTime))
    #print("Pred elec demand: " + str(tmpElecDemandPred))
    
    tmpSolarPred = int(getSolarPred(currentTime))
    #print("Pred solar out: " + str(tmpSolarPred))
    
    for plant in sim.plantList:
        tmpCurProd = getCurrentProduction()
        #print("Current Production: " + str(tmpCurProd))
        
        tmpDemand = tmpElecDemandPred-tmpSolarPred-tmpCurProd
        tmpPlantInfo = getSpecPlantInformation(plant)
        #print(str(plant) + " Max Ramp: " + str(tmpPlantInfo["MaxRamp"]) + " Current Level: " + str(tmpPlantInfo["CurrentLevel"]))
        tmpGap = tmpDemand - tmpCurProd
        #print("Demand: " + str(tmpDemand))
        #print("Production: " + str(tmpCurProd))
        rampDir = 0
        if(tmpGap > 0):
            rampDir = 1
            #print("Plant: " + str(plant) + " up")
        elif(tmpGap < 0):   
            rampDir = -1
            #print("Plant: " + str(plant) + " down")
        else:
            #raise Exception("Bro")
            print("Pass")
            pass
        mostEfficentSpeed, hR, efVal = getMostEfficent(rampDir, tmpPlantInfo)
        change = abs(int(tmpPlantInfo['CurrentLevel']-mostEfficentSpeed))
        #print("Change: " + str(change))
        #print("Old: " + str(tmpPlantInfo['CurrentLevel']) + " New: " + str(mostEfficentSpeed))
        ud.setPlantLevel(plant, mostEfficentSpeed)
        ud.updatePlantRampLeft(plant, change)
        #print("Newlevel" + str(tmpPlantInfo['CurrentLevel']))
        los.logFutStep(plant,mostEfficentSpeed,hR,efVal)
        #time.sleep(1)
    los.logAllFutStep(currentTime, tmpElecDemandPred, tmpSolarPred)

def currentAlgorithm(currentTime):
    random.shuffle(sim.plantList)
    tmpRealDemand = int(rgetElecDemand(currentTime))
    #print("Real elec demand: " + str(tmpRealDemand))
    tmpRealSolar = int(getCurrentSolar(currentTime))
        
    for plant in sim.plantList:
        tmpPlantInfo = getSpecPlantInformation(plant)
        tmpCurProd = getCurrentProduction()  
        tmpDem = tmpRealDemand-tmpRealSolar-tmpCurProd
        #print(str(plant) + " Max Ramp: " + str(tmpPlantInfo["MaxRamp"]) + "  Current Level: " + str(tmpPlantInfo["CurrentLevel"]))
        tmpGap = tmpDem - tmpCurProd
        #print("Gap: " + str(tmpGap))
        #print("Left Ramp" + str(tmpPlantInfo["LeftRamp"]))
        if(abs(tmpGap) < tmpPlantInfo["LeftRamp"]):
            newLevel = tmpPlantInfo["CurrentLevel"] + tmpGap
        else:
            if(tmpGap > 0):
                newLevel = tmpPlantInfo['CurrentLevel'] + tmpPlantInfo['LeftRamp']
            elif(tmpGap < 0):
                newLevel = tmpPlantInfo['CurrentLevel'] - tmpPlantInfo['LeftRamp']
            else:
                newLevel = tmpPlantInfo['CurrentLevel']
        if(newLevel >= tmpPlantInfo["Max"]):
            newLevel = tmpPlantInfo["Max"]-1
        elif(newLevel <= tmpPlantInfo["Min"]):
            newLevel = tmpPlantInfo["Min"]+1
        ud.setPlantLevel(plant, newLevel)
        ud.resetPlantRampLeft(plant)
        #print("Live level: " + str(getSpecPlantInformation(plant)["CurrentLevel"]))
        try:
            heat = tmpPlantInfo["efficencyData"].iloc[:,0][newLevel]
        except Exception as e:
            print(e)
            print(newLevel)
            print(tmpPlantInfo['Max'])
            print(plant)
            Exception(e)
        heatPerOut = tmpPlantInfo["efficencyData"].iloc[:,1][newLevel]
        los.logCurStep(plant, newLevel,heat,heatPerOut)
    los.logCurFutStep(currentTime, tmpRealDemand, tmpRealSolar)
 
'''
xVals = []
yVals = []
fig = plt.figure()

def liveGraph(xVals, yVals):
    ax1 = fig.add_subplot(1,1,1)
    xVals.append(currentTime)
    yVals.append(los.logFutDict["totalProd"][currentTime])
    xVals = xVals[-30:]
    yVals = yVals[-30:]
    ax1.clear()
    ax1.plot(xVals, yVals)  
    
    plt.title("Production over time (Future)")

'''
xVals = []
production = []
predDemand = []
realDemand = []
plt.ion()
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(xVals, production, label="Total Production")
line2, = ax.plot(xVals, predDemand, label="Predicted Electricity Demand")
line3, = ax.plot(xVals, realDemand, label="Real Electricity Demand")
def livePlot(xVals,production,predDemand,realDemand,currentTime):
    xVals.append(currentTime)
    production.append(los.logCurDict["totalProd"][currentTime])
    #print("CurPlotProd: " + str(los.logCurDict["totalProd"][currentTime]))
    predDemand.append(los.logFutDict["elecDem"][currentTime])
    realDemand.append(los.logCurDict["elecDem"][currentTime])

    '''
    xVals = xVals[-72: ]
    production = production[-72: ]
    predDemand = predDemand[-72: ]
    realDemand = realDemand[-72: ]
    '''

    # Set x-axis limit to show a specific range of x values
    if(currentTime <= 72):
        ax.set_xlim(min(xVals), max(xVals))
    else:
        ax.set_xlim(currentTime - 72, currentTime)

    # Set y-axis limit to show a specific range of y values
    ax.set_ylim(0, 4000)

    #print("Lengths - xVals:", len(xVals), "production:", len(production), "predDemand:", len(predDemand), "realDemand:", len(realDemand))

    # Update x-data
    line1.set_xdata(xVals)
    line2.set_xdata(xVals)
    line3.set_xdata(xVals)

    # Update y-data
    line1.set_ydata(production)
    line2.set_ydata(predDemand)
    line3.set_ydata(realDemand)

    plt.title("Production, Predicted Demand, Real Demand (MWh)")
    plt.legend(loc="upper left")
    figure.canvas.draw()
    figure.canvas.flush_events()
    #plt.pause(0.1)
    #time.sleep(.1)
    #https://www.youtube.com/watch?v=Ercd-Ip5PfQ
    
resetSimulation()
currentTime = 0
while currentTime != MAXTIME:
    futureAlgorithm(currentTime)

    currentAlgorithm(currentTime)
    livePlot(xVals, production, predDemand, realDemand, currentTime)

    #print("Current Time: " + str(currentTime))
    currentTime = currentTime+1
