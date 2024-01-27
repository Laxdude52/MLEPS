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
import time
import logSimulation as los

STARTTIME = 0
currentTime = 0 
solarModelInfo = ['default', 'Solar']

#Get current plant levels + max ramp up/down
def getAllPlantInformation():
    return ud.plantInformation

def getSpecPlantInformation(plant):
    return ud.plantInformation[plant]

#Get solar pred 
def getSolarPred():
    return ud.livePredict(currentTime, 'Solar', 'Future', 'default')

def getCurrentSolar():
    data = ud.dataModels['Solar']['Simple']['default']['initData'].y_train.iloc[[currentTime]]
    return data    

#Get elecDemandPRed
def fgetElecDemand():
    return ud.livePredict(currentTime, 'Demand', 'Future', 'default')

#Get real elec demand
def rgetElecDemand():
    data = ud.dataModels['Demand']['Simple']['default']['initData'].y_train.iloc[[currentTime]]
    return data

def getCurrentProduction():
    tmpCurProd = 0
    for plant in sim.plantList:
        tmpPlantInfo = getSpecPlantInformation(plant)
        tmpCurProd = tmpCurProd + tmpPlantInfo["CurrentLevel"]
    return tmpCurProd

def setPlantLevel(plant, level):
    ud.plantInformation[plant]["CurrentLevel"] = level
    
sim.initDefaultSimulation()

def getMostEfficent(rampDir, plant):
    mostEfficentVal = 9*10^3
    mostEfficentSpeed = 9*10^3
    data = plant["efficencyData"].iloc[:,1] 
    currentLevel = plant["CurrentLevel"]
    if (rampDir == 1):
        #Error here, can go above absolute max output, use additional logic to fix
        for i in range(currentLevel,int(currentLevel + plant["MaxRamp"])):
            if(data[i] < mostEfficentVal):
                mostEfficentVal = data[i]
                mostEfficentSpeed = i
        heatRate = plant["efficencyData"].iloc[:,0][mostEfficentSpeed]
        return mostEfficentSpeed, heatRate, mostEfficentVal
    elif (rampDir == -1):
        for i in range(currentLevel, int(currentLevel-plant["MaxRamp"])):
            if(data[i] < mostEfficentVal):
                mostEfficentVal = data[i]
                mostEfficentSpeed = i
        heatRate = plant["efficencyData"].iloc[:,0][mostEfficentSpeed]
        return mostEfficentSpeed, heatRate, mostEfficentVal
    
def futureAlgorithm(currentTime):
    tmpElecDemandPred = fgetElecDemand()
    print("Pred elec demand: " + str(tmpElecDemandPred))
    
    tmpSolarPred = getSolarPred()
    print("Pred solar out: " + str(tmpSolarPred))
    
    for plant in sim.plantList:
        tmpCurProd = getCurrentProduction()
        print("Current Production: " + str(tmpCurProd))
        
        tmpDemand = tmpElecDemandPred-tmpSolarPred-tmpCurProd
        tmpPlantInfo = getSpecPlantInformation(plant)
        #print(str(plant) + " Max Ramp: " + str(tmpPlantInfo["MaxRamp"]) + " Current Level: " + str(tmpPlantInfo["CurrentLevel"]))
        tmpGap = tmpDemand - tmpCurProd
        print("Demand: " + str(tmpDemand))
        print("Productin: " + str(tmpCurProd))
        rampDir = 0
        if(tmpGap > 0):
            rampDir = 1
        elif(tmpGap < 0):
            rampDir = -1
        else:
            pass
        mostEfficentSpeed, hR, efVal = getMostEfficent(rampDir, tmpPlantInfo)
        print(mostEfficentSpeed)
        setPlantLevel(plant, mostEfficentSpeed)
        los.logFutStep(plant,[mostEfficentSpeed,hR,efVal])
    los.logAllFutStep(currentTime, tmpDemand)

def currentAlgorithm(currentTime):
    tmpRealDemand = rgetElecDemand()
    #print("Real elec demand: " + str(tmpRealDemand))
    tmpRealSolar = getCurrentSolar()
    
    tmpDem = tmpRealDemand-tmpRealSolar
    tmpCurProd = getCurrentProduction()
    
    for plant in sim.plantList:
        tmpPlantInfo = getSpecPlantInformation(plant)
        #print(str(plant) + " Max Ramp: " + str(tmpPlantInfo["MaxRamp"]) + " Current Level: " + str(tmpPlantInfo["CurrentLevel"]))
        tmpGap = tmpDem - tmpCurProd
        
        if(abs(tmpGap) < tmpPlantInfo["MaxRamp"]):
            newLevel = tmpPlantInfo["CurrentLevel"] + tmpGap
        else:
            if(tmpGap > 0):
                newLevel = tmpPlantInfo['CurrentLevel'] + tmpPlantInfo['MaxRamp']
            elif(tmpGap < 0):
                newLevel = tmpPlantInfo['CurrentLevel'] - tmpPlantInfo['MaxRamp']
            else:
                pass
        setPlantLevel(plant, newLevel)
        heat = tmpPlantInfo["efficencyData"].iloc[:,0][newLevel]
        heatPerOut = tmpPlantInfo["efficencyData"].iloc[:,1][newLevel]
        los.logCurStep(plant, [newLevel,heat,heatPerOut])
    los.logCurFutStep(currentTime, tmpDem)

while True: 
    futureAlgorithm(currentTime)
    
    currentAlgorithm(currentTime)    

    currentTime = currentTime+1
    time.sleep(1)