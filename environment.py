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
        tmpCurProd = tmpCurProd + plant["CurrentLevel"]
    return tmpCurProd

def setPlantLevel(plant, level):
    ud.plantInformation[plant]["CurrentLevel"] = level
    
sim.initDefaultSimulation()

def getMostEfficent(rampDir, plant):
    mostEfficentVal = 9*10^3
    mostEfficentSpeed = 9*10^3
    data = plant["efficencyData"].iloc[:,1]
    if (rampDir == 1):
        for i in range(0,plant["MaxRamp"]):
            if(data[i] < mostEfficentVal):
                mostEfficentVal = data[i]
                mostEfficentSpeed = i
        heatRate = plant["efficencyData"].iloc[:,0][mostEfficentSpeed]
        return mostEfficentSpeed, heatRate, mostEfficentVal
    elif (rampDir == -1):
        for i in range(-plant["MaxRamp"],0):
            if(data[i] < mostEfficentVal):
                mostEfficentVal = data[i]
                mostEfficentSpeed = i
        heatRate = plant["efficencyData"].iloc[:,0][mostEfficentSpeed]
        return mostEfficentSpeed, heatRate, mostEfficentVal
    
def futureAlgorithm(currentTime):
    tmpElecDemandPred = fgetElecDemand()
    #print("Pred elec demand: " + str(tmpElecDemandPred))
    
    tmpSolarPred = getSolarPred()
    #print("Pred solar out: " + str(tmpSolarPred))
    
    tmpCurProd = getCurrentProduction()
    #print("Current Production: " + tmpCurProd)
    
    tmpDemand = tmpElecDemandPred-tmpSolarPred
    
    for plant in sim.plantList:
        tmpPlantInfo = getSpecPlantInformation(plant)
        #print(str(plant) + " Max Ramp: " + str(tmpPlantInfo["MaxRamp"]) + " Current Level: " + str(tmpPlantInfo["CurrentLevel"]))
        tmpGap = tmpDemand - tmpCurProd
        rampDir = 0
        if(tmpGap > 0):
            rampDir = 1
        elif(tmpGap < 0):
            rampDir = -1
        else:
            pass
        mostEfficentSpeed, hR, efVal = getMostEfficent(rampDir, tmpPlantInfo)
        setPlantLevel(plant, mostEfficentSpeed)
        los.logFutStep(plant,[mostEfficentSpeed,hR,efVal])
    los.logAllFutStep(currentTime, tmpDemand)

while True: 
    futureAlgorithm()
    tmpRealDemand = rgetElecDemand()
    #print("Real elec demand: " + str(tmpRealDemand))
    currentTime = currentTime+1
    time.sleep(3)