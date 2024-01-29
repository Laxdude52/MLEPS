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
import copy
import random

STARTTIME = 0
solarModelInfo = ['default', 'Solar']

#Get current plant levels + max ramp up/down
def getAllPlantInformation():
    return ud.plantInformation

def getSpecPlantInformation(plant):
    return ud.plantInformation[plant]

#Get solar pred 
def getSolarPred(currentTime):
    print(currentTime)
    return ud.livePredict(currentTime, 'Solar', 'Future', 'default')

def getCurrentSolar(currentTime):
    data = ud.dataModels['Solar']['Simple']['default']['initData'].y_train.iloc[[currentTime]]
    return data    

#Get elecDemandPRed
def fgetElecDemand(currentTime):
    print(currentTime)
    return ud.livePredict(currentTime, 'Demand', 'Future', 'default')

#Get real elec demand
def rgetElecDemand(currentTime):
    data = ud.dataModels['Demand']['Simple']['default']['initData'].y_train.iloc[[currentTime]]
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
    
sim.initDefaultSimulation()

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
        heatRate = plant["efficencyData"].iloc[:,0][mostEfficentSpeed]
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
        pass
        #raise Exception("No dir")
    if(mostEfficentSpeed > plant["Max"]):
        mostEfficentSpeed = plant["Max"]-1
    elif(mostEfficentSpeed < plant["Min"]):
        mostEfficentSpeed = plant["Min"]+1
    #print("Plant new level: " + str(mostEfficentSpeed))
    heatRate = plant["efficencyData"].iloc[:,0][mostEfficentSpeed]
    return mostEfficentSpeed, heatRate, mostEfficentVal
    
def futureAlgorithm(currentTime):
    random.shuffle(sim.plantList)
    tmpElecDemandPred = int(fgetElecDemand(currentTime))
    print("Pred elec demand: " + str(tmpElecDemandPred))
    
    tmpSolarPred = int(getSolarPred(currentTime))
    print("Pred solar out: " + str(tmpSolarPred))
    
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
            print("Plant: " + str(plant) + " up")
        elif(tmpGap < 0):   
            rampDir = -1
            print("Plant: " + str(plant) + " down")
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
                pass
        if(newLevel > tmpPlantInfo["Max"]):
            newLevel = tmpPlantInfo["Max"]-1
        elif(newLevel < tmpPlantInfo["Min"]):
            newLevel = tmpPlantInfo["Min"]+1
        ud.setPlantLevel(plant, newLevel)
        ud.resetPlantRampLeft(plant)
        #print("Live level: " + str(getSpecPlantInformation(plant)["CurrentLevel"]))
        heat = tmpPlantInfo["efficencyData"].iloc[:,0][newLevel]
        heatPerOut = tmpPlantInfo["efficencyData"].iloc[:,1][newLevel]
        los.logCurStep(plant, newLevel,heat,heatPerOut)
    los.logCurFutStep(currentTime, tmpRealDemand, tmpRealSolar)
 
resetSimulation()
currentTime = 0
while True: 
    futureAlgorithm(currentTime)
    
    currentAlgorithm(currentTime)    

    currentTime = currentTime+1
    #time.sleep(.5)