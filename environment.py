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

sim.initDefaultSimulation()

while True:
    for plant in sim.plantList:
        tmpPlantInfo = getSpecPlantInformation(plant)
        print(str(plant) + " Max Ramp: " + str(tmpPlantInfo["MaxRamp"]) + " Current Level: " + str(tmpPlantInfo["CurrentLevel"]))
    
    tmpElecDemandPred = fgetElecDemand()
    print("Pred elec demand: " + str(tmpElecDemandPred))
    
    tmpSolarPred = getSolarPred()
    print("Pred solar out: " + str(tmpSolarPred))
    
    tmpRealDemand = rgetElecDemand()
    print("Real elec demand: " + str(tmpRealDemand))
    
    print("One run done, time=" + str(currentTime))
    currentTime = currentTime+1
    time.sleep(3)