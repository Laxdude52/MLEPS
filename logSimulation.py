# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:53:33 2024

@author: School Account
"""
import defaultSimulation as sim
import copy

plantNameList = list(copy.deepcopy(sim.plantList))
logFutDict = dict()

for plant in plantNameList:
    newCol = plant + "_level"
    logFutDict.update({newCol:list()})
    newCol = plant + "_heatrate"
    logFutDict.update({newCol:list()})
    newCol = plant + "_ratePerOut"
    logFutDict.update({newCol:list()})

logFutDict.update({"totalProd":list()})    
logFutDict.update({"totalHeat":list()})    
logFutDict.update({"totalHeatPerOut":list()})    
logFutDict.update({"elecDem":list()})    
logFutDict.update({"diffOutDem":list()})    

logCurDict = copy.deepcopy(logFutDict)

def logFutStep(plant, data):
    logFutDict[str(plant+"_level")].append(data[0])
    logFutDict[str(plant+"_heatrate")].append(data[1])
    logFutDict[str(plant+"_ratePerOut")].append(data[2])

def logAllFutStep(idx, demand):
    tmpOut = 0
    tmpHeat = 0
    tmpHeatOut = 0
    tmpElecDem = 0
    tmpDiff = 0
    
    for plant in plantNameList:
        tmpOut = tmpOut + logFutDict[str(plant+"_level")][idx]
        tmpHeat = tmpHeat + logFutDict[str(plant+"_heatrate")][idx]
    
    tmpElecDem = demand
    tmpDiff = tmpOut - tmpElecDem
    tmpHeatOut = tmpHeat/tmpOut
    
    logFutDict["totalProd"].append(tmpOut)
    logFutDict["totalHeat"].append(tmpHeat)
    logFutDict["totalHeatPerOut"].append(tmpHeatOut)
    logFutDict["elecDem"].append(tmpElecDem)
    logFutDict["diffOutDem"].append(tmpDiff)
    print("\nFuture: \ndemand: " + str(tmpElecDem) + "\nout: " + str(tmpOut) + "\ndiff: " + str(tmpDiff))
    
def logCurStep(plant, data):
    logCurDict[str(plant+"_level")].append(data[0])
    logCurDict[str(plant+"_heatrate")].append(data[1])
    logCurDict[str(plant+"_ratePerOut")].append(data[2])

def logCurFutStep(idx, demand):
    tmpOut = 0
    tmpHeat = 0
    tmpHeatOut = 0
    tmpElecDem = 0
    tmpDiff = 0
    
    for plant in plantNameList:
        tmpOut = tmpOut + logCurDict[str(plant+"_level")][idx]
        tmpHeat = tmpHeat + logCurDict[str(plant+"_heatrate")][idx]
    
    tmpElecDem = demand
    tmpDiff = tmpOut - tmpElecDem
    tmpHeatOut = tmpHeat/tmpOut
    
    logCurDict["totalProd"].append(tmpOut)
    logCurDict["totalHeat"].append(tmpHeat)
    logCurDict["totalHeatPerOut"].append(tmpHeatOut)
    logCurDict["elecDem"].append(tmpElecDem)
    logCurDict["diffOutDem"].append(tmpDiff)
    print("\Current: \ndemand: " + str(tmpElecDem) + "\nout: " + str(tmpOut) + "\ndiff: " + str(tmpDiff))
