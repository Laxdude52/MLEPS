# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 10:53:33 2024

@author: School Account
"""
import copy
import pickle
import defaultSimulation as sim

plantNameList = list(copy.deepcopy(sim.plantList))
logFutDict = dict()

for plant in plantNameList:
    newCol = plant + "_level"
    logFutDict.update({newCol: list()})
    newCol = plant + "_heatrate"
    logFutDict.update({newCol: list()})
    newCol = plant + "_ratePerOut"
    logFutDict.update({newCol: list()})

logFutDict.update({"totalProd": list()})
logFutDict.update({"totalHeat": list()})
logFutDict.update({"totalHeatPerOut": list()})
logFutDict.update({"elecDem": list()})
logFutDict.update({"diffOutDem": list()})

logCurDict = copy.deepcopy(logFutDict)


def logFutStep(plant, level, heatRate, heatPerOut):
    logFutDict[str(plant + "_level")].append(level)
    logFutDict[str(plant + "_heatrate")].append(heatRate)
    logFutDict[str(plant + "_ratePerOut")].append(heatPerOut)


def logAllFutStep(idx, demand, solar):
    tmpOut = 0
    tmpHeat = 0
    tmpHeatOut = 0
    tmpElecDem = 0
    tmpDiff = 0

    for plant in plantNameList:
        tmpOut = tmpOut + logFutDict[str(plant + "_level")][idx]
        tmpHeat = tmpHeat + logFutDict[str(plant + "_heatrate")][idx]

    tmpElecDem = demand
    tmpDiff = tmpElecDem - tmpOut - solar
    tmpHeatOut = tmpHeat / tmpOut

    logFutDict["totalProd"].append(tmpOut)
    logFutDict["totalHeat"].append(tmpHeat)
    logFutDict["totalHeatPerOut"].append(tmpHeatOut)
    logFutDict["elecDem"].append(tmpElecDem)
    logFutDict["diffOutDem"].append(tmpDiff)
    # print("\nFuture: \ndemand: " + str(tmpElecDem) + "\nout: " + str(tmpOut) + "\ndiff: " + str(tmpDiff))


def logCurStep(plant, level, heatRate, heatPerOut):
    logCurDict[str(plant + "_level")].append(level)
    logCurDict[str(plant + "_heatrate")].append(heatRate)
    logCurDict[str(plant + "_ratePerOut")].append(heatPerOut)


def logCurFutStep(idx, demand, solar):
    tmpOut = 0
    tmpHeat = 0
    tmpHeatOut = 0
    tmpElecDem = 0
    tmpDiff = 0

    for plant in plantNameList:
        tmpOut = tmpOut + logCurDict[str(plant + "_level")][idx - 1]
        tmpHeat = tmpHeat + logCurDict[str(plant + "_heatrate")][idx - 1]
    tmpOut = tmpOut + solar

    tmpElecDem = demand
    tmpDiff = tmpElecDem - tmpOut - solar
    tmpHeatOut = tmpHeat / tmpOut

    logCurDict["totalProd"].append(tmpOut)
    logCurDict["totalHeat"].append(tmpHeat)
    logCurDict["totalHeatPerOut"].append(tmpHeatOut)
    logCurDict["elecDem"].append(tmpElecDem)
    logCurDict["diffOutDem"].append(tmpDiff)
    # print("\Current: \ndemand: " + str(tmpElecDem) + "\nout: " + str(tmpOut) + "\ndiff: " + str(tmpDiff))


def saveAll():
    allResults = dict()
    allResults.update({"Current: ": logCurDict})
    allResults.update({"Future: ": logFutDict})
    with open('loggedResult.pickle', 'wb'):
        pickle.dump(allResults)


def loadAll():
    with open('loggedResult.pickle', 'rb'):
        allResults = pickle.load()
