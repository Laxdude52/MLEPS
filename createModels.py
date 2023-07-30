# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:57:26 2023

@author: School Account
"""

import machineLearningTool as aim
import pandas as pd
import copy
import pickle 
import matplotlib.pyplot as plt
import numpy as np
import universalData as ud
import copy

solarDataList = ud.solarDataList
solarData = aim.data(solarDataList)

windDataList = ud.windDataList
windData = aim.data(windDataList)

energyDemandList = ud.energyDemandList
demandData = aim.data(energyDemandList)

coalList = ud.coalList
coalData = aim.data(coalList)

hydroList1 = ud.hydroList1
hydroData1 = aim.data(hydroList1) 

hydroList2 = ud.hydroList2
hydroData2 = aim.data(hydroList2)

combList1 = ud.combList1
combData1 = aim.data(combList1)
combList2 = ud.combList2
combData2 = aim.data(combList1)
combList3 = ud.combList3
combData3 = aim.data(combList1)
combList4 = ud.combList4
combData4 = aim.data(combList1)
combList5 = ud.combList5
combData5 = aim.data(combList1)
combList6 = ud.combList6
combData6 = aim.data(combList1)
combList7 = ud.combList1
combData7 = aim.data(combList1)
'''
The models that must be created:
Simple Solar Output Prediction AI
Future POAI Predictor
NAH FAM Future GHI Predictor
NAH FAM Future TmpF Predictor
Future Solar Energy Predictor
Quick Prediction scenario solar model
'''
#Create dictonaries
simpModels = dict()
futureModels = dict()
simpData = dict()
futureData = dict()
solarModels = dict()
windModels = dict()
energyDemandModels = dict()
coalModels = dict()
hydroModels1 = dict()
hydroModels2 = dict()
combModels = dict()
tmpDict = dict() 
models = dict()

combModels.update({"combModels1": copy.deepcopy(tmpDict)})
combModels.update({"combModels2": copy.deepcopy(tmpDict)})
combModels.update({"combModels3": copy.deepcopy(tmpDict)})
combModels.update({"combModels4": copy.deepcopy(tmpDict)})
combModels.update({"combModels5": copy.deepcopy(tmpDict)})
combModels.update({"combModels6": copy.deepcopy(tmpDict)})
combModels.update({"combModels7": copy.deepcopy(tmpDict)})

solarModels.update({"simpModels": copy.deepcopy(simpModels)})
solarModels.update({"futureModels": copy.deepcopy(futureModels)})
solarModels.update({"simpData": copy.deepcopy(simpData)})
solarModels.update({"futureData": copy.deepcopy(futureData)})
models.update({"solarModels":solarModels})

windModels.update({"simpModels": copy.deepcopy(simpModels)})
windModels.update({"futureModels": copy.deepcopy(futureModels)})
windModels.update({"simpData": copy.deepcopy(simpData)})
windModels.update({"futureData": copy.deepcopy(futureData)})
models.update({"windModels":windModels})

energyDemandModels.update({"simpModels": copy.deepcopy(simpModels)})
energyDemandModels.update({"futureModels": copy.deepcopy(futureModels)})
energyDemandModels.update({"simpData": copy.deepcopy(simpData)})
energyDemandModels.update({"futureData": copy.deepcopy(futureData)})
models.update({"energyDemandModels":energyDemandModels})

coalModels.update({"simpModels": copy.deepcopy(simpModels)})
coalModels.update({"simpData": copy.deepcopy(simpData)})
models.update({"coalModels":coalModels})

hydroModels1.update({"simpModels": copy.deepcopy(simpModels)})
hydroModels1.update({"simpData": copy.deepcopy(simpData)})
models.update({"hydroModels1":hydroModels1})

hydroModels2.update({"simpModels": copy.deepcopy(simpModels)})
hydroModels2.update({"simpData": copy.deepcopy(simpData)})
models.update({"hydroModels2":hydroModels2})

for key in combModels.keys():
    combModels[key].update({"simpModels": copy.deepcopy(simpModels)})
    combModels[key].update({"simpData": copy.deepcopy(simpData)})
    models.update({(str(key)):combModels[key]})

#Create the simple output AI
def createSimpModel(name, dataList, columns, modelPack, structure, category):
    #Create Data
    simpData = aim.data(dataList)
    simpData.baseFilter(columns['all'], 'na', 0, 'rmv')
    simpData.split(.3, .1, columns['y'], columns['X'], shuffle=True)

    #Create Model
    if structure == 'regular':
        simpModel = aim.model(modelPack['model'], modelPack['name'], modelPack['param_dist'], simpData)
        simpModel.train(1, 'NA')
    elif structure == 'ensemble':
        simpModel = aim.ensemble(modelPack['modelList'], modelPack['newModelDataframe'], simpData)
        simpModel.create_Base()
        simpModel.lastLayer(modelPack['lastList'])
    elif structure == 'breakout':
        simpModel = aim.breakout(modelPack['clusterSearch'], modelPack['cluster'], modelPack['clusterName'], modelPack['cluster_params'], modelPack['run_params'], simpData)
        simpModel.startCluster()
        simpModel.createData(prepTrain=True)
        simpModel.trainSecond(modelPack['secondLayerModel'])
        
    if category.type==list:
        models[category[0]][category[1]]['simpModels'].update({name:simpModel})
        models[category[0]][category[1]]['simpData'].update({name:simpData})
    else:
        models[category]['simpModels'].update({name:simpModel})
        models[category]['simpData'].update({name:simpData})

#Create the Future kW model    
def createFutureModel(name, dataList, dataPack, columns, modelPack, structure, category):
    #name, numPastSteps, numFutureSteps, aggTime, timeCols, lagTime, notLagTime, modelPack, structure
    #Create Data
    Data = aim.data(dataList)
    Data.baseFilter(columns['all'], 'na', 0, 'mean')
    multiTime = False
    if len(dataPack['timeCols']) > 1:
        multiTime = True
    Data.aggrigate(dataPack['aggTime'], dataPack['timeCols'], multiTime)
    Data.lagFuture(columns['y'], dataPack['lagTime'], dataPack['notLagTime'], dataPack['numPastSteps'], dataPack['numFutureSteps'])
    Data.split(.25, .05, columns['y'], Data.input_cols)
    
    #Create Model
    if structure == 'regular':
        Model = aim.model(modelPack['model'], modelPack['name'], modelPack['param_dist'], Data)
        Model.train(1, 'NA')
    elif structure == 'ensemble':
        Model = aim.ensemble(modelPack['modelList'], modelPack['newModelDataframe'], Data)
        Model.create_Base()
        Model.lastLayer(modelPack['lastList'])
    elif structure == 'breakout':
        Model = aim.breakout(modelPack['clusterSearch'], modelPack['cluster'], modelPack['clusterName'], modelPack['cluster_params'], modelPack['run_params'], Data)
        Model.startCluster()
        Model.createData(prepTrain=True)
        Model.trainSecond(modelPack['secondLayerModel'])
        
        
    if category.type==list:
        models[category[0]][category[1]]['futureModels'].update({name:Model})
        models[category[0]][category[1]]['futureData'].update({name:Data})
    else:
        models[category]['simpModels'].update({name:Model})
        models[category]['simpData'].update({name:Data})
        