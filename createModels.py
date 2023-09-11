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
import data as ud
import copy
import os

#Create dictonaries
simpModels = dict()
futureModels = dict()
simpData = dict()
futureData = dict()
models = dict()
loadedData = dict()

def prepModel(modelType, numModels, modelsStoredName):
    for i in range(numModels):
        tmpModelDict = dict()
        tmpModelDict.update({"simpModels": copy.deepcopy(simpModels)})
        tmpModelDict.update({"futureModels": copy.deepcopy(futureModels)})
        tmpModelDict.update({"simpData": copy.deepcopy(simpData)})
        tmpModelDict.update({"futureData": copy.deepcopy(futureData)})
        tmpKey = modelsStoredName+str(i)
        try:
            models[modelType].update({tmpKey:copy.deepcopy(tmpModelDict)})
        except:
            models.update({modelType:dict()})
            models[modelType].update({tmpKey:copy.deepcopy(tmpModelDict)})
prepModel('Solar', 3, 'Test')
prepModel('Wind', 5, 'Alpha')
prepModel('Combustion', 2, 'Beta')

'''
The models that must be created:
Simple Solar Output Prediction AI
Future POAI Predictor
NAH FAM Future GHI Predictor
NAH FAM Future TmpF Predictor
Future Solar Energy Predictor
Quick Prediction scenario solar model
'''

#Create the simple output AI
def createSimpModel(name, dataList, columns, modelPack, structure, category, sectionName):
    #Create Data
    if type(dataList) == dict:
        dataList = list(dataList.values())
    print("Line 56")
    simpData = aim.data(dataList)
    #return simpData
    simpData.baseFilter(columns['all'], 'na', 0, 'rmv')
    #return a
    simpData.split(.3, .1, columns['y'], columns['X'], shuffle=True)

    #Create Model
    if structure == 'regular':
        simpModel = aim.model(modelPack['model'], modelPack['modelType'], modelPack['param_dist'], simpData)
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
        
    if type(category)==list:
        models[category[0]][category[1]]['simpModels'].update({name:simpModel})
        models[category[0]][category[1]]['simpData'].update({name:simpData})
    else:
        models[category][sectionName]['simpModels'].update({name:simpModel})
        models[category][sectionName]['simpData'].update({name:simpData})

#Create the Future kW model    
def createFutureModel(name, dataList, dataPack, columns, modelPack, structure, category, sectionName):
    #name, numPastSteps, numFutureSteps, aggTime, timeCols, lagTime, notLagTime, modelPack, structure
    #Create Data
    if type(dataList) == dict:
        dataList = list(dataList.values())
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
        models[category][sectionName]['simpModels'].update({name:Model})
        models[category][sectionName]['simpData'].update({name:Data})

def save_copy(modelDict, parentDir, name):
    path = os.path.join(parentDir, name)
    try:
        os.mkdir(path)
    except:
        print("No path made")
    futurePath = os.path.join(path, 'futurePath')
    try:
        os.mkdir(futurePath)
    except:
        print("No path made")
    simpPath = os.path.join(path, 'simpPath')
    try:
        os.mkdir(simpPath)
    except:
        print("No path made")
    
    for key in modelDict['futureData'].keys():
        tmpPath = os.path.join(futurePath, key)
        os.mkdir(tmpPath)
        model = modelDict['futureModels'][key]
        data = modelDict['futureData'][key]
        
        fileHandle = open(tmpPath, 'wb')
        pickle.dump(model, fileHandle)
 
        fileHandle = open(tmpPath, 'wb')
        pickle.dump(data, fileHandle) 
        
    for key in modelDict['simpData'].keys():
        tmpPath = os.path.join(simpPath, key)
        try:
            os.mkdir(tmpPath)
        except:
            print("No path made")
        model = modelDict['simpModels'][key]
        data = modelDict['simpData'][key]

        tmpPathM = os.path.join(tmpPath, 'Model.pkl')
        fileHandle = open(tmpPathM, 'wb')
        pickle.dump(model, fileHandle)
 
        tmpPathD = os.path.join(tmpPath, 'Data.pkl')
        fileHandle = open(tmpPathD, 'wb')
        pickle.dump(data, fileHandle)
           
def add_new(parentDir, name, model, data):
    #Make sure to include r before text, like r"c:/users..."
    tmpPath = os.path.join(parentDir, name)
    os.mkdir(tmpPath)
    
    tmpPathM = os.path.join(tmpPath, 'Model.pkl')
    fileHandle = open(tmpPathM, 'wb')
    pickle.dump(model, fileHandle)
 
    tmpPathD = os.path.join(tmpPath, 'Data.pkl')
    fileHandle = open(tmpPathD, 'wb')
    pickle.dump(data, fileHandle)

#def scan_update()
#Scan current stored models and such and only add the new ones    