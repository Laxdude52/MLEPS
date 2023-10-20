# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 11:18:38 2023

@author: School Account
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import gc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from scipy.stats import randint, uniform
import math
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pygam as pg
import pickle 
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.cluster import Birch
import numbers 
from numpy import unique
from sklearn.cluster import Birch
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
import warnings
import copy

class simpleData():
    def __init__(self, df):
        self.df = df
    def Append(self, new, name):
        self.df[name] = new

class simpleEnsemble():
    def __init__(self, data, base, layer1, layer2):
        self.data = data
        self.base = base
        self.layer1 = layer1
        self.layer2 = layer2

class data:
    def __init__(self, dfList):
        self.dfList = dfList
        self.valid = True
        if(dfList != "NA"):
            for df in dfList:
                try:
                    test = pd.DataFrame(df)
                except:
                    warnings.warn(("Dataframe: " + str(df) + "can NOT be conerted to a datraframe"))
                    self.valid = False
            if not self.valid:
                raise Exception("Invalid Input")
                    
            self.rawDf = pd.concat(self.dfList) 
    def aggrigate(self, timeframe, retain, multiTimeRetain): 
        #columns: columns to keep
        #Timeframe: new timeframe 
        
        print("Aggrigate")
        tmpdf = self.filtered.copy()
        #Come back and fix this, there is an issue where dataframes are being eltered incorrectly
        #The try except just coveres up this issue
        try:
            tmpdf['Timestamp'] = pd.to_datetime(tmpdf['Timestamp'])
            self.filtered = tmpdf.resample(timeframe, on='Timestamp').sum()
        except KeyError as e:
            warnings.warn(e)
            warnings.warn("NO TIMESTAMP FOUND")
            raise Exception("NoColumn")
            
        except TypeError as e:
            warnings.warn(e)
            warnings.warn("CAN NOT CONVERT TO DATETIME")
            raise Exception("InalidColumn")
        
        if multiTimeRetain:
            for i in range(len(retain)): 
                if(retain[i].upper() == 'YEAR'):
                    self.filtered[retain[i]] = self.filtered.index.year
                elif(retain[i].upper()  == 'MONTH'):
                    self.filtered[retain[i]] = self.filtered.index.month
                elif(retain[i].upper()  == 'DAY'):
                    self.filtered[retain[i]] = self.filtered.index.day
                elif(retain[i].upper()  == 'HOUR'):
                    self.filtered[retain[i]] = self.filtered.index.hour
                elif(retain[i].upper()  == 'MINUTE'):
                    self.filtered[retain[i]] = self.filtered.index.minute
                else:
                    warnings.warn("No time column kept")
        else:
            if(retain.upper()  == 'YEAR'):
                self.filtered[retain] = self.filtered.index.year
            elif(retain.upper()  == 'MONTH'):
                self.filtered[retain] = self.filtered.index.month
            elif(retain.upper()  == 'DAY'):
                self.filtered[retain] = self.filtered.index.day
            elif(retain.upper()  == 'HOUR'):
                self.filtered[retain] = self.filtered.index.hour
            elif(retain.upper()  == 'MINUTE'):
                self.filtered[retain] = self.filtered.index.minute


    def baseFilter(self, columnNames, aboveVal, belowVal, naDecision):
        print("Base FIlter")
        print(columnNames)
        self.valid= True
        try:
            self.filtered = self.rawDf[columnNames]
        except:
            for column in columnNames:
                try:
                    self.rawDf[column]
                except KeyError as e:
                    warnings.warn(str(e))
                    warnings.warn(("column: " + column + " does not exist"))
                    self.valid = False
        if not self.valid:
            raise Exception("Invalid Input")
        #return self.filtered
        for (columnName, columnData) in self.filtered.iteritems():
            print(columnName)
            if(naDecision == 'zero'):
                self.filtered[columnName] = pd.to_numeric(self.filtered[columnName], errors='coerce')
                self.filtered[columnName].fillna(0, inplace=True)
            if(isinstance(columnData.values[0], numbers.Number)):
                if(belowVal != 'na'):
                    print("Start Below")
                    self.filtered.loc[self.filtered[columnName] < belowVal, columnName] = belowVal
                    print("Below done")
                if(aboveVal != 'na'): 
                    print("Start Above")
                    self.filtered.loc[self.filtered[columnName] > aboveVal, columnName] = aboveVal
                    print("Above done")
                if(naDecision == 'mean'):
                    print("Start Mean Filter")
                    self.filtered[columnName].fillna(self.filtered[columnName].mean(), inplace=True)
                    '''
                    tmpData = self.filtered[[columnName]].copy()
                    meanVal = tmpData.mean()
                    tmp = tmpData.fillna(meanVal)
                    self.filtered = self.filtered.assign(columnName = tmp)
                    '''
                    print("Replace NA Done")
                if(naDecision == 'zero'):
                    self.filtered[columnName].fillna(0, inplace=True)
                    print("Replace NA Done")
                print("Finished one")
            else:
                print(columnName)
                print("This column is not a int OR it is not a numpy int/float, make sure you apply any preprocessing to it manually")

        if(naDecision == 'rmv'):
            print("Start Remove Filter")
            self.filtered.dropna(inplace=True)
            print("Remove done")     
            
    def summary(self):
        print(self.filtered.describe().transpose())
        print(self.filtered.describe())
        
    def lagFuture(self, target_variable, laggedVars, notLaggedXVars, num_lags, future_steps):
        print("Lag Future")
        
        try:
            # Step 1: Create the target variable
            self.filtered[target_variable+'_future'] = self.filtered[target_variable].shift(-future_steps)
        except KeyError:
            raise Exception("Invalid target variable")
        
        # Step 2: Feature engineering
        for i in range(1, num_lags+1):
            for col in laggedVars:
                try:
                    self.filtered[f'{col}_lag{i}'] = self.filtered[col].shift(i) 
                except KeyError:
                    warnings.warn(("Column" + col + " does not exist"))
               
        #Remove new na data
        self.filtered = self.filtered[num_lags:(len(self.filtered)-future_steps)]
        
        #Create column list
        self.input_cols = np.array(notLaggedXVars)
        for i in range(1, num_lags+1):
            for col in laggedVars:
                tmpcol = f'{col}_lag{i}'
                self.input_cols = np.append(self.input_cols, tmpcol)
    
    def split(self, testSize, validSize, target_variable, input_cols, scaleVal='NA', shuffle=False):      
        print("Split")
        # Step 3: Split the data
        test_size = int(len(self.filtered) * testSize)
        valid_size = int(len(self.filtered) * validSize)
        train_size = len(self.filtered) - valid_size - test_size
        
        if not scaleVal == 'NA':
            print("SCALING")
            newTargetVal = self.filtered[target_variable].values.reshape(-1, 1)
            scaler = MinMaxScaler(feature_range=(scaleVal[0], scaleVal[1]))
            fitScaler = scaler.fit_transform(newTargetVal)
            self.filtered[target_variable] = copy.deepcopy(fitScaler)
        
        self.train_df = self.filtered.iloc[:train_size].copy()
        self.train_df.reset_index(inplace=True)
        self.valid_df = self.filtered.iloc[train_size:train_size+valid_size].copy()
        self.valid_df.reset_index(inplace=True)
        self.test_df = self.filtered.iloc[-test_size:].copy()
        self.test_df.reset_index(inplace=True)
        
        if shuffle:
            self.train_df = self.train_df.sample(frac = 1)
            self.test_df = self.train_df.sample(frac = 1)
            self.valid_df = self.train_df.sample(frac = 1)
        
        self.X_train = self.train_df[input_cols]
        self.y_train = self.train_df[target_variable]
        
        self.X_valid = self.valid_df[input_cols]
        self.y_valid = self.valid_df[target_variable]
        
        self.X_test = self.test_df[input_cols]
        self.y_test = self.test_df[target_variable]
        
    def rescale(self):
        print("Rescale")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.fit_transform(self.X_test)
        self.X_valid = self.scaler.fit_transform(self.X_valid)
        
    def standardize(self):
        print("Standardize")
        self.scaler = StandardScaler().fit(self.X_train)
        self.X_train = self.scaler.fit(self.X_train)
        self.X_test = self.scaler.fit(self.X_test)
        self.X_valid = self.scaler.fi(self.X_valid)        
 
    def normalize(self):
        print("Normalize") 
        self.scaler = Normalizer().fit(self.X_train)
        self.X_train = self.scaler.fit(self.X_train)
        self.X_test = self.scaler.fit(self.X_test)
        self.X_valid = self.scaler.fi(self.X_valid) 
        
class model:
    def __init__(self, model, modelType, param_dist, data, isClusterModel=False):
        self.model = model
        self.param_dist = param_dist
        self.data = data
        self.modelType = modelType
        self.searchDone = False
        self.isClusterModel = isClusterModel
    def searchCV(self, n_iter, cv, searchDsetSize):
        print("Searching (CV)")
        if(self.param_dist != 'NA'):
            self.search = RandomizedSearchCV(
                self.model, 
                param_distributions=self.param_dist, 
                n_iter=n_iter,
                n_jobs=-1,
                cv=cv,
                random_state=42,
                verbose=True,
                )
        
            # Train the model on the training data
            self.search.fit(self.data.X_train[0:int(len(self.data.X_train)/searchDsetSize), :], self.data.y_train[0:int(len(self.data.X_train)/searchDsetSize)])
            # Print the best hyperparameters and corresponding validation score
            print("Best hyperparameters: ", self.search.best_params_)
            print("Validation score: ", self.search.best_score_)
            self.searchDone = True
    def train(self, trainingDsetSize, iterators):
        print("Training")
        if(self.modelType.upper() == 'XGB'):
            # Create a new XGBoost model with the best hyperparameters
            self.model = xgb.XGBRegressor(**self.search.best_params_, n_estimators=iterators, objective='reg:squarederror', tree_method='gpu_hist', random_state=42, verbose=True)
            
            # Train the model on the full training data
            self.model.fit(self.data.X_train[0:int(len(self.data.X_train)/trainingDsetSize)], self.data.y_train[0:int(len(self.data.y_train)/trainingDsetSize)],verbose=True)
        elif(self.modelType.upper() == 'SVM'):
            self.model = SVR(kernel='rbf', C=self.search.best_params_['C'], gamma=self.search.best_params_['gamma'])
            self.model.fit(self.data.X_train[0:int(len(self.data.X_train)/trainingDsetSize)], self.data.y_train[0:int(len(self.data.y_train)/trainingDsetSize)],verbose=True, n_jobs=-1)   
        elif(self.modelType.upper() == 'GAM'):
            print("Training GAM")
            self.Xtrain = self.data.X_train[0:int(len(self.data.X_train)/trainingDsetSize)]
            self.Ytrain = self.data.y_train[0:int(len(self.data.y_train)/trainingDsetSize)]
            self.model = self.model.fit(self.Xtrain, self.Ytrain)
        elif(self.modelType.upper() == 'DNN'):
            print("Training DNN")
            self.es = tf.keras.callbacks.EarlyStopping(patience=10)
            self.history = self.model.fit(self.data.X_train[0:int(len(self.data.X_train)/trainingDsetSize)], self.data.y_train[0:int(len(self.data.y_train)/trainingDsetSize)], epochs=iterators, callbacks=[self.es], validation_data=(self.data.X_valid, self.data.y_valid))
        #Models to make: BIRCH, K_Means, Spectral, Mixture of Gaussians, Maybe: Optics 
        elif(self.searchDone == False and self.isClusterModel):
            self.tmpData = self.data.X_train.copy()
            self.tmpTrain = np.ascontiguousarray(self.tmpData[0:int(len(self.tmpData)/trainingDsetSize)])
            self.model.fit(self.tmpTrain)
            self.initgroupPreds = self.model.predict(np.ascontiguousarray(self.data.X_train[0:int(len(self.data.X_train)/trainingDsetSize)]))
            self.initgroups = unique(self.initgroupPreds)
        else:
            raise Exception("Incorrect model name, or attempted to grid search cluster model, could not train")
            
    def plot_loss(self):
      if(self.modelType == 'DNN'):
          plt.plot(self.history.history['loss'], label='loss')
          plt.plot(self.history.history['val_loss'], label='val_loss')
          plt.ylim([0, 700])
          plt.xlabel('Epoch')
          plt.ylabel('Error [kW]')
          plt.legend()
          plt.grid(True)
      else:
          warnings.warn("Must use a DNN for this function")
    
    def modPredict(self, X, y, returnResults=False, returnPred=False):
        print("Predicting")
        # Make predictions on the validation data and calculate the mean absolute error
        if(self.modelType == 'XGBDMX'):
            self.val_preds = self.model.predict(xgb.DMatrix(X))
        else: 
            if self.isClusterModel:
                self.val_preds = self.model.predict(np.ascontiguousarray(X))
                self.groups = unique(self.val_preds)
                return self.val_preds, self.groups
            else:
                self.val_preds = self.model.predict(X)
                try:
                    self.mse = mean_squared_error(y, self.val_preds)
                    self.rmse = math.sqrt(self.mse)
                    print("Test MSE: ", self.mse)
                    print("Test RMSE: ", self.rmse)
                except:
                    pass
                if(returnResults and returnPred):
                    return self.rmse, self.val_preds
                elif(returnResults):
                    return self.rmse 
                elif(returnPred):
                    return self.val_preds

class ensemble():
    def __init__(self, modelList, newModelDataframe, data):
        self.modelList = modelList
        self.newModelDataframe = newModelDataframe
        self.data = data
        
    def create_Base(self):
        print("Create Base")
        self.dataLayerTrain = [len(self.modelList)]
        self.dataLayerTest = [len(self.modelList)] 
        self.dataLayerValid = [len(self.modelList)]
        self.num_layers = len(self.modelList)

        for i in range(self.num_layers): 
            #i is the layer, j is the model
            #UPDATE I AND J TO BE ACTUAL NAME LATER
            j = 0
            self.num_models = len(self.modelList[i])
            while j < self.num_models:
                print("I IS " + str(i))
                print("J IS " + str(j))
                #Unsure what this does
                
                if (self.modelList[i][j] == 0):
                    break
                
                print("Creating Data")
                print(i)
                #Train the models
                if(isinstance(self.modelList[i][j], str)):
                    if not i == 0:
                        self.data = data('NA')
                        self.data.X_train = self.dataLayerTrain[i-1].df
                        self.data.X_test = self.dataLayerTest[i-1].df
                        self.data.X_valid = self.dataLayerValid[i-1].df
                        self.data.y_train = self.modelList[i-1][0].data.y_train
                        self.data.y_test = self.modelList[i-1][0].data.y_test
                        self.data.y_valid = self.modelList[i-1][0].data.y_valid
                    
                    self.TmpModel = model(self.newModelDataframe[i][j][0], self.newModelDataframe[i][j][1], 
                                                      self.newModelDataframe[i][j][2], self.data)
                    #ADD CODE TO CONDUCT A RANDOM SEARCH
                    #n_iter, cv, searchDsetSize
                    '''
                    self.model.searchCV(self.newModelDataframe[i][j][3], self.newModelDataframe[i][j][4],
                                        self.newModelDataframe[i][j][5])
                    '''
                    #trainingDsetSize, iterators
                    self.TmpModel.train(self.newModelDataframe[i][j][6], self.newModelDataframe[i][j][7])
                    #self.TmpModel.predict() 
                    self.modelList[i][j] = self.TmpModel 
                    if(j == len(self.modelList[i])-1):
                        j == 0
                elif(j == 0 and i == 0):
                    print("i and j are 0")
                    self.dataLayerTrain[i] = simpleData(self.modelList[i][0].data.X_train) 
                    self.dataLayerTest[i] = simpleData(self.modelList[i][0].data.X_test)
                    self.dataLayerValid[i] = simpleData(self.modelList[i][0].data.X_valid)
                    self.TmpName = str("Layer" + str(i) + "_Model" + str(j))
                    self.dataLayerTrain[i].Append(self.modelList[i][j].model.predict(self.modelList[i][0].data.X_train), (self.modelList[i][j].modelType + self.TmpName))
                    self.dataLayerTest[i].Append(self.modelList[i][j].model.predict(self.modelList[i][0].data.X_test), (self.modelList[i][j].modelType + self.TmpName))
                    self.dataLayerValid[i].Append(self.modelList[i][j].model.predict(self.modelList[i][0].data.X_valid), (self.modelList[i][j].modelType + self.TmpName))
                elif(j == 0):
                    print("j is 0")
                    self.dataLayerTrain[i] = simpleData(self.dataLayerTrain[i-1].df)
                    self.dataLayerTest[i] = simpleData(self.dataLayerTest[i-1].df)
                    self.dataLayerValid[i] = simpleData(self.dataLayerValid[i-1].df)
                    self.TmpName = str("Layer" + str(i) + "_Model" + str(j))
                    self.dataLayerTrain[i].Append(self.modelList[i][j].model.predict(self.modelList[i][0].data.X_train), (self.modelList[i][j].modelType + self.TmpName))
                    self.dataLayerTest[i].Append(self.modelList[i][j].model.predict(self.modelList[i][0].data.X_test), (self.modelList[i][j].modelType + self.TmpName))
                    self.dataLayerValid[i].Append(self.modelList[i][j].model.predict(self.modelList[i][0].data.X_valid), (self.modelList[i][j].modelType + self.TmpName))
                elif(i==0):
                    print("i is 0")
                    self.TmpName = str("Layer" + str(i) + "_Model" + str(j))
                    self.dataLayerTrain[i].Append(self.modelList[i][j].model.predict(self.modelList[i][0].data.X_train[baseInput]), (self.modelList[i][j].modelType + self.TmpName))
                    self.dataLayerTest[i].Append(self.modelList[i][j].model.predict(self.modelList[i][0].data.X_test[baseInput]), (self.modelList[i][j].modelType + self.TmpName))
                    self.dataLayerValid[i].Append(self.modelList[i][j].model.predict(self.modelList[i][0].data.X_valid[baseInput]), (self.modelList[i][j].modelType + self.TmpName))
                else:
                    print("i and j are not 0")
                    self.TmpName = str("Layer" + str(i) + "_Model" + str(j))
                    self.dataLayerTrain[i].Append(self.modelList[i][j].model.predict(self.dataLayerTrain[i-1].df), (self.modelList[i][j].modelType + self.TmpName))
                    self.dataLayerTest[i].Append(self.modelList[i][j].model.predict(self.dataLayerTest[i-1].df), (self.modelList[i][j].modelType + self.TmpName))
                    self.dataLayerValid[i].Append(self.modelList[i][j].model.predict(self.dataLayerValid[i-1].df), (self.modelList[i][j].modelType + self.TmpName))
                
                j += 1
                print("One model finished all 3 predictions")
                print(j)
    def lastLayer(self, lastList):
        print("Last Layer")
        self.finalModelList = []
        self.lastList = lastList 
        #Fix later, i should not be -1, not sure why i is -1
        i = len(self.modelList)-1
        self.data = data('NA')
        self.data.X_train = self.dataLayerTrain[i-1].df
        self.data.X_test = self.dataLayerTest[i-1].df
        self.data.X_valid = self.dataLayerValid[i-1].df
        self.data.y_train = self.modelList[i-1][0].data.y_train
        self.data.y_test = self.modelList[i-1][0].data.y_test
        self.data.y_valid = self.modelList[i-1][0].data.y_valid
        for i in range(len(self.lastList)):
            self.model = model(self.lastList[i][0], self.lastList[i][1], self.lastList[i][2], self.data)
            self.model.train(self.lastList[i][3], self.lastList[i][4])
            #self.model.predict()
            self.finalModelList.append(self.model)
            
    def basePred(self, data, returnLast=True, returnAll=False):
        self.layerPredictions = []
        self.preds = []
        for layer in range(self.num_layers):
            self.modelCount = 0
            while self.modelCount < self.num_models:
                print("Layer: " + str(layer))
                print("Model " + str(self.modelCount))
                if not layer == 0:
                    tmpData = self.layerPredictions[layer-1]
                    self.X = tmpData[0]
                    self.y = tmpData[1]
                else:
                    if type(data) == 'list':
                        self.X = data[0]
                        self.y = data[1]
                    else:
                        tmpData = data
                        self.X = pd.concat([tmpData.X_train, tmpData.X_valid, tmpData.X_test], axis=0)
                        self.y = pd.concat([tmpData.y_train, tmpData.y_valid, tmpData.y_test], axis=0)
                
                tmpPred = self.modelList[layer][self.modelCount].modPredict(self.X, self.y, returnPred=True)
                self.preds.append(tmpPred)
                self.modelCount= self.modelCount +1
            X = pd.concat([self.X, self.preds], axis=0)
            y = self.y
            self.layerPredictions[layer] = [X, y]
        if returnLast:
            return self.layerPredictions[(len(self.layerPredictions)-1)]
        elif returnAll:
            return self.layerPredictions
        else:
            raise Exception("No value returned, set reutrnLast or returnAll to true")
        
    def modPredict(self, X, y, returnResults = False, returnPred = False, firstDoneData = False, baseDataInput=[]):
        if not firstDoneData:
            print("Starting base layer predictions")
            tmpData = [X,y]
            self.baseData = self.basePred(tmpData)
        else:
            self.baseData = baseDataInput
        #Final Layer Predictions
        if(len(self.finalModelList) == 1):
            self.pred = self.finalModelList[0].modPredict(self.baseData[0], self.baseData[1], returnPred=True)
        else:
            #UPDATE TO BE A DICT NOT A LIST!!!
            self.pred = []
            for i in range(len(self.finalModelList)):
                tmpPred = self.finalModelList[i].modPredict(self.baseData[0], self.baseData[1], returnPred=True)
                self.pred.append(tmpPred)
        if returnPred and returnResults:
            raise Exception("Can not return results yet")
        elif returnPred:
            return self.pred
        elif returnResults:
            raise Exception("Can not return results yet")
            
            
def sortGroups(tmpData, catagoriesTrain, catagoriesTest, catagoriesGroup):
    print("Sort Groups")
    tmpdata = tmpData
    tmpcatagoriesTrain = pd.DataFrame(catagoriesTrain).copy()
    tmpcatagoriesTest = pd.DataFrame(catagoriesTest).copy()
    tmpcatagoriesGroup = catagoriesGroup.copy()
    
    tmpdataTest = pd.concat([tmpdata.y_test, tmpdata.X_test, tmpcatagoriesTest], axis=1)
    tmpdataTrain = pd.concat([tmpdata.y_train, tmpdata.X_train, tmpcatagoriesTrain], axis=1)
    tmpidx = (len(tmpdataTrain.columns) -1) 
    tmpcolumn = tmpdataTrain.columns[tmpidx]
    
    print(tmpidx)
    dictOfDataTrain = dict() 
    dictOfDataTest = dict()
    dictOfData = dict()
    for i in range(len(catagoriesGroup)):
        dictOfDataTrain.update({i:pd.DataFrame(tmpdataTrain[tmpdataTrain[tmpcolumn] == tmpcatagoriesGroup[i]])})
        dictOfDataTest.update({i:pd.DataFrame(tmpdataTest[tmpdataTest[tmpcolumn] == tmpcatagoriesGroup[i]])})
    yColumnName = dictOfDataTrain[0].columns[0]
    for i in range(len(dictOfDataTrain)):
        tmpData = dictOfDataTrain[i]
        tmpy = tmpData[yColumnName]
        tmpx = tmpData.drop(yColumnName, axis=1, inplace=False)
        prepData = data('NA')
        prepData.X_train = tmpx
        prepData.y_train = tmpy
        
        tmpData = dictOfDataTest[i]
        tmpy = tmpData[yColumnName]
        tmpx = tmpData.drop(yColumnName, axis=1, inplace=False)
        prepData.X_test = tmpx 
        prepData.y_test = tmpy
        
        dictOfData.update({i:prepData})
    return dictOfData
        
class breakout():
    #Models to make: BIRCH, K_Means, Spectral, Mixture of Gaussians, Maybe: Optics 
    def __init__(self, clusterSearch, cluster, clusterName, cluster_params, run_params, data):
        self.cluster = cluster
        self.clusterName = clusterName
        self.data = data
        self.clusterSearch = clusterSearch
        self.run_params = run_params
        #To be plased out or updated in future version
        self.cluster_params = cluster_params

    def startCluster(self):
        print("Start Cluster")
        if(self.clusterSearch):
            for i in range(len(self.clusterName)):
                self.clustModel[i] = model(self.cluster[i], self.clusterName[i], self.cluster_params[i][0], self.data, True)
                self.clustModel[i].train(self.cluster_params[i][4], self.cluster_params[i][5])
        else:
            self.clustModel = model(self.cluster, self.clusterName, self.cluster_params[0], self.data, True)
            self.clustModel.train(self.run_params[0], self.run_params[1])
    def createData(self, prepTrain, data='na'):
        print("Create Data")
        self.newData = pd.DataFrame
        if prepTrain:
            if(self.clusterSearch):
                self.chosenClustModel = str(input("What cluster should be used?\n")).upper()
                for i in range(len(self.clusterName)):
                    if(self.clusterModel[i].modelType == self.chosenClustModel):
                        self.chosenClustModel = self.clustModel[i]
                        break 
                self.predTMPTrain = self.chosenClustModel.modPredict(self.data.X_train, 'na')
                self.predTMPTest = self.chosenClustModel.modPredict(self.data.X_test, 'na')
                self.newData = sortGroups(self.data.X_test, self.predTMPTrain, self.predTMPTest, self.chosenClustModel.groups)
            else:
                self.predTMPTrain, self.predGroups = self.clustModel.modPredict(self.data.X_train, 'na')
                self.predTMPTest, tmp = self.clustModel.modPredict(self.data.X_test, 'na')
                self.newData = sortGroups(self.data, self.predTMPTrain, self.predTMPTest, self.predGroups)
        else:
            if(self.clusterSearch):
                raise Exception("Functionn not built yet, can not cluster search")
            else:
                self.tmpX = data[0]
                self.tmpy = data[1]
                self.predTMP, self.predGroups = self.clustModel.modPredict(self.tmpX, 'na')
                #self.TMPx = self.predTMP.drop(yCol, axis=1, inplace=False)
                self.predTMP = pd.concat([self.TMPx, self.predTMP], axis=1)
                tmpidx = len(self.predTMP.columns)-1
                tmpCol = self.predTMP.columns[tmpidx]
                for i in range(len(self.predGroups)):
                    self.newData.update({1:pd.DataFrame(self.predTMP[self.predTMP[tmpCol] == self.predGroups[i]])})                   
        
    def analyzeClusterSearch(self):
        #Fix this later
        #Update your predict funtion to use a specified dataframe
        #Also change the current name of your predict funtion as it's used for testing
        print("Analyze Search")
        self.gamBase = pg.LinearGAM(n_splines=30, verbose=True, max_iter=500)
        if(self.clusterSearch):    
            self.results = [[0]*len(self.clusterName)]*2
            for i in range(len(self.clusterName())):
                    self.testingData[i] = self.clustModel[i].group()
                    #train example models
                    for i in range(len(self.testingData)):
                        self.testingmodel[i] = model(self.gamBase, 'GAM', 'NA', self.testingData)
                        self.testingmodel[i].train()
                        self.resutls[i][0] = self.testingModel[i].predict(True)
                        self.results[i][1] = self.testingmodel.modelType()
                    print("Done evaluating")
        else:
            self.testingData = self.newData
            for i in range(len(self.testingData)):
                self.testingmodel[i] = model(self.gamBase, 'GAM', 'NA', self.testingData[i])
                self.testingmodel[i].train()
                self.resutls[i] = self.testingModel[i].predict(True)
                
    def trainSecond(self, secondLayerModel):
        print("Train Second")
        self.secondLayerModel = secondLayerModel
        if(self.clusterSearch):
            #Add code later
            raise Exception("Code not created yet")
        else:
            self.models = dict()
            for i in range(len(self.newData)):
                self.tmpModel = self.secondLayerModel
                self.breakoutModel = model(self.tmpModel, 'GAM', 'NA', self.newData[i])
                self.breakoutModel.train(1, 1)
                self.models.update({i:self.breakoutModel})
                
    def testSecond(self):
        print("Test Second")
        self.results = dict()
        if(self.clusterSearch):
            #Add code later
            raise Exception("Code not created yet")
        else:
            for i in range(len(self.models)):
                self.results.update({i:self.models[i].modPredict(self.newData[i].X_test, self.newData[i].y_test, True)})
   
    def modPredict(self, X, y, returnResults=False, returnPred=False, firstDoneData=False):
       data = pd.concat([X, y], axis=1)
       self.preds = dict()
       if not firstDoneData:
           groupedData = self.createData(False, data)
           for i in range(self.models):
               tmpdata = groupedData[i]
               ycol = tmpdata.columns[len(tmpdata-2)]
               tmpx = tmpdata.drop(ycol)
               tmpy = tmpdata[ycol]
               tmppred = self.models[i].modPredict(tmpx, tmpy, False, True)
               self.preds.update({i:tmppred})
       elif firstDoneData:
           for i in range(self.models):
               tmpdata = data[i]
               ycol = tmpdata.columns[len(tmpdata-2)]
               tmpx = tmpdata.drop(ycol)
               tmpy = tmpdata[ycol]
               tmppred = self.models[i].modPredict(tmpx, tmpy, False, True)
               self.preds.update({i:tmppred})
               
       if returnResults and returnPred:
           raise Exception("BAD you make return results true")
       elif returnPred:
           return self.preds
       elif returnResults:
           raise Exception("lol you make return results true")
           
    
baseInput = []
'''   
print("0")
r2016 = pd.read_csv("BS_2016.csv")
r2016 = r2016[0:int(len(r2016)/5)]
print("1")
allData = data([r2016])
print("2")
allData.baseFilter(["POAI", "GHI", "Hour", "kW"], 'na', 0, 'mean')
allData.summary()
allData.lagFuture('kW', ["POAI", "GHI"], "Hour", 3, 2)
allData.split(.2, .1, 'kW', allData.input_cols)
baseInput = allData.input_cols

birchTest = Birch(branching_factor=150, n_clusters=5)
btrun_params = [1, 1]
test = breakout(False, birchTest, "birch", 'na', btrun_params, allData)
test.startCluster()
test.createData()

breakoutGam = pg.LinearGAM(n_splines=30, verbose=True, max_iter=500)
test.trainSecond(breakoutGam)
test.testSecond()
#test.analyzeClusterSearch()
'''
#To test our ensemble and such
'''
gam = pg.LinearGAM(n_splines=34, verbose=True, max_iter=500)
gamL1 = model(gam, 'GAM', 'NA', allData)
gamL1.train(1, 500)   

normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(allData.X_train)
dnn_model = keras.Sequential([
    normalizer,
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='sigmoid'),
    layers.Dense(1)
])
dnn_model.compil e(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(.001))
dnnL1 = model(dnn_model, 'DNN', 'NA', allData)
dnnL1.train(1, 10)
dnnL1.predict()
 
gam2 = pg.LinearGAM(n_splines=15, verbose=True, max_iter=500)
gam2L1 = model(gam, 'GAM', 'NA', allData)
gam2L1.train(1, 500)   

dnn_model1 = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='sigmoid'),
    layers.Dense(1)
])
dnn_model1.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(.001))

dnn_model2 = keras.Sequential([
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='sigmoid'),
    layers.Dense(1)
])
dnn_model2.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(.001))

gam2 = pg.LinearGAM(n_splines=34, verbose=True, max_iter=500)
gam3 = pg.LinearGAM(n_splines=34, verbose=True, max_iter=500)

futureModels = [[['NA','NA','NA'],['NA','NA','NA'],['NA','NA','NA']], [[gam2, "GAM", 'NA'],[dnn_model1, "DNN", "NA"]]]
lastList = [[gam3, "GAM", "NA"], [dnn_model2, "DNN", "NA"]]

multiModel = ensemble([[gamL1, dnnL1, gam2L1],["gamL2", "dnnL2"]] , futureModels, allData)
multiModel.create_Base()
multiModel.lastLayer(lastList)
'''