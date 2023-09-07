# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 17:02:41 2023

@author: School Account
"""

import pandas as pd

dataLists = dict()

def createList(files, storedName):
    #files is a list
    tmpDataList = dict()
    for i in range(len(files)):
        tmpData = pd.read_csv(files[i])
        tmpDataList.update({i:tmpData})
    dataLists.update({storedName: tmpDataList})
    
    
'''
#Save all Datasets hee
r2016 = pd.read_csv("BS_2016.csv")
r2017 = pd.read_csv("BS_2017.csv")
r2018 = pd.read_csv("BS_2018.csv")
r2019 = pd.read_csv("BS_2019.csv")
r2020 = pd.read_csv("BS_2020.csv")
r2021 = pd.read_csv("BS_2021.csv")
r2022 = pd.read_csv("BS_2022.csv")

solarDataList = [r2016, r2017, r2018, r2019, r2020, r2021, r2022]

windDataList = 'NA'
energyDemandList = 'NA'
coalList = 'NA'

hydroList1 = 'NA'
hydroList2 = 'NA'

combList1 = 'NA'
combList2 = 'NA'
combList3 = 'NA'
combList4 = 'NA'
combList5 = 'NA'
combList6 = 'NA'
combList7 = 'NA'
'''