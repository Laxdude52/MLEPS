# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 13:39:12 2023

@author: School Account
"""

import PySimpleGUI as sg
import createModels as cm
import data as dd

a = []
defaultDirectory = r'C:\Users\every\Desktop\testMLEPS'


def modelManageWindow():
    print("Start Model Manage Window")
    
#Depreciated 
   
def loadDataWindow(files):
    window.hide()
    loadData_layout = [
        [sg.Text("Here are the files you selected:")],
        [sg.Listbox(files, size=(40, 5), disabled=True)],
        [sg.Input(default_text='Stored File Name', enable_events=True, key='-Stored Filename-', size=(30,3))],
        [sg.Button("Save", key='-Save List-', size=(20,2), disabled=True)],
        [sg.Button("Change Dataframes", key='-New Dsets-', size=(20,2))],
        ]
    
    loadData_window = sg.Window("Load Csv Files", loadData_layout)
    
    while True:
        event_loadData, values_loadData = loadData_window.read()
        if (event_loadData == sg.WIN_CLOSED):
            loadData_window.close()
            break
        elif event_loadData == '-New Dsets-':
           loadData_window.close()
           new_dataset = sg.popup_get_file('Multi-File select', multiple_files=True)
           loadDataWindow(new_dataset)
        elif event_loadData == '-Save List-':
            name = values_loadData['-Stored Filename-']
            dd.createList(files, name)
        elif event_loadData == '-Stored Filename-':
            loadData_window.Element('-Save List-').update(disabled=False)

def dataManageWindow():
    loaded_lists = dd.dataLists
    tmpData = cm.models
    print("Start Data Management Window")
    window.hide()
    
    left_column = [
        [sg.Text("Manage Datasets")],
        [sg.Button('Load Data', key='-Load Data-', size=(20,3))],
        [sg.Button('Create Data', key='-Create Data-', size=(20,3))],
        [sg.Text("Loaded dataset lists:")],
        #Include the loaded dset lists on the left 
        [sg.Listbox(loaded_lists, size=(30,5), key='-selected lists-', enable_events=True)],
        [sg.Button("Edit Selected List", key='-edit list-', size=(20,3))],
        [sg.Button('Back', key='Exit', size=(10,2))],
        ]
    
    overall_types = ['all']
    tmpTypeKeys = list(tmpData.keys())
    for i in range(len(tmpTypeKeys)):
        overall_types.append(tmpTypeKeys[i])
    
    specified_types = []
    loaded_dsets = []
    
    right_column = [
        [sg.Text("Loaded Datasets")],
        [sg.Text("Overall Dataset Types")],
        [sg.Listbox(overall_types, size=(20,4), key='-overall_type-', select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, enable_events=True)],
        [sg.Text("Specified Dataset Types")],
        [sg.Listbox(specified_types, size=(20,4), key='-specified_type-', select_mode=sg.LISTBOX_SELECT_MODE_MULTIPLE, enable_events=True, disabled=True)],
        [sg.Text("Specified Dataset Types")],
        [sg.Listbox(loaded_dsets, size=(20,7), key='-datasets-', enable_events=True, disabled=True)],
        [sg.Button('Manage Selected Dataframe', key='-Manage Selected Data-', size=(20,2))],
        ]
     
    dataManage_layout = [
        [
            sg.Column(left_column),
            sg.VerticalSeparator(),
            sg.Column(right_column),
            ]
        ]
    
    dataManage_window = sg.Window('Data Management', dataManage_layout)
    
    while True:
        event_dataManage, values_dataManage = dataManage_window.read()
        loaded_dsets = []
        
        if (event_dataManage == sg.WIN_CLOSED) or (event_dataManage == 'Exit'):
            dataManage_window.close()
            window.un_hide()
            break
        elif event_dataManage == '-overall_type-':
            specified_types = []
            chosenTypes = list(values_dataManage['-overall_type-'])
            for i in range(len(chosenTypes)):
                if chosenTypes[i] == 'all':
                    for key in tmpData.keys():
                        for key1 in tmpData[key]:
                            specified_types.append(key1)
                    break
                else:
                    tmpKeys = list(tmpData[chosenTypes[i]].keys())
                    for j in range(len(tmpKeys)):
                        specified_types.append(tmpKeys[j])

            dataManage_window.Element('-specified_type-').update(values=specified_types, disabled=False)
        elif event_dataManage == '-specified_type-':
            for i in range(len(chosenTypes)):
                selectedKeys = list(values_dataManage['-specified_type-'])
                #Add all functionality later
                '''
                if selectedKeys[i] == 'all':
                    for j in range(len(specified_types)):
                        tmpNameS = specified_types[j] + ' Simple Data'
                        tmpNameF = specified_types[j] + ' Future Data'
                        loaded_dsets.append(tmpNameS)
                        loaded_dsets.append(tmpNameF)
                        break
                '''
                for j in range(len(selectedKeys)):
                    print(selectedKeys)
                    tmpNameS = selectedKeys[j] + ' Simple Data'
                    tmpNameF = selectedKeys[j] + ' Future Data'
                    loaded_dsets.append(tmpNameS)
                    loaded_dsets.append(tmpNameF)
                    '''
                    loaded_dsets.append(tmpData[chosenTypes[i]][selectedKeys[j]]['simpData'])
                    loaded_dsets.append(tmpData[chosenTypes[i]][selectedKeys[j]]['futureData'])
                    '''
            
            a = loaded_dsets
            dataManage_window.Element('-datasets-').update(values=loaded_dsets, disabled=False)
        elif event_dataManage == '-Create Data-':
            #new_dataset = loadDataWindow()
            new_dataset = sg.popup_get_file('Multi-File select', multiple_files=True)
            new_dataset = new_dataset.split(";")
            loadDataWindow(new_dataset)
            print(new_dataset) 

# Define the layout of the left column
left_column = [
    [sg.Button('Manage Data', key='-Data-', size=(20, 3))],
    [sg.Button('Manage Models', key='-Models-', size=(20, 3))],
]

# Define the layout of the right column
right_column = [
    [sg.Button('Energy Management System', key='-EMS-', size=(25, 8))],
    # Add more buttons on the right side if needed
]

# Combine the left and right columns in the main layout
layout = [
    [
        sg.Column(left_column),
        sg.VerticalSeparator(),
        sg.Column(right_column),
    ]
]

# Create the window
window = sg.Window('Energy Management GUI', layout)

# Event loop
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break
    elif event == '-Data-':
        print("Data")
        dataManageWindow()
    elif event == '-Models-':
        print("Models")
        modelManageWindow()
    elif event == '-EMS-':
        print("EMS")
        #EMSWindow()

window.close()