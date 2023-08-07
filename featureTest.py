# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 17:38:14 2023

@author: every
"""

import PySimpleGUI as sg

# Create some sample elements to group together
feature_group1 = [
    [sg.Text('Feature 1:')],
    [sg.Input(key='-FEATURE1-')],
    [sg.Checkbox('Option 1', key='-OPTION1-')],
    [sg.Checkbox('Option 2', key='-OPTION2-')],
]

feature_group2 = [
    [sg.Text('Feature 2:')],
    [sg.Input(key='-FEATURE2-')],
    [sg.Radio('Radio 1', group_id='RADIO_GROUP', key='-RADIO1-')],
    [sg.Radio('Radio 2', group_id='RADIO_GROUP', key='-RADIO2-')],
]

# Create the layout with the grouped features
layout = [
    [sg.Frame('Feature Group 1', feature_group1)],
    [sg.Frame('Feature Group 2', feature_group2)],
    [sg.Button('Submit')],
]

# Create the window
window = sg.Window('Grouping Features', layout)

# Event loop
while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED:
        break
    elif event == 'Submit':
        feature1_value = values['-FEATURE1-']
        option1_value = values['-OPTION1-']
        option2_value = values['-OPTION2-']
        feature2_value = values['-FEATURE2-']
        radio_value = 'Radio 1' if values['-RADIO1-'] else 'Radio 2'

        sg.popup(f'Feature 1: {feature1_value}\nOption 1: {option1_value}\nOption 2: {option2_value}\n'
                 f'Feature 2: {feature2_value}\nSelected Radio: {radio_value}')

# Close the window
window.close()

'''
    solarLCol = [
        [sg.Text('Type')],
        [sg.Text('Solar')],
        [sg.Text('POAI')],
        [sg.Text('GHI')],
        [sg.Text('TmpF')],
        ]
    
    solarMCol = [
        [sg.Text('Real Time')],
        [sg.Column(
        [sg.DropDown(values=['Option 1', 'Option 2'], key='-Solar RDrop-')], 
        [sg.DropDown(values=['Option 1', 'Option 2'], key='-POAI RDrop-')], 
        [sg.DropDown(values=['Option 1', 'Option 2'], key='-GHI RDrop-')], 
        [sg.DropDown(values=['Option 1', 'Option 2'], key='-TmpF RDrop-')], 
        )],
        
        [sg.Column(
            [sg.Button("View", key='Solar RView', size=(15,2))],
            [sg.Button("View", key='POAI RView', size=(15,2))],
            [sg.Button("View", key='GHI RView', size=(15,2))],
            [sg.Button("View", key='TmpF RView', size=(15,2))],
        ),
        ],
        ]
    
    
    solarRCol = [
        [sg.Text('Future')],
        [sg.DropDown(values=['Option 1', 'Option 2'], key='-Solar FDrop-'), sg.Button("View", key='Solar FView', size=(15,2))],
        [sg.DropDown(values=['Option 1', 'Option 2'], key='-POAI FDrop-'), sg.Button("View", key='POAI FView', size=(15,2))],
        [sg.DropDown(values=['Option 1', 'Option 2'], key='-GHI FDrop-'), sg.Button("View", key='GHI FView', size=(15,2))],
        [sg.DropDown(values=['Option 1', 'Option 2'], key='-TmpF FDrop-'), sg.Button("View", key='TmpF FView', size=(15,2))],
        ]
    
    left_columnR = [
        [sg.Column(solarLCol)],
        [sg.VerticalSeparator()],
        [sg.Column(solarMCol)],
        [sg.VerticalSeparator()],
        [sg.Column(solarRCol)],
        ]
    right_columnR = [
        [sg.Button('Wind', key='Wind', size=(20,3))],
        ]
    renewables_layout = [
        [
            sg.Column(left_columnR),
            sg.VerticalSeparator(),
            sg.Column(right_columnR),
            ]
        ]
    '''