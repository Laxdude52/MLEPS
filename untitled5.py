# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 07:24:04 2023

@author: School Account
"""

import PySimpleGUI as sg

layout = [
    [sg.Text("Select CSV Files:")],
    [sg.Input(key="-FILES-"), sg.FilesBrowse(file_types=(("CSV Files", "*.csv"),))],
    [sg.Button("Submit"), sg.Button("Exit")]
]

window = sg.Window("CSV File Selector", layout)

while True:
    event, values = window.read()

    if event == sg.WINDOW_CLOSED or event == "Exit":
        break
    elif event == "Submit":
        selected_files = values["-FILES-"].split(";")
        print("Selected Files:")
        for file_path in selected_files:
            print(file_path)

window.close()
