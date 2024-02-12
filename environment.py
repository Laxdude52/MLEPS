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

import warnings
import tensorflow as tf
import numpy as np
import data as ud
import defaultSimulation as sim
import logSimulation as los
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
MAXTIME = 44316
MAX_DEMAND = 3795
MAX_RAMP_RATE = 350
MAX_SOLAR = 10
solarModelInfo = ['default', 'Solar']
currentTime = 0

# Define environment parameters
state_dim = 31  # Define the dimension of the state space
action_dim = 100572000000 # Define the dimension of the action space 11524500000000000
learning_rate = 0.001
gamma = 0.99  # Discount factor 

# Get current plant levels + max ramp up/down
def getAllPlantInformation():
    return ud.plantInformation

#Get Plant information
def getSpecPlantInformation(plant):
    return ud.plantInformation[plant]


# Get solar pred
def getSolarPred(currentTime):
    # print(currentTime)
    pred = ud.livePredict(currentTime, 'Solar', 'Future', 'default', numPast=2)
    if pred > MAX_SOLAR:
        pred = MAX_SOLAR
    return pred

#Get current solar output for live simulation
def getCurrentSolar(currentTime):
    try:
        data = ud.dataModels['Solar']['Simple']['default']['initData'].y_train.iloc[[currentTime]]
    except:
        data = ud.dataModels['Solar']['Simple']['default']['initData'].y_test.iloc[[currentTime-len(ud.dataModels['Solar']['Simple']['default']['initData'].y_train)]]
    data = data.item()
    if data > MAX_SOLAR:
        data = MAX_SOLAR
    return int(data)


# Get predicted electricity demand
def fgetElecDemand(currentTime):
    # if(currentTime <= 5):
    demand = ud.livePredict(currentTime, 'Demand', 'Future', 'default', numPast=5)
    demand = demand[0][0]
    if demand > MAX_DEMAND:
        return MAX_DEMAND
    else:
        return demand

# Get real electricity demand
def rgetElecDemand(currentTime):
    #Consider two datasets
    try:
        data = ud.dataModels['Demand']['Simple']['default']['initData'].y_train.iloc[[currentTime]]
    except:
        data = ud.dataModels['Demand']['Simple']['default']['initData'].y_test.iloc[[currentTime-len(ud.dataModels['Demand']['Simple']['default']['initData'].y_train)]]
    data = data[0][0]
    if data > MAX_DEMAND:
        data = MAX_DEMAND
    return data

#Get total current set plant production
def getCurrentProduction(currentTime):
    tmpCurProd = 0
    for plant in sim.plantList:
        tmpPlantInfo = getSpecPlantInformation(plant)
        tmpCurProd = tmpCurProd + tmpPlantInfo["CurrentLevel"]
    tmpCurProd = int(tmpCurProd+int(getCurrentSolar(currentTime)))
    #print(tmpCurProd)
    return tmpCurProd

#Reset the simulation
def resetSimulation():
    for plant in sim.plantList:
        ud.setPlantLevel(plant, ud.plantInformation[plant]["Min"])

#Get the most efficent plant output level
#Due to data constraints, the highest available output level is always the most efficent, so it only has access to half of the max ramp
def getMostEfficent(rampDir, plant):
    mostEfficentVal = 90000000
    mostEfficentSpeed = 90000000
    i = 0
    data = plant["efficencyData"].iloc[:, 1]
    # print("CurLevel" + str(plant["CurrentLevel"]))
    currentLevel = plant["CurrentLevel"]
    # print(plant)
    # print("CurLevel" + str(currentLevel))
    maxOut = plant["Max"] - 1
    if (maxOut < currentLevel + int(plant["MaxRamp"]/2)):
        upperBound = maxOut
    else:
        upperBound = int(currentLevel + (plant["MaxRamp"] / 2))
    if ((plant['Min'] + 1) > currentLevel - int(plant["MaxRamp"]/2)):
        lowerBound = plant['Min'] + 1
    else:
        lowerBound = currentLevel - int(plant["MaxRamp"]/2)
    if (rampDir == 1):
        # print("Cur: " + str(currentLevel) + "\nUppB: " + str(upperBound))
        for i in range(currentLevel, upperBound):
            if (data[i] < mostEfficentVal):
                mostEfficentVal = data[i]
                mostEfficentSpeed = i
        # print("E level" + str(mostEfficentSpeed))
    elif (rampDir == -1):
        # print("Down")
        # print("CurLevel: (Loop) " + str(currentLevel))
        # print("lowBound: (Loop) " + str(lowerBound))
        if (int(currentLevel - plant['MaxRamp'] / 2) > lowerBound):
            currentLevel = int(currentLevel - (plant['MaxRamp'] / 2))
        for i in range(lowerBound, currentLevel):
            if (data[i] < mostEfficentVal):
                mostEfficentVal = data[lowerBound]
                mostEfficentSpeed = lowerBound
    else:
        # print(rampDir)
        mostEfficentSpeed = currentLevel
        # raise Exception("No dir")
    if (mostEfficentSpeed >= plant["Max"]):
        mostEfficentSpeed = plant["Max"] - 1
    elif (mostEfficentSpeed <= plant["Min"]):
        mostEfficentSpeed = plant["Min"] + 1
    # print("Plant new level: " + str(mostEfficentSpeed) + " Plant max: " + str(plant['Max']))
    heatRate = plant["efficencyData"].iloc[:, 0][mostEfficentSpeed]
    return mostEfficentSpeed, heatRate, mostEfficentVal


def futureAlgorithm(currentTime):
    #Reinforcement model here
    print()

def currentAlgorithm(currentTime):
    tmpRealDemand = int(rgetElecDemand(currentTime))
    # print("Real elec demand: " + str(tmpRealDemand))
    tmpRealSolar = int(getCurrentSolar(currentTime))

    for plant in sim.plantList:
        tmpPlantInfo = getSpecPlantInformation(plant)
        tmpCurProd = getCurrentProduction(currentTime)
        tmpDem = tmpRealDemand - tmpCurProd
        # print(str(plant) + " Max Ramp: " + str(tmpPlantInfo["MaxRamp"]) + "  Current Level: " + str(tmpPlantInfo["CurrentLevel"]))
        tmpGap = tmpDem - tmpCurProd
        print("Gap: ", tmpGap)
        # print("Left Ramp" + str(tmpPlantInfo["LeftRamp"]))
        if (abs(tmpGap) < tmpPlantInfo["LeftRamp"]):
            newLevel = tmpPlantInfo["CurrentLevel"] + tmpGap
        else:
            if (tmpGap > 0):
                newLevel = tmpPlantInfo['CurrentLevel'] + tmpPlantInfo['LeftRamp']
            elif (tmpGap < 0):
                newLevel = tmpPlantInfo['CurrentLevel'] - tmpPlantInfo['LeftRamp']
            else:
                newLevel = tmpPlantInfo['CurrentLevel']
        if (newLevel >= tmpPlantInfo["Max"]):
            newLevel = tmpPlantInfo["Max"] - 1
        elif (newLevel <= tmpPlantInfo["Min"]):
            newLevel = tmpPlantInfo["Min"] + 1
        ud.setPlantLevel(plant, newLevel)
        ud.resetPlantRampLeft(plant)
        # print("Live level: " + str(getSpecPlantInformation(plant)["CurrentLevel"]))
        try:
            heat = tmpPlantInfo["efficencyData"].iloc[:, 0][newLevel]
        except Exception as e:
            print(e)
            print(newLevel)
            print(tmpPlantInfo['Max'])
            print(plant)
            Exception(e)
        heatPerOut = tmpPlantInfo["efficencyData"].iloc[:, 1][newLevel]
        los.logCurStep(plant, newLevel, heat, heatPerOut)
    los.logCurFutStep(currentTime, tmpRealDemand, tmpRealSolar)


'''
xVals = []
yVals = []
fig = plt.figure()

def liveGraph(xVals, yVals):
    ax1 = fig.add_subplot(1,1,1)
    xVals.append(currentTime)
    yVals.append(los.logFutDict["totalProd"][currentTime])
    xVals = xVals[-30:]
    yVals = yVals[-30:]
    ax1.clear()
    ax1.plot(xVals, yVals)  
    
    plt.title("Production over time (Future)")

'''

# sim.initDefaultSimulation()
sim.initDefaultSimulation(load=False, save=False)

# Define the neural network model for each production unit
class ProductionUnitModel(tf.keras.Model):
    def __init__(self):
        super(ProductionUnitModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.dense3 = tf.keras.layers.Dense(32, activation='relu')
        self.dense4 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

# Create a list of models, one for each production unit
num_units = 7
models = [ProductionUnitModel() for _ in range(num_units)]
optimizer = tf.keras.optimizers.Adam(learning_rate)

# Define the preprocessing function for the state
def preprocess_state(state):
    normalized_state = []
    
    # Normalize current time
    normalized_state.append(state[0] / MAXTIME)
    
    # Normalize predicted demand
    normalized_state.append(state[1] / MAX_DEMAND)
    
    # Normalize solar data
    normalized_state.append(state[2] / MAX_SOLAR)
    
    # Normalize information about each power plant
    for i in range(3, len(state), 4):
        current_level = state[i]
        max_ramp_rate = state[i + 1]
        min_level = state[i + 2]
        max_level = state[i + 3]
        
        # Normalize current level
        normalized_current_level = (current_level - min_level) / (max_level - min_level)
        
        # Normalize maximum ramp rate
        normalized_ramp_rate = max_ramp_rate / MAX_RAMP_RATE
        
        # Append normalized values to the state
        normalized_state.extend([normalized_current_level, normalized_ramp_rate, min_level, max_level])
    
    normalized_state = np.array(normalized_state)
    normalized_state = np.reshape(normalized_state, (1, -1))
    return normalized_state.astype('float32')
'''
# Define the function to choose actions for each production unit
def choose_actions(states):
    actions = []
    for i, state in enumerate(states):
        action = models[i](state)
        actions.append(action)
    return actions
'''

def choose_action_from_distribution(action_probabilities, min_level, max_level, max_ramp_rate, current_level):
    # Scale action probabilities to the allowable range
    scaled_probabilities = np.zeros_like(action_probabilities)
    allowable_range = np.arange(min_level, max_level + 1)  # Allowable range of actions

    for i in range(len(allowable_range)):
        scaled_probabilities[i] = action_probabilities[i]
    scaled_probabilities /= np.sum(scaled_probabilities)  # Normalize probabilities
    
    # Choose action based on scaled probabilities
    action = np.random.choice(allowable_range, p=scaled_probabilities)
    
    # Ensure action is within the bounds of the maximum ramp rate
    action = min(action, current_level + max_ramp_rate)
    action = max(action, current_level - max_ramp_rate)
    
    # Ensure action is within the allowable range
    action = min(max(action, min_level), max_level)
    
    return action


def choose_action(states):
    # Assuming state contains information about current production levels for each unit
    # You need to preprocess the state to make it suitable for feeding into the policy network
    actions = []
    for i, plant in enumerate(ud.plantInformation):
        state = states[i]
        processed_state = preprocess_state(state)
    
        # Pass the processed state through the policy network to get logits
        logits = models[i](processed_state)
        
        # Apply softmax to logits to get probabilities
        probabilities = tf.nn.softmax(logits)
        
        # Convert probabilities tensor to numpy array
        probabilities_np = probabilities.numpy().squeeze()
        
        current_level = state[i * 4 + 3]  # Current level of the unit
        max_ramp_rate = state[i * 4 + 4]  # Maximum ramp rate of the unit
        MIN_PRODUCTION_LEVEL = state[i * 4 + 5]  # Minimum production level of the unit
        MAX_PRODUCTION_LEVEL = state[i * 4 + 6]  # Maximum production level of the unit
        
        # Calculate the allowable range of production levels based on the maximum ramp rate
        min_level = max(current_level - max_ramp_rate, MIN_PRODUCTION_LEVEL)
        max_level = min(current_level + max_ramp_rate, MAX_PRODUCTION_LEVEL)
        
        # Choose the action based on the probability distribution
        chosen_action = choose_action_from_distribution(probabilities_np, min_level, max_level, max_ramp_rate, current_level)
        actions.append(chosen_action)
        ud.setPlantLevel(plant, chosen_action)
        
    return actions


def get_state(unit, current_time):
    # Get predicted demand and solar data for the current time
    predicted_demand = fgetElecDemand(current_time)
    solar_data = getSolarPred(current_time)
    
    # Get current levels and maximum ramp rates of all power plants
    plant_info = dict()
    for unit in range(len(sim.plantList)):
        current_level = getSpecPlantInformation(sim.plantList[unit])["CurrentLevel"]
        max_ramp_rate = getSpecPlantInformation(sim.plantList[unit])["MaxRamp"]
        min_level = getSpecPlantInformation(sim.plantList[unit])["Min"]
        max_level = getSpecPlantInformation(sim.plantList[unit])["Max"]
        plant_info.update({
            "current_level": current_level,
            "max_ramp_rate": max_ramp_rate,
            "min_level": min_level,
            "max_level": max_level
        })
    
    # Construct state array including time, demand, solar data, current levels, and maximum ramp rates for the specified unit
    state = [current_time, predicted_demand, solar_data]
    state.extend([plant_info["current_level"], plant_info["max_ramp_rate"], plant_info["min_level"], plant_info["max_level"]])
    
    return np.array(state)

# Define the reward calculation function
def calculate_reward(total_production, predDemand):
    # Calculate the difference between current production and predicted demand
    production_diff = abs(getCurrentProduction(currentTime) - predDemand)
    
    # Efficiency reward based on how close the current production is to the maximum production
    if(production_diff == 0):
        efficiency_reward=.9
    else:
        efficiency_reward = 1/production_diff
    
    # Total reward as a combination of matching demand and efficiency
    total_reward = efficiency_reward
    return total_reward

# Define loss function
def compute_loss(logits, actions, rewards):
    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=actions)
    loss = tf.reduce_mean(neg_log_prob * rewards)
    return loss

def getDone(currentTime):
    if(currentTime >= 10):
        return True
    else:
        return False

# Function to interact with environment and collect experience
def run_episode(currentTime):
    allStates = []
    allActions = []
    allRewards = []
    # Get current states for all production units
    states = [get_state(unit, currentTime) for unit in ud.plantInformation]    
    episode_reward = 0
    done = False
    while not done:
        actions = choose_action(states)
        print(actions)

        # After updating all production levels, calculate reward
        total_production = getCurrentProduction(currentTime)
        predDemand = fgetElecDemand(currentTime)
        reward = calculate_reward(total_production, predDemand)
        next_states = get_state(currentTime)
        done =  getDone(currentTime)
        episode_reward += reward
        
        allStates.append(states)
        allActions.append(actions)
        allRewards.append(reward)
        
        states = next_states
    return states, actions, rewards, episode_reward

# Training loop
num_episodes = 100
for episode in range(num_episodes):
    if currentTime > MAXTIME:
        currentTime = 0
    states, actions, rewards, episode_reward = run_episode(currentTime)

    # Calculate discounted rewards
    discounted_rewards = []
    cumulative_rewards = 0
    for reward in reversed(rewards):
        cumulative_rewards = reward + gamma * cumulative_rewards
        discounted_rewards.append(cumulative_rewards)
    discounted_rewards.reverse()

    # Normalize discounted rewards
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    # Convert lists to TensorFlow tensors
    states = tf.convert_to_tensor(states, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)
    discounted_rewards = tf.convert_to_tensor(discounted_rewards, dtype=tf.float32)

    for policy_network in models:
        # Train the policy network
        with tf.GradientTape() as tape:
            logits = policy_network(states)
            loss = compute_loss(logits, actions, discounted_rewards)
        gradients = tape.gradient(loss, policy_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))

    print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
    
xVals = []
production = []
predDemand = []
realDemand = []
plt.ion()
figure, ax = plt.subplots(figsize=(10, 8))
line1, = ax.plot(xVals, production, label="Total Production")
line2, = ax.plot(xVals, predDemand, label="Predicted Electricity Demand")
line3, = ax.plot(xVals, realDemand, label="Real Electricity Demand")


def livePlot(xVals, production, realDemand, currentTime):
    xVals.append(currentTime)
    production.append(los.logCurDict["totalProd"][currentTime])
    # print("CurPlotProd: " + str(los.logCurDict["totalProd"][currentTime]))
    predDemand.append(los.logFutDict["elecDem"][currentTime])
    realDemand.append(los.logCurDict["elecDem"][currentTime])

    '''
    xVals = xVals[-72: ]
    production = production[-72: ]
    predDemand = predDemand[-72: ]
    realDemand = realDemand[-72: ]
    '''

    # Set x-axis limit to show a specific range of x values
    if (currentTime <= 72):
        ax.set_xlim(min(xVals), max(xVals))
    else:
        ax.set_xlim(currentTime - 72, currentTime)

    # Set y-axis limit to show a specific range of y values
    ax.set_ylim(0, 4000)

    # print("Lengths - xVals:", len(xVals), "production:", len(production), "predDemand:", len(predDemand), "realDemand:", len(realDemand))

    # Update x-data
    line1.set_xdata(xVals)
    line2.set_xdata(xVals)
    line3.set_xdata(xVals)

    # Update y-data
    line1.set_ydata(production)
    line2.set_ydata(predDemand)
    line3.set_ydata(realDemand)

    plt.title("Production, Predicted Demand, Real Demand (MWh)")
    plt.legend(loc="upper left")
    figure.canvas.draw()
    figure.canvas.flush_events()
    # plt.pause(0.1)
    # time.sleep(.1)
    # https://www.youtube.com/watch?v=Ercd-Ip5PfQ


resetSimulation()
while currentTime != MAXTIME:
    futureAlgorithm(currentTime)

    currentAlgorithm(currentTime)
    livePlot(xVals, production, realDemand, currentTime)

    # print("Current Time: " + str(currentTime))
    currentTime = currentTime + 1

sim.saveAll()
