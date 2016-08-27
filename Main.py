# Diabetes Prediction using Naive Bayes model
#	Made By - Anshuman Dixit

import csv
import random
import math

################## Load the csv file ##################################

def loadcsv(filename):
        lines = csv.reader(open(filename,"rb"))
        dataset = list(lines)
        for i in range(len(dataset)):
                dataset[i] = [float(x) for x in dataset[i]]
        return dataset

################## Split the data into train and test #################

def splitDataset(dataset,splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]

################## Separate instances using Class Value ###############    

def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated  

################## Mean of an attribute ###############################    

def mean(numbers):
    return sum(numbers)/float(len(numbers))

################## Standard Deviation of an attribute #################    

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float   (len(numbers)-1)
    return math.sqrt(variance)

################## Summarize the mean and stdev #######################    

def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)] 
    del summaries[-1]
    return summaries

################## Summarize with Class Value #########################    

def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.iteritems():
        summaries[classValue] = summarize(instances)
    return summaries

################## Gaussian Probability Density Function ##############    

def calculateProbability(x,mean,stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return (1/(math.sqrt(2*math.pi)*stdev)) * exponent

################## Calculate Probability ##############################    

def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.iteritems():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean,stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x,mean,stdev)
    return probabilities

################### Make Predictions ###################################    

def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries,inputVector)
    bestlabel, bestProb = None, -1
    for classValue, probability in probabilities.iteritems():
        if bestlabel is None or probability > bestProb:
         bestProb = probability
         bestlabel = classValue
    return bestlabel

################### Make Predictions for the test dataset ##############

def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

################### Get the accuracy of the model ######################

def getAccuracy(testSet, predictions):
 correct = 0
 for x in range(len(testSet)):
    if testSet[x][-1] == predictions[x]:
        correct += 1
    return (correct/float(len(testSet))) * 100.0

################### Main Function ######################################            

def main():
    filename = 'pima-indians.data.csv'
    splitRatio = 0.79
    dataset = loadcsv(filename)
    trainingSet, testSet = splitDataset(dataset,splitRatio)
    print (len(trainingSet), len(testSet)) # 606 , 162
    summaries = summarizeByClass(trainingSet)
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet,predictions)
    print (accuracy) # 0.617283950617
main()
               
	
