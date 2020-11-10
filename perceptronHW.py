# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:40:34 2020

@author: -
"""

import numpy as np
import pandas as pd
from numpy import genfromtxt

#Global Variables

w=np.zeros(1025)

#############################################################################
def predict(row):
    global w
    activation = w[0]
    for i in range(len(row)-1):
         activation += w[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else -1.0
#############################################################################
def weightUpdate(labelMistake, row):
    global w
    #wrong prediction on a 1 
    index=0;
    if(labelMistake==1):
        while(index!=1023):
            w[index+1]= w[index+1]+row[index]
            index+=1
    #wrong prediction on a -1
    elif(labelMistake==-1):
        while(index!=1023):
            w[index+1]= w[index+1]-row[index]
            index+=1       
            
#############################################################################
def runPerceptron():
    global w
    #Import Data
    data = pd.read_csv('train-a1-449.txt', sep=" ", header=None)
    data = data.iloc[:, :-1]
    labels= data.iloc[:,-1]
    data = data.iloc[:, :-1]
    if(abs(data.min().min())>=abs(data.max().max())):
        data=data.div(abs(data.min().min()))
        print("meme")
    else:
        data=data.div(abs(data.max().max()))
        print("meme")
    #reformat so normalized
    data['labels'] = labels[:].apply(lambda x: 1 if x =='Y' else -1)
    
    #set up weight vector

    errors=1
    while(errors>0):
        errors=0
        for index, row in data.iterrows():
            if(predict(row[:-1])!=row['labels']):
               weightUpdate(row['labels'], row[:-1]) # Book says to update weights like w<- w + xi 
               errors+=1
        print("Number of errors: " + str(errors))
    minimum=pd.Series
    #calculate L2 Margin           
    for index, row in data.iterrows():
        if index==0:
           minimum= row
           print(row)
        else:
            if abs(sum(minimum))<(abs(sum((w[1:]*row[:-1])/w[1:]))):
               minimum = ((w[1:]*row[:-1])/w[1:])
    return abs(sum(minimum))
                
    

print(runPerceptron())