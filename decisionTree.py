# -*- coding: utf-8 -*-
"""
Created on October 2016

@author: Albert Ribé
"""
import csv
import matplotlib.pylab as plt
import numpy as np
from astropy.io import fits
from enum import Enum
import sys
import time

from filesManager import *

# Enumerat amb les 4 tipus d'objecte
class SampleClass(Enum):
    Star = 6
    Galaxy = 3
    Unknown = 0
    Sky = 8    
    
# Funció per eliminar els valors duplicats d'una seqüència
def removeDup(seq): 
   checked = []
   for e in seq:
       if e not in checked:
           checked.append(e)
   return checked

# Obté la classe majoritària d'una col·lecció passada per paràmetre
# el resultat conté el nom de la classe i el nombre d'aparicions
def getMajorityClass(items):
    nGalaxies = {"type": SampleClass.Galaxy, "count": len([item for item in items if item["type"] == SampleClass.Galaxy])}
    nStars = {"type": SampleClass.Star, "count": len([item for item in items if item["type"] == SampleClass.Star])}
    nUnknown = {"type": SampleClass.Unknown, "count": len([item for item in items if item["type"] == SampleClass.Unknown])}
    nSky = {"type": SampleClass.Sky, "count": len([item for item in items if item["type"] == SampleClass.Sky])}

    majority = {}
    if nGalaxies["count"] >= nStars["count"]:
        majority = nGalaxies
    else:
        majority = nStars

    if nUnknown["count"] >= majority["count"]:
        majority = nUnknown

    if nSky["count"] >= majority["count"]:
        majority = nSky

    return majority

# Funció de bondat
def goodness(classes,attr_values):
    start = time.time()
    best_cutoff = (0,0,'','') 

    if(len(classes)==0):
        return best_cutoff

    # Es crea una llista de diccionaries amb la 
    # classe i l'attribut  
    allData=[]
    for i in range(len(attr_values)):
        dictValue = {'type': classes[i], 'v': attr_values[i]}
        allData.append(dictValue)

    # ordenar la llista d'atributs
    orderedAttr = sorted(attr_values)
    # eliminar atributs duplicats
    attributes = removeDup(orderedAttr)
    if(len(attributes) == 0):
        printAndSave("Error in the attributes Length")

    # calcular els punts de tall fent la mitjana entre valor i valor consecutiu
    cutoffs=[]
    for i in range(len(attributes)-1):
        cutoffs.append((attributes[i]+attributes[i+1])/2)

    for cutoff in cutoffs:
        # calcular la classe majoritaria dels menors que el punt de tall
        minorValues = [item for item in allData if item["v"]<=cutoff]
        mCMinor = getMajorityClass(minorValues)
        v1 = mCMinor["count"]    
        
        # calcular la classe majoritaria dels majors que el punt de tall
        greaterValues = [item for item in allData if item["v"]>cutoff]
        mCMajor = getMajorityClass(greaterValues)
        v2 = mCMajor["count"]    

        # es calcula la bondat del punt de tall, si es major que la 
        # actualment més elevada s'estableix com a millor bondat
        goodness = float((v1+v2))/len(attr_values)

        if (goodness>best_cutoff[1]):
            best_cutoff= (cutoff,goodness,mCMinor["type"],mCMajor["type"])
    
    return best_cutoff

# Per defecte s'utilitza una profunditat de 10
def generateDT(training_set, max_depth=10, min_accuracy=1):
    start_time = time.time()

    printAndSave("Generating Decision Tree for a training set of " + str(len(training_set)) + " instances...") 
    
    decision_tree = []
    training_subset1 = []
    training_subset2 = []
    
    classes = [instance['type'] for instance in training_set]

    # valida si tots els elements tenen el mateix tipus
    if classes.count(classes[0]) == len(classes): 
        return ("type",classes[0])

    attributes = {attribute:[instance[attribute] for instance in training_set] for attribute in training_set[0] if attribute != 'type'}
    
    printAndSave("Obtaining goodness values... Remaining levels: " + str(max_depth))     

    # Atribut amb la major bondat
    best_attribute = max([(attr,) + goodness(classes,attributes[attr]) for attr in attributes.keys()], key = lambda x: x[2])

    printAndSave("Level processed in {}".format(time.time()-start_time))

    # S'ha arribat al limit de precissió o a la maxima profunditat
    if (best_attribute[2] >= min_accuracy) | (max_depth == 1): 
        return [(best_attribute[0],best_attribute[1],("type",best_attribute[3]),("type",best_attribute[4]))]        
        
    for i in range(len(training_set)):        
        if training_set[i][best_attribute[0]] <= best_attribute[1]:
            training_subset1.append(training_set[i])
        else:
            training_subset2.append(training_set[i])
       
    decision_tree.append((best_attribute[0],best_attribute[1],generateDT(training_subset1,max_depth=max_depth-1,min_accuracy=min_accuracy),generateDT(training_subset2,max_depth=max_depth-1,min_accuracy=min_accuracy)))

    return decision_tree
    
# Classifica un element mitjançant un arbre passat per paràmetre    
def classify(instance, dectree):
    if dectree[0] == 'type':
        return dectree[1]

    if instance[dectree[0][0]] < dectree[0][1]:
        return classify(instance, dectree[0][2])
    else:
        return classify(instance, dectree[0][3])

# Executa els tests 
def runTest(instance_list, dectree):
    hits = 0    
    
    for i in instance_list:
        real_class = i['type']
        pred_class = classify(i,dectree)
        
        if real_class == pred_class:
            hits += 1
    
    printAndSave('-------------------\nAccuracy: ' + str(100*hits/len(instance_list)) + '%\n-------------------')

# Genera una llista etiquetant la columna tipus          
def generateLists(matrix,headers):
    keys = headers

    values=[]
    for row in matrix:
        dictValue = dict(zip(keys,row))
        if(row[0]==6):
            dictValue['type'] = SampleClass.Star
        elif(row[0]==3):
            dictValue['type'] = SampleClass.Galaxy
        elif(row[0]==8):
            dictValue['type'] = SampleClass.Sky
        else:
            dictValue['type'] = SampleClass.Unknown

        values.append(dictValue)
    
    return values


def main():
    trainingSize,_, _,testSize,_,_,getMDFromFile = getConfigData()

    printAndSave("Preparing data ...")
    
    # Segons l'escollit al fitxer de configuració les metadades 
    # es generen o obtenen d'un fitxer
    if (getMDFromFile):
        printAndSave("Getting metadata from file...")
        getMetadata = getMetadataFromFile   
    else:
        printAndSave("Calculating metadata...")
        getMetadata = calculateMetadata

    # s'obtenen les col·leccions d'entrenament i test normalitzades per evitar valors molt grans
    trainingLines, testLines, headers = getDTNormalizedLines(trainingSize, testSize,getMetadata)

    trainingList = generateLists(trainingLines,headers)
    testList = generateLists(testLines,headers)

    # Generar l'arbre
    start_time = time.time()
    printAndSave("Start training ...")
    dectree = generateDT(trainingList)
    printAndSave("Training ended in: {}".format(time.time()-start_time))

    start_time = time.time()
    printAndSave("--------------\n RUNNING TESTS \n--------------")
    runTest(testList,dectree)

    printAndSave("Test ended in: {}".format(time.time()-start_time))

if __name__ == '__main__':
    kwargs = {}
    main(**kwargs)




