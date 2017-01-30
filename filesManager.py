# -*- coding: utf-8 -*-
"""
Created on October 2016

@author: Albert Ribé
"""

from astropy.io import fits
import numpy as np
import time 
from sklearn import preprocessing

import ConfigParser
import os.path

# variables globals
glob_fitsFile = None
glob_metadataFile = "metadata.dat"
glob_colsToDeleteFile = "rowsToDelete.dat"

# ******** Gestió de FITS *********
# Obrir un fitxer fits
def openFitsFile(file):
    hdulist = fits.open(file, memmap = True)
    return hdulist

# Tancar un fitxer fits
def closeFitsFile(hdulist):
    hdulist.close()

# Obté un llistat d'elements del fitxer passat
def getDatasetList(size = 100, index = 0):
    hdulist = openFitsFile(glob_fitsFile)

    tbddata = hdulist[1].data
    if (size == None):
        result = tbddata
    else:
        result = tbddata[index:size]
    
    closeFitsFile(hdulist)
    return result

# Obté coleccions de dades de diferents mides
def getTrainingTestSets(*arg):
    offset = 0
    orderedColections = []
    for numberofRows in arg:
        tmp = getDatasetList(numberofRows + offset, offset)
        offset += numberofRows
        orderedColections.append(tmp)
    return orderedColections

# S'obtenen i normalitzen les col·leccions d'entrenament i test per al DT
def getDTNormalizedLines(trainingSize, testSize, getMetadata = None):
    lines = getTrainingTestSets(trainingSize, testSize)

    trainingLines = np.array(lines[0].tolist())
    testLines = np.array(lines[1].tolist())
    columnNames = lines[0].columns.names
    metadata, colsToRemove = getMetadata()

    # s'extreu la columna del tipus
    trainingTypes = trainingLines[:,7]
    training = np.delete(trainingLines,colsToRemove,1) 
    training = normBatch(training, metadata)
    # una vegada normalitzat el tipus es situa com a primera columna
    training = np.insert(training,0,trainingTypes,1)

    testTypes = testLines[:,7]
    test = np.delete(testLines,colsToRemove,1) 
    test = normBatch(test, metadata)
    test = np.insert(test,0,testTypes,1)

    # S'eliminen les columnes marcades per eliminar excepte el tipus
    validColumnNames = []
    for i in range(len(columnNames)):
        removeCol = False
        for c in colsToRemove:
            if(i==c):
                removeCol = True
        if(not removeCol):
            validColumnNames.append(columnNames[i])
        if(columnNames[i] == 'type'):
            validColumnNames.insert(0,columnNames[i])

    return training, test, validColumnNames

# Normalitzar les dades d'entrada 
# a partir d'un conjunt de metadades
# el nombre d'input data ha de ser 
# el mateix que les files de metadata 
def normBatch(inputData, metaData):
    for row in inputData:
        for index in range(0,len(row)):
            max = metaData[index][0]
            min = metaData[index][1]
            aux = row[index]
            row[index] -= min
            row[index] /= (max-min)
               
    return inputData

# Genera els conjunts d'entrenament i test per la FNN
def getTrainingTestLists(traiSize = 100, valSize = 10, testSize = 10, getMetadata = None):
    orderedCollections = getTrainingTestSets(traiSize,valSize,testSize)
    trainingList = orderedCollections[0]
    valList = orderedCollections[1]
    testList = orderedCollections[2]

    # extreure la columna type
    trainingTargets = trainingList['type']
    valTargets = valList['type']
    testTargets = testList['type']

    # s'obté les metadades necessaries per normalitzar 
    # del conjunt d'entrenament
    metadata, colsToRemove = getMetadata()

    # eliminar la columna de l'etiqueta dels conjunts inicials
    trainingList.columns.del_col('type')
    valList.columns.del_col('type')
    testList.columns.del_col('type')

    return trainingList, trainingTargets, valList, valTargets, testList, testTargets, metadata, colsToRemove

# Obté un fitxer fits de casjobs mitjançant la tool de casjobs
# requereix disposar de l'aplicació Java de CasJobs
def getFitsFromSDSS(tableName = "MyTable", traiSize = 100, testSize = 10):
    import subprocess
    res = subprocess.call(['java', '-jar', 'casjobs.jar', 'extract', '-table', 'MyTable', '-F', '-type', 'fits', '-d'])
    return res

# ******** Gestió de metadades ********
# Intenta obtenir les metadades dels fitxers corresponents, 
# si no existeixen o són buits reprodueix un error
def getMetadataFromFile():
    if(not os.path.isfile(glob_metadataFile) or os.stat(glob_metadataFile).st_size == 0):
        raise ValueError('The Metadata File not exists or is empty, please recreate it with getFromMetadataFile = False')

    if(not os.path.isfile(glob_colsToDeleteFile) or os.stat(glob_colsToDeleteFile).st_size == 0):
        raise ValueError('The Columns to Delete File not exists or is empty, please recreate it with getFromMetadataFile = False')

    metadata = np.loadtxt(glob_metadataFile, dtype = float, delimiter =',')
    colsToRemove = np.loadtxt(glob_colsToDeleteFile, dtype = float)

    return metadata, colsToRemove 

# Es calculen els minims i màxims de cada columna a 
# partir de la col·lecció global d'elements establerta 
# a la configuració
def calculateMetadata():
    inputList = getDatasetList(None, 0)

    aux = np.array(inputList.tolist())
    max = np.amax(aux, axis = 0)
    min = np.min(aux, axis = 0)

    i = 0
    colsToRemove =[]
    for ma,mi in zip(max,min):
        if ma == mi:
            colsToRemove.append(i)
        i+=1
    # el tipus s'empre es marca per eliminar
    colsToRemove.append(6)

    metaData = np.column_stack((max,min))
    metaData = np.delete(metaData,colsToRemove,0)

    # si existeix elimina el fitxer de 
    # metadades per crear-ne un de nou
    if(os.path.exists(glob_metadataFile)):
        print("Removing metadata file...")
        os.remove(glob_metadataFile)
    if(os.path.exists(glob_colsToDeleteFile)):
        print("Removing cols to delete files...")
        os.remove(glob_colsToDeleteFile)
    
    print("Saving new metadata files...")
    f=open(glob_metadataFile,'ab')
    np.savetxt(f,metaData, delimiter = ',')
    f.close()

    f=open(glob_colsToDeleteFile,'ab')
    np.savetxt(f,colsToRemove,delimiter = ',')
    f.close()

    return metaData, colsToRemove

# ******** Gestió de configuració ********
# S'obre el fitxer de configuració per prendren els valors 
def getConfigData(filename = 'config.cfg'):
    Config = ConfigParser.ConfigParser()
    Config.read(filename)

    configSection =Config.sections()[0]
    options = Config.options(configSection)
    trainingSize = int(Config.get(configSection,options[0]))
    validationSize = int(Config.get(configSection,options[1]))
    batchSize = int(Config.get(configSection,options[2]))
    testDataSize = int(Config.get(configSection,options[3]))
    nHiddenLayers = int(Config.get(configSection,options[4]))
    num_epochs = int(Config.get(configSection,options[5]))
    global glob_fitsFile
    glob_fitsFile = str(Config.get(configSection,options[6]))
    global glob_metadataFile
    glob_metadataFile = str(Config.get(configSection,options[7]))
    getFromFile = Config.get(configSection,options[8]) == "True"
    global glob_colsToDeleteFile
    glob_colsToDeleteFile = str(Config.get(configSection,options[9]))
    
    print("*"*53)
    print("The configuration data is:")
    print("*"*53)
    print("Training set Size: {}".format(trainingSize))
    print("Validation set Size: {}".format(validationSize))
    print("Batch Size: {}".format(batchSize))
    print("Test set Size: {}".format(testDataSize))
    print("Number of hidden Layers: {}".format(nHiddenLayers))
    print("Number of epochs: {}".format(num_epochs))
    print("Fits File: {}".format(glob_fitsFile))
    print("Metadata File: {}".format(glob_metadataFile))
    print("Get metadata from File: {}".format(getFromFile))
    print("Columns to delete File: {}".format(glob_colsToDeleteFile))
    print("*"*53)   
    
    return trainingSize, validationSize, batchSize, testDataSize, nHiddenLayers, num_epochs, getFromFile

# Escriu un missatge tant per pantalla 
# com en un fitxer de log
# al parametre DT s'especifica si l'execució bé 
# del arbre o la FNN així es pot generar diferents logs           
def printAndSave(msg, dt=True):
    logFileName = "dt_log.txt"
    if (not dt):
        logFileName = "nn_log.txt"

    print(msg)    
    with open(logFileName, "a") as myfile:
        myfile.write("\n" + msg)
