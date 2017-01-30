# FNN
Feed forward neural net to classify text information related to photometry observations of SDSS 

neuralNetwork.py: FNN script.

decisionTree.py: Script  with a sample decision tree to compare with the FNN.

filesManager.py: File manager to import and read the fits files.

config.cfg: configuration file.

# Config
Recomended configuration values in the first executions:

    [Config]
    trainingSize = 1000
    validationSize = 100
    batchSize = 25
    testDataSize = 100
    nHiddenLayers = 5
    num_epochs = 10
    fits_file = [fits file path]
    metadata_file = metadata.dat
    getFromMetadataFile = False
    colsToDeleteFile = colsToDelete.dat

With getFromMetadataFile = False the metadata files will be generated, then is possible to set a true and reuse these metadata files if the datasource is the same.


## To execute

DT: python decisionTree.py 

FNN: python neuronalNetwork.py
 
