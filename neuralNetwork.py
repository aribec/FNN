# -*- coding: utf-8 -*-
"""
Created on October 2016

@author: Albert Ribé
"""

from __future__ import print_function

import sys
import os
import time
import math
import numpy as np
import theano
import theano.tensor as T
import lasagne

from theano.compile.nanguardmode import NanGuardMode
from sklearn import preprocessing

from filesManager import *

# Funció per crear una FNN, s'especifiquen com arguments:  
# depth: la profunditat (hidden layers), 
# width: l'amplada de cada layer, 
# drop_input: index de drop out a l'entrada
# drop_hidden: index de drop out a les hidden layers 
# nCols: nombre de columnes que l'input
def buid_MLP(input_var=None, depth=2, width=500, drop_input=.2,
                     drop_hidden=.5, nCols = None):

    network_ = lasagne.layers.InputLayer(shape=(None, nCols),
                                     input_var=input_var)

    if drop_input:
        network_ = lasagne.layers.DropoutLayer(network_, p=drop_input)
    
    # creació de les capes ocultes
    for _ in range(depth):
        network_ = lasagne.layers.DenseLayer(
                network_, num_units = width, nonlinearity=lasagne.nonlinearities.rectify)
        if drop_hidden:
            network_ = lasagne.layers.DropoutLayer(network_, p=drop_hidden)

    # capa de sortida
    l_out = lasagne.layers.DenseLayer(
        network_, num_units=10,
        nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

# S'obté un bloc d'elements, a partir del llistat generar del set,
# posteriorment es normalitza per tal de centrar els valors i evitar 
# nombres grans que produíssin errors al executar el MLP 
def iterate_minibatches(list,list_targets, batchsize, metadata, colsToRemove):
    for start_idx in range(0, len(list) - batchsize + 1, batchsize):
        batchLines = list[start_idx: start_idx + batchsize]
        batchTargets = list_targets[start_idx: start_idx + batchsize]
        
        targets = batchTargets
        targets = targets.astype(int)
        tmp = np.array(batchLines.tolist())

        inputs = np.delete(tmp,colsToRemove,1) 
        
        # s'estandaritzen les dades mitjancant les metadades calculades
        inputs = normBatch(inputs, metadata)

        yield inputs, targets

def main():
    # Al inici es recuperen els valor del fitxer de configuració
    trainingSize, validationSize, batchSize, testDataSize, nLayer, num_epochs, getFromFile = getConfigData()
 
    printAndSave("Loading data...",dt = False)

    # Segons l'escollit al fitxer de configuració les metadades 
    # es generen o obtenen d'un fitxer
    if (getFromFile):
        printAndSave("Getting metadata from file...", dt = False)
        getMetadata = getMetadataFromFile   
    else:
        printAndSave("Calculating metadata...", dt = False)
        getMetadata = calculateMetadata

    # S'obtenen les dades tant de les coleccions d'entrada com de les etiquetes per validar
    train, trainTargets, val, valTargets, test, \
    testTargets, metadata, colsToRemove = \
    getTrainingTestLists( traiSize = trainingSize, 
                            valSize = validationSize, 
                            testSize = testDataSize, 
                            getMetadata =  getMetadata)

    # Es preparen les variables de theano per a 
    # l'entrada i les etiquetes que s'utilitzen
    # per validar els reusltats
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    # Es crea la FNN
    network = buid_MLP(input_var = input_var, depth = nLayer, drop_input=.2,drop_hidden=.5, nCols = len(metadata))
    
    # S'obté la predicció a partir de la sorida de la MLP 
    prediction = lasagne.layers.get_output(network)
    # Expressió per la perdua
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    
    # Es creen les expressions d'update per modificar els 
    # parametres en cada pas del training
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # S'obté la predicció a partir de la sorida de la MLP per la validació i testing, 
    # a diferència de l'anterior aquí es desactiven les capes de dropout passant a 
    # través de tota la xarxa amb el mode deterministic a True
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
    test_loss = test_loss.mean()

    # Expressió per la precissió de la classificació es realitza a 
    # partir de la predicció obtinguda al a sortida del MLP
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compilar la funcio executant un pas de training mitjancant un petit 
    # paquet de dades, es retornara la perdua
    # S'activa el mode NanGuardMode per tal d'obtenir un error en cas de 
    # nombres massa grans això val per comprobar la validesa en la 
    # normalització de les dades
    train_fn = theano.function([input_var, target_var], loss, updates=updates, name="TrainingFunc", 
        mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

    # Es recuperarà la pèrdua i precissió, 
    # s'utilitza tant en la validació com en el test
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc], name="ValidationFunc")
    
    # S'inicialitza l'entrenament
    printAndSave("*"*53,dt = False)
    printAndSave("Starting training...", dt = False)
    training_start_time = time.time()

    for epoch in range(num_epochs):
        start_time = time.time()
        # Per cada iteració es fa una execució completa de les dades d'entrenament
        train_err = 0
        train_batches = 0
        for batch in iterate_minibatches(train,trainTargets, batchSize, metadata, colsToRemove):
            inputs, targets = batch
            tmp = train_fn(inputs, targets)
            train_err += tmp
            train_batches += 1
        
        # Validació de la iteració
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(val,valTargets, batchSize, metadata, colsToRemove):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Impressió de resultats de la iteració
        printAndSave("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time),dt = False)
        printAndSave("  training loss:\t\t{:.6f}".format(train_err / train_batches),dt = False)
        printAndSave("  validation loss:\t\t{:.6f}".format(val_err / val_batches),dt = False)
        printAndSave("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100),dt = False)

        # #######################################
        # Activar per calcular l'error i precisió 
        # amb les dades de test en cada epoch
        # #######################################
        # Es realitza el test 
        # start_time = time.time()
        # test_err = 0
        # test_acc = 0
        # test_batches = 0
        # for batch in iterate_minibatches(test,testTargets, batchSize, metadata, colsToRemove):
        #     inputs, targets = batch
        #     err, acc = val_fn(inputs, targets)
        #     test_err += err
        #     test_acc += acc
        #     test_batches += 1
        # # Impressió de resultats del test
        # printAndSave("Final results:",dt=False)
        # printAndSave("  test loss:\t\t\t{:.6f}".format(test_err / test_batches),dt=False)
        # printAndSave("  test accuracy:\t\t{:.2f} %".format(
        #     test_acc / test_batches * 100),dt=False)
        # printAndSave("Tests in {}".format(time.time()-start_time),dt=False)
        
    printAndSave("Training in {}".format(time.time()-training_start_time),dt = False)

    # Es realitza el test
    start_time = time.time()
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(test,testTargets, batchSize, metadata, colsToRemove):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    
    # Impressió de resultats del test
    printAndSave("Final results:",dt=False)
    printAndSave("  test loss:\t\t\t{:.6f}".format(test_err / test_batches),dt=False)
    printAndSave("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100),dt=False)
    printAndSave("Tests in {}".format(time.time()-start_time),dt=False)

if __name__ == '__main__':
    kwargs = {}
    main(**kwargs)