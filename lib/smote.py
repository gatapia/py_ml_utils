#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
The MIT License (MIT)
Copyright (c) 2012-2013 Karsten Jeschkies <jeskar@web.de>

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all 
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION 
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

'''
Created on 24.11.2012

@author: karsten jeschkies <jeskar@web.de>

This is an implementation of the SMOTE Algorithm. 
See: "SMOTE: synthetic minority over-sampling technique" by
Chawla, N.V et al.
'''
import logging, math
import numpy as np
from random import randrange, choice
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger("main")

def SMOTE(T, N, k, h = 1.0):
    """
    Returns (N/100) * n_minority_samples synthetic minority samples.

    Parameters
    ----------
    T : array-like, shape = [n_minority_samples, n_features]
        Holds the minority samples
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 

    Returns
    -------
    S : Synthetic samples. array, 
        shape = [(N/100) * n_minority_samples, n_features]. 
    """    
    n_minority_samples, n_features = T.shape    
    N = N/100.
    n_synthetic_samples = N * n_minority_samples
    S = np.zeros(shape=(n_synthetic_samples, n_features))
    
    #Learn nearest neighbours
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(T)
    
    #Calculate synthetic samples    
    for n in xrange(int(math.ceil(N))):
        if n == math.ceil(N):
            sample_size = int(n_minority_samples - (n*100))
        else:
            sample_size = int(n_minority_samples)
        T2 = T[np.random.choice(len(T), sample_size)]
        for i in xrange(T2.shape[0]):
            nn = neigh.kneighbors(T2[i], return_distance=False)
            nn_index = choice(nn[0])
            #NOTE: nn includes T[i], we don't want to select it 
            while nn_index == i:
                nn_index = choice(nn[0])
                
            dif = T2[nn_index] - T2[i]
            gap = np.random.uniform(low = 0.0, high = h)
            S[n + i * N, :] = T2[i,:] + gap * dif[:]
    
    return S

def borderlineSMOTE(X, y, minority_target, N, k):
    """
    Returns synthetic minority samples.

    Parameters
    ----------
    X : array-like, shape = [n__samples, n_features]
        Holds the minority and majority samples
    y : array-like, shape = [n__samples]
        Holds the class targets for samples
    minority_target : value for minority class
    N : percetange of new synthetic samples: 
        n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
    k : int. Number of nearest neighbours. 
    h : high in random.uniform to scale dif of snythetic sample

    Returns
    -------
    safe : Safe minorities
    synthetic : Synthetic sample of minorities in danger zone
    danger : Minorities of danger zone
    """ 
    
    n_samples, _ = X.shape

    #Learn nearest neighbours on complete training set
    neigh = NearestNeighbors(n_neighbors = k)
    neigh.fit(X)
    
    safe_minority_indices = list()
    danger_minority_indices = list()
    
    for i in xrange(n_samples):
        if y[i] != minority_target: continue
        
        nn = neigh.kneighbors(X[i], return_distance=False)

        majority_neighbours = 0
        for n in nn[0]:
            if y[n] != minority_target:
                majority_neighbours += 1
                
        if majority_neighbours == len(nn):
            continue
        elif majority_neighbours < (len(nn)/2):
            logger.debug("Add sample to safe minorities.")
            safe_minority_indices.append(i)
        else:
            #DANGER zone
            danger_minority_indices.append(i)
            
    #SMOTE danger minority samples
    synthetic_samples = SMOTE(X[danger_minority_indices], N, k, h = 0.5)
    
    return (X[safe_minority_indices],
            synthetic_samples, 
            X[danger_minority_indices])