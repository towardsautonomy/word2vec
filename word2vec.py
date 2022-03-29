#!/usr/bin/env python

import argparse
import numpy as np
import random

from utils.utils import normalizeRows, softmax


def sigmoid(x):
    """
    Compute the sigmoid function for the input
    Arguments:
    x -- A scalar or numpy array.
    Return:
    s -- sigmoid(x)
    """
    s = 1. / (1. + np.exp(-x))

    return s


def naiveSoftmaxLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset
):
    """ Naive Softmax loss & gradient function for word2vec models

    This function implements the naive softmax loss and gradients between a center word's 
    embedding and an outside word's embedding. This will be the building block
    for our word2vec models. 
    Note that a numpy ndarray with a shape of (x, ) is a one-dimensional array, which
    we can effectively treat as a vector with length x.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """
    ### We will use a custom softmax function (imported earlier in this file)
    ### which is numerically stable implementation helps avoid issues pertaining
    ### to integer overflow.
    naive_softmax = softmax(np.dot(outsideVectors, centerWordVec)) 
    loss = -np.log(naive_softmax[outsideWordIdx])
    gradCenterVec = -outsideVectors[outsideWordIdx] + np.dot(naive_softmax, outsideVectors)

    # compute gradient with respect to outside word vectors
    gradOutsideVecs = np.dot(naive_softmax.reshape(-1, 1), centerWordVec.reshape(1, -1)) # when w != o
    gradOutsideVecs[outsideWordIdx] = np.dot((naive_softmax[outsideWordIdx].reshape(-1, 1) - 1.0), centerWordVec.reshape(1, -1))
    # gradOutsideVecs[outsideWordIdx] -= centerWordVec

    return loss, gradCenterVec, gradOutsideVecs


def getNegativeSamples(outsideWordIdx, dataset, K):
    """ Samples K indexes which are not the outsideWordIdx """

    negSampleWordIndices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == outsideWordIdx:
            newidx = dataset.sampleTokenIdx()
        negSampleWordIndices[k] = newidx
    return negSampleWordIndices


def negSamplingLossAndGradient(
    centerWordVec,
    outsideWordIdx,
    outsideVectors,
    dataset,
    K=10
):
    """ Negative sampling loss function for word2vec models

    This function implements the negative sampling loss and gradients for a centerWordVec
    and a outsideWordIdx word vector as a building block for word2vec
    models. K is the number of negative samples to take.

    Note: The same word may be negatively sampled multiple times. For
    example if an outside word is sampled twice, you shall have to
    double count the gradient with respect to this word. Thrice if
    it was sampled three times, and so forth.

    Arguments:
    centerWordVec -- numpy ndarray, center word's embedding
                    in shape (word vector length, )
                    (v_c)
    outsideWordIdx -- integer, the index of the outside word
                    (o of u_o)
    outsideVectors -- outside vectors is
                    in shape (num words in vocab, word vector length) 
                    for all words in vocab (tranpose of U)
    dataset -- needed for negative sampling, unused here.

    Return:
    loss -- naive softmax loss
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (word vector length, )
                     (dJ / dv_c)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    # Negative sampling of words
    negSampleWordIndices = getNegativeSamples(outsideWordIdx, dataset, K)
    indices = [outsideWordIdx] + negSampleWordIndices

    # compute the logits
    scores = np.dot(outsideVectors[indices], centerWordVec)
    # convert to probabilities
    probOutsideWords = sigmoid(scores[0])
    probNegativeWords = sigmoid(-scores[1:])
    # compute the loss
    loss = -np.log(probOutsideWords) - np.sum(np.log(probNegativeWords))
    # compute the gradients with respect to the center word vector
    gradCenterVec = np.dot((probOutsideWords - 1.), outsideVectors[outsideWordIdx]) + \
                    np.sum((1.0 - probNegativeWords)[:, np.newaxis] * outsideVectors[negSampleWordIndices], axis=0)
    # compute the gradients with respect to the outside word vectors
    gradOutsideVecs = np.zeros_like(outsideVectors)
    gradOutsideVecs[outsideWordIdx] = (probOutsideWords - 1.) * centerWordVec
    for i in range(K):
        gradOutsideVecs[negSampleWordIndices[i]] += (1. - probNegativeWords[i]) * centerWordVec

    # return the loss and gradients
    return loss, gradCenterVec, gradOutsideVecs

def skipgram(currentCenterWord, windowSize, outsideWords, word2Ind,
             centerWordVectors, outsideVectors, dataset,
             word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    """ Skip-gram model in word2vec

    This function implements the skip-gram model.

    Arguments:
    currentCenterWord -- a string of the current center word
    windowSize -- integer, context window size
    outsideWords -- list of no more than 2*windowSize strings, the outside words
    word2Ind -- a dictionary that maps words to their indices in
              the word vector list
    centerWordVectors -- center word vectors (as rows) is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (V)
    outsideVectors -- outside vectors is in shape 
                        (num words in vocab, word vector length) 
                        for all words in vocab (transpose of U)
    word2vecLossAndGradient -- the loss and gradient function for
                               a prediction vector given the outsideWordIdx
                               word vectors, could be one of the two
                               loss functions you implemented above.

    Return:
    loss -- the loss function value for the skip-gram model (J)
    gradCenterVec -- the gradient with respect to the center word vector
                     in shape (num words in vocab, word vector length)
                     (dJ / dv_c)
    gradOutsideVecs -- the gradient with respect to all the outside word vectors
                    in shape (num words in vocab, word vector length) 
                    (dJ / dU)
    """

    loss = 0.0
    gradCenterVecs = np.zeros(centerWordVectors.shape)
    gradOutsideVectors = np.zeros(outsideVectors.shape)

    for ow in outsideWords:
        outsideWordIdx = word2Ind[ow]
        centerWordIdx = word2Ind[currentCenterWord]
        loss_j, gradCenterVec_j, gradOutsideVecs_j = \
            word2vecLossAndGradient(centerWordVectors[centerWordIdx], outsideWordIdx, outsideVectors, dataset)
        loss += loss_j
        gradCenterVecs[centerWordIdx] += gradCenterVec_j
        gradOutsideVectors += gradOutsideVecs_j
    
    return loss, gradCenterVecs, gradOutsideVectors

def word2vec_sgd_wrapper(word2vecModel, word2Ind, wordVectors, dataset,
                         windowSize,
                         word2vecLossAndGradient=naiveSoftmaxLossAndGradient):
    batchsize = 50
    loss = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    centerWordVectors = wordVectors[:int(N/2),:]
    outsideVectors = wordVectors[int(N/2):,:]
    for i in range(batchsize):
        windowSize1 = random.randint(1, windowSize)
        centerWord, context = dataset.getRandomContext(windowSize1)

        c, gin, gout = word2vecModel(
            centerWord, windowSize1, context, word2Ind, centerWordVectors,
            outsideVectors, dataset, word2vecLossAndGradient
        )
        loss += c / batchsize
        grad[:int(N/2), :] += gin / batchsize
        grad[int(N/2):, :] += gout / batchsize

    return loss, grad