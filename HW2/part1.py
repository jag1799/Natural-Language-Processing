# Author: Jakob Germann
# Start Date: 2/15/2022
# Disclaimer: 
#  Structure for the Markov Model was taken from 
# "Speech and Language Processing" by Daniel Jurafsky & James H. Martin
# Copyright 2021. All rights reserved. Draft of December 29, 2021


from turtle import back
import numpy as np

observations = list(['C', 'G', 'T', 'C', 'A'])

class MarkovModel:

    def __init__(self):
        # Initialize states with the transition probabilities already inside.
        # self.states = {'s1': {'s1': 0.6, 's2': 0.7}, 's2': {'s1': 0.3, 's2': 0.7}}

        # Form of matrix is:
        #       s1  |   s2
        # s1   0.6  |  0.3
        # s2   0.7  |  0.7
        self.states = np.array([[0.6, 0.7], 
                                [0.3, 0.7]])
        # S is hidden, but needed to initialize the HMM
        # self.startProbabilities = {'s': {'s1': 0.5, 's2': 0.5}}
        # Form of matrix is:
        #    s1    |  s2
        # s  0.5   |  0.5
        self.startProbabilities = np.array([0.5, 0.5])

        # Structure of emissions is as follows:
        #       A(1)    |    C(2)    |    G(3)    |    T(4)
        #  s1   0.3     |    0.2     |    0.3     |    0.2
        #  s2   0.1     |    0.4     |    0.1     |    0.4
        self.emissions = np.array([[0.3, 0.2, 0.3, 0.2],
                                    [0.1, 0.4, 0.1, 0.4]])


def Viterbi():

    # Initialize starting systems
    global observations
    T = len(observations)
    
    model = MarkovModel()
    N = len(model.states)

    backpointer = np.zeros(shape=(N, T))



Viterbi()
