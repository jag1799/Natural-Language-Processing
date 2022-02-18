from turtle import back
import numpy as np

class MarkovModel:

    def __init__(self):

        # Form of matrix is:
        #         s1(0)  |   s2(1)
        # s1(0)     0.6  |     0.3
        # s2(1)     0.7  |     0.7
        self.states = np.array([[0.6, 0.7], 
                                [0.3, 0.7]])

        # S is hidden, but needed to initialize the HMM
        # Form of matrix is:
        #       s1(0)   |  s2(1)
        # s(0)    0.5   |    0.5
        self.startProbabilities = np.array([0.5, 0.5])

        # Structure of emissions is as follows:
        #       s1(0)   |    s2(1)
        # A(0)    0.3   |      0.1
        # C(1)    0.2   |      0.4
        # G(2)    0.3   |      0.1
        # T(3)    0.2   |      0.4
        self.emissions = np.array([
                                   [0.3, 0.1],
                                   [0.2, 0.4],
                                   [0.3, 0.1],
                                   [0.2, 0.4],
                                ])



def Viterbi(model, observations):

    # Length of observations
    T = len(observations)
    
    # Number of states
    N = len(model.states)
    
    viterbiMatrix = np.zeros(shape=(N, T))

    translation = Translate(observations)
    backpointer = np.zeros(shape=(N, T))
    
    # Initialization step
    for s in range(N):
        # Initialize the starting hidden states with appropriate probabilities for future states
        viterbiMatrix[s][0] = model.startProbabilities[s] * model.emissions[translation[0]][s]
        backpointer[s][0] = 0
    
    # For the remaining number of observations
    for t in range(1, T):
        # For each of the hidden states, N
        for s in range(N):
            transition_probs = list()
            for sprime in range(N):
                transition_probs.append(viterbiMatrix[sprime][t-1] * model.states[sprime][s] * model.emissions[translation[t]][s])
            
            viterbiMatrix[s][t] = max(transition_probs)
            
            
            # backpointer refers to the previous node that generated the maximum probability.  
            # Example: At observation t = 1, the bottom row in column 1 (not 0!) contains a 1 to represent that the greatest
            # probability transition to hidden state s2 in observation 1 came from hidden state s2 in observation 0.
            
            # In the top row of observation 1, it contains a 0 because the greatest probability transition ocurred from
            # hidden state 0 in observation 0(or hidden state s1).
            backpointer[s][t] = np.argmax(transition_probs)

def Translate(observations):

    translation = list()

    for i in observations:
        if i == 'A':
            translation.append(0)
        elif i == 'C':
            translation.append(1)
        elif i == 'G':
            translation.append(2)
        elif i == 'T':
            translation.append(3)
        else:
            print("Unrecognized character. Exiting program...")
            exit()
    
    return translation

model = MarkovModel()
observations = list(['C', 'G', 'T', 'C', 'A'])

Viterbi(model, observations)