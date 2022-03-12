# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):

        successorGameState = currentGameState.generatePacmanSuccessor(self.index, action)
        newPos = successorGameState.getPacmanPosition(self.index)
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        
        if len(newFood.asList()):
            fooddist = util.manhattanDistance(newPos, newFood.asList()[0])
        else:
            fooddist = 0

        return successorGameState.getScore()[self.index] - fooddist
    

def scoreEvaluationFunction(currentGameState, index):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()[index]

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, index = 0, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = index # Pacman is always agent index 0
        self.evaluationFunction = lambda state:util.lookup(evalFn, globals())(state, self.index)
        self.depth = int(depth)



class MultiPacmanAgent(MultiAgentSearchAgent):
    """
    You implementation here
    """
    def myEvaluationFunction(self, currentGameState):
        # Useful information you can extract from a GameState (pacman.py)
        
        newPos = currentGameState.getPacmanPosition(self.index)
        newFood = currentGameState.getFood()
        newGhostStates = currentGameState.getGhostStates()
        #Calculate remaining number of food&Capsules:
        newFoodCount = currentGameState.getNumFood()
        newCapsuleCount = len(currentGameState.getCapsules())
        #Calculate the distance of the furthest food from pacman current location:
        newFoodList = newFood.asList()
        fooddist = -1
        if len(newFoodList)==0:
            fooddist=1
        else:
            for food in newFoodList:
                distance = util.manhattanDistance(newPos, food)
                if distance <= fooddist:
                    fooddist = distance
        #Calculate the total distance of ghosts from pacman current location:
        newGhostPosList = currentGameState.getGhostPositions()
        #distanceOfGhosts = -1
        dangerousSignal = 0
        if len(newGhostPosList)==0:
            dangerousSignal = 0
        else:
            for ghostPos in newGhostPosList:
                if(util.manhattanDistance(newPos, ghostPos)<=3):
                    dangerousSignal += 1
                else:
                    dangerousSignal += 0
            #distanceOfGhosts = distance
        #If ghost is nearby (within distance of 1), also take such situation into consideration, the closer the smaller likelihood to choose this move
        dangerousIndex = -1
        if len(newGhostPosList)==0:
            dangerousIndex = 0
        else:
            for ghostPos in newGhostPosList:
                if util.manhattanDistance(newPos, ghostPos) == 2:
                    dangerousIndex += 2.0
                elif util.manhattanDistance(newPos, ghostPos) == 1:
                    dangerousIndex += 3.0
                elif util.manhattanDistance(newPos, ghostPos) == 0:
                    dangerousIndex += 4.0
                else:
                    dangerousIndex = 0

        if(dangerousSignal<=0):
            return currentGameState.getScore()[self.index] - newFoodCount/80.0 - newCapsuleCount + 1/float(fooddist)
        
        else:
            return currentGameState.getScore()[self.index] - newFoodCount/100.0 - newCapsuleCount + 1/float(fooddist) - dangerousIndex



    def minimax(self, gameState,idx, depth):     
        if depth >= self.depth  or gameState.isWin() or gameState.isLose():
            return self.myEvaluationFunction(gameState)
        if idx == 0:  # pacman
            maxval = float('-inf')  
            actions = gameState.getLegalActions(idx) 
            for action in actions:
                next_state = gameState.generateSuccessor(idx, action)  
                tmpval = self.minimax(next_state,1, depth) #+ 1)  
                if maxval < tmpval:
                    maxval=tmpval
            return maxval

        else:  # ghost
            tmpvals = []
            minval = float('inf') 
            actions = gameState.getLegalActions(idx)
            for action in actions:
                next_state = gameState.generateSuccessor(idx, action)
                if idx == gameState.getNumAgents() - 1: 
                    tmpvals.append(self.minimax(next_state, 0,depth) )
                else:  
                    tmpvals.append(self.minimax(next_state,idx+1, depth+1)) 
                tmpval = min(tmpvals)
                if minval > tmpval:
                    minval=tmpval
            return minval


    def getAction(self, gameState):
        
        "*** YOUR CODE HERE ***"
        ret_action = None
        maxval = float('-inf')
        
        for action in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, action)
            tmpval = self.minimax(next_state,1, 0)
            
            if  maxval < tmpval:
                maxval = tmpval
                ret_action = action
        return ret_action  
        
class RandomAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        legalMoves = gameState.getLegalActions(self.index)
        return random.choice(legalMoves)




