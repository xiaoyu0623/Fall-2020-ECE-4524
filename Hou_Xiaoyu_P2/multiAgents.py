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
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # The pacman will eat the nearby food first
        food_list = newFood.asList()
        min_food_distance = float("inf")
        for i in food_list:
            distance = manhattanDistance(newPos,i)
            min_food_distance = min(min_food_distance,distance)

        # when the ghost is too close, the pacman will run away
        ghost_list = newGhostStates
        for i in ghost_list:
            distance = manhattanDistance(newPos,i.getPosition())
            if(distance < 3):
                return -float("inf")

        score = successorGameState.getScore() + 1.0/min_food_distance

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(gameState, depth, agent):
            # Check goal
            if (depth == self.depth) or (gameState.getLegalActions(agent) == 0) or (gameState.isWin()) or (gameState.isLose()):
                return (self.evaluationFunction(gameState), None)

            value = float("-inf")
            if (agent == 0):
                for action in gameState.getLegalActions(agent):
                    (value_inside, action_inside) = minimax(gameState.generateSuccessor(agent, action), depth, ((agent+1) % gameState.getNumAgents()))
                    if(value_inside > value):
                        value = value_inside
                        max_action = action
            if value != (float("-inf")):
                return (value, max_action)
            
            value = float("inf")
            if agent != 0:
                for action in gameState.getLegalActions(agent):
                    if((agent + 1) % gameState.getNumAgents()) != 0:
                        (value_inside, action_inside) = minimax(gameState.generateSuccessor(agent, action), depth, (agent+1) % gameState.getNumAgents())
                    else:
                        (value_inside, action_inside) = minimax(gameState.generateSuccessor(agent, action), (depth+1), (agent+1) % gameState.getNumAgents())
                    if(value_inside < value):
                        value = value_inside
                        min_action = action
            if value != (float("inf")):
                return(value,min_action)
    
        #util.raiseNotDefined()
        return minimax(gameState, 0, 0)[1]
    
        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphabeta(gameState, depth, agent, alpha, beta):
            if (depth == self.depth) or (gameState.getLegalActions(agent) == 0) or (gameState.isWin()) or (gameState.isLose()):
                return (self.evaluationFunction(gameState), None)

            value = float("-inf")
            if (agent == 0):
                for action in gameState.getLegalActions(agent):
                    (value_inside, action_inside) = alphabeta(gameState.generateSuccessor(agent, action), depth, ((agent+1) % gameState.getNumAgents()), alpha, beta)
                    if(value_inside > value):
                        value = value_inside
                        max_action = action

                    if value > beta:
                        return (value, max_action)
                    alpha = max(alpha,value)

            if value != (float("-inf")):
                return (value, max_action)
            
            value = float("inf")
            if agent != 0:
                for action in gameState.getLegalActions(agent):
                    if((agent + 1) % gameState.getNumAgents()) != 0:
                        (value_inside, action_inside) = alphabeta(gameState.generateSuccessor(agent, action), depth, (agent+1) % gameState.getNumAgents(), alpha, beta)
                    else:
                        (value_inside, action_inside) = alphabeta(gameState.generateSuccessor(agent, action), (depth+1), (agent+1) % gameState.getNumAgents(), alpha, beta)

                    if(value_inside < value):
                        value = value_inside
                        min_action = action

                    if value < alpha:
                        return (value,min_action)
                    beta = min(beta, value)

            if value != (float("inf")):
                return(value,min_action)

        #util.raiseNotDefined()
        return alphabeta(gameState, 0, 0, float("-inf"), float("inf"))[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax (gameState, depth, agent):
            if (depth == self.depth) or (gameState.getLegalActions(agent) == 0) or (gameState.isWin()) or (gameState.isLose()):
                return (self.evaluationFunction(gameState), None)

            value = float("-inf")
            if (agent == 0):
                for action in gameState.getLegalActions(agent):
                    (value_inside, action_inside) = expectimax(gameState.generateSuccessor(agent, action), depth, ((agent+1) % gameState.getNumAgents()))
                    if(value_inside > value):
                        value = value_inside
                        max_action = action
            if value != (float("-inf")):
                return (value, max_action)
            
            value = 0.0
            score = 0.0
            if agent != 0:
                for action in gameState.getLegalActions(agent):
                    if((agent + 1) % gameState.getNumAgents()) != 0:
                        (value_inside, action_inside) = expectimax(gameState.generateSuccessor(agent, action), depth, (agent+1) % gameState.getNumAgents())
                    else:
                        (value_inside, action_inside) = expectimax(gameState.generateSuccessor(agent, action), (depth+1), (agent+1) % gameState.getNumAgents())
                    """    
                    if(value_inside < value):
                        value = value_inside
                        min_action = action
                    """
                    value = value + value_inside
                    score = score + 1
                    min_action = action
                    averge = value / score

            if value != (float("inf")):
                return(averge, min_action)

        #util.raiseNotDefined()
        return expectimax(gameState, 0, 0)[1]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    #Get the current score of the successor state
    score = currentGameState.getScore()

    # Set a list of ghost, a list of food and append all manhattanDistance to list food_distances
    ghost_list_better = newGhostStates
    foodList = newFood.asList()
    food_distances = []
    for food in foodList:
        food_distances.append(manhattanDistance(newPos, food))

    # score for food and ghost. Scared ghost will cost more
    ghost_score = 10.0   
    food_score = 10.0
    scared_ghost_score = 50.0     

    for ghost in ghost_list_better:
        distance = manhattanDistance(newPos, ghost.getPosition())
        if distance > 0:
            if ghost.scaredTimer > 0:                            #if the ghost is scared, we pefer this action
                score = score + (scared_ghost_score / distance)
            else:
                score = score - (ghost_score / distance)


    if len(food_distances) != 0:
        score = score + (food_score / min(food_distances))
    
    return score

    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
