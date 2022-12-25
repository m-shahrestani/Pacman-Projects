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
        successor_game_state = currentGameState.generatePacmanSuccessor(action)
        new_pos = successor_game_state.getPacmanPosition()
        new_food = successor_game_state.getFood()
        new_ghost_states = successor_game_state.getGhostStates()
        new_scared_times = [ghostState.scaredTimer for ghostState in new_ghost_states]
        scared_time = min(new_scared_times)

        current_food_list = currentGameState.getFood().asList()
        current_cap = currentGameState.getCapsules()

        new_food_list = new_food.asList()
        new_capsules = successor_game_state.getCapsules()

        closest_food_dist = 9999999
        farthest_food_dist = -9999999
        closest_cap_dist = 9999999
        closest_ghost = 9999999
        evaluation = 0
        found_closest_food = False
        found_farthest_food = False
        found_closest_cap = False

        for food in new_food_list:
            dist = manhattanDistance(new_pos, food)
            if dist < closest_food_dist and dist != 0:
                closest_food_dist = dist
                closest_food_position = food
                found_closest_food = True

        if found_closest_food:
            evaluation += 1000.0 / closest_food_dist

        if found_closest_food:
            for food in new_food_list:
                dist = manhattanDistance(food, closest_food_position)
                if dist > farthest_food_dist and dist != 0:
                    farthest_food_dist = dist
                    found_farthest_food = True

        if found_farthest_food:
            evaluation += 1000.0 / farthest_food_dist

        for capsule in new_capsules:
            dist = manhattanDistance(capsule, new_pos)
            if dist < closest_cap_dist and dist != 0:
                closest_cap_dist = dist
                found_closest_cap = True

        if found_closest_cap:
            evaluation += 1000.0 / closest_cap_dist

        for ghost in new_ghost_states:
            dist = manhattanDistance(ghost.getPosition(), new_pos)
            if dist < closest_ghost:
                closest_ghost = dist

        evaluation += closest_ghost
        if len(new_food_list) < len(current_food_list):
            evaluation += 10000.0

        if len(new_capsules) < len(current_cap):
            evaluation += 10000.0

        if (len(new_capsules) < len(current_cap)) and scared_time < 2:
            evaluation += 15000.0

        evaluation += 10000.0
        if scared_time > closest_ghost:
            evaluation += 10000.0

        if closest_ghost < 2:
            if scared_time < 2:
                evaluation -= 100000.0

        return evaluation

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
        legalMoves = gameState.getLegalActions(0)
        futureStates = [gameState.generateSuccessor(0, move) for move in legalMoves]

        scores = [self.minimizer(0, state, 1) for state in futureStates]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        return legalMoves[chosenIndex]

    def maximizer(self, currentDepth, gameState):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        return max([self.minimizer(currentDepth, state, 1) for state in
                    [gameState.generateSuccessor(0, move) for move in gameState.getLegalActions(0)]])

    def minimizer(self, currentDepth, gameState, ghostIndex):
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return self.evaluationFunction(gameState)

        elif ghostIndex + 1 >= gameState.getNumAgents():
            return min([self.maximizer(currentDepth + 1, state) for state in
                        [gameState.generateSuccessor(ghostIndex, move) for move in
                         gameState.getLegalActions(ghostIndex)]])

        return min([self.minimizer(currentDepth, state, ghostIndex + 1) for state in
                    [gameState.generateSuccessor(ghostIndex, move) for move in
                     gameState.getLegalActions(ghostIndex)]])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        val = -10000.0
        alpha = -10000.0
        beta = 10000.0
        actionSeq = []
        moves = gameState.getLegalActions(0)
        for move in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, move)
            t = self.minimaxPrune(1, range(gameState.getNumAgents()), state, self.depth, self.evaluationFunction,
                                  alpha, beta)
            if t > val:
                val = t
                actionSeq = move
            if val > beta:
                return actionSeq
            alpha = max(alpha, val)
        return actionSeq

    def minimaxPrune(self, agent, agents, state, depth, eval_function, alpha, beta):
        if depth <= 0 or state.isWin() or state.isLose():
            return eval_function(state)

        val = -9999999.0 if agent == 0 else 9999999.0

        for move in state.getLegalActions(agent):
            successor = state.generateSuccessor(agent, move)
            if agent == agents[-1]:
                val = min(val, self.minimaxPrune(agents[0], agents, successor, depth - 1, eval_function, alpha, beta))
                beta = min(beta, val)
                if val < alpha:
                    return val
            elif agent == 0:
                val = max(val,
                          self.minimaxPrune(agents[agent + 1], agents, successor, depth, eval_function, alpha, beta))
                alpha = max(alpha, val)
                if val > beta:
                    return val
            else:
                val = min(val,
                          self.minimaxPrune(agents[agent + 1], agents, successor, depth, eval_function, alpha, beta))
                beta = min(beta, val)
                if val < alpha:
                    return val
        return val


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
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()
    
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
