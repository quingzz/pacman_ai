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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # calculate distance to ghosts
        ghosts_dist = sum([manhattanDistance(ghostPos, newPos) for ghostPos in successorGameState.getGhostPositions()])

        # food left
        no_food = len(newFood.asList())

        # calculate average distance and min distance to food of new position
        food_dist = [manhattanDistance(foodPos, newPos) for foodPos in newFood.asList()]
        sum_food_dist = sum(food_dist) if no_food > 0 else 0
        min_food_dist = min(food_dist) if no_food > 0 else 0
        average_dist = sum_food_dist / no_food if no_food > 0 else 0

        # calculate average distance and min distance to food of current position
        curr_food_dist = [manhattanDistance(foodPos, currentGameState.getPacmanPosition()) for foodPos in
                          newFood.asList()]
        sum_curr_food_dist = sum(curr_food_dist) if no_food > 0 else 0
        curr_avg_dist = sum_curr_food_dist / no_food if no_food > 0 else 0
        min_curr_dist = min(curr_food_dist) if no_food > 0 else 0

        # calculate the drop in average distance and min distance, the higher the better
        dist_drop = (curr_avg_dist - average_dist) + (min_curr_dist - min_food_dist)

        # penalize stopping
        stop_penalty = -100 if action == 'Stop' else 0
        # penalize wall cells
        wall_penalty = -99999999 if currentGameState.getWalls()[newPos[0]][newPos[1]] else 0

        # evaluation is score of successor state + distance from ghosts + drop in distance to food + penalties
        return successorGameState.getScore() + ghosts_dist + dist_drop + wall_penalty + stop_penalty


def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        #  get number of agents
        no_agents = gameState.getNumAgents()

        def minimax(state: GameState, local_depth: int):
            """
                Function to check current turn (local_depth) and call min_value, max_value accordingly
                args:
                - state: current game state
                - local_depth: keep track total turn of every agent (different from self.depth)

                return tuple of optimal (action, value)
            """

            # terminate when reach win or lose state or predefined depth is reached
            if local_depth == self.depth * no_agents or state.isWin() or state.isLose():
                return (None, self.evaluationFunction(state))

            if local_depth % no_agents == 0:
                # pacman (maximizing agent) optimal move
                return max_value(state, local_depth)
            else:
                # ghost (minimizing agent) optimal move
                return min_value(state, local_depth, local_depth % no_agents)

        def min_value(state: GameState, depth: int, index: int):
            """
                return tuple of optimal (action, value) for minimizing agents with given index
            """

            min_val = float('inf')
            opt_action = None
            # loop through possible actions and update optimal move/ value for minimizing agent
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)
                # get value of successor state by calling minimax
                val = minimax(successor, depth + 1)[1]

                # update optimal action and value
                if val < min_val:
                    min_val = val
                    opt_action = action

            return opt_action, min_val

        def max_value(state: GameState, depth):
            """
            get tuple of optimal (action, value) for maximizing agent
            """

            max_val = -float('inf')
            opt_action = None

            # loop through possible actions and update optimal move/ value for maximizing agent (pacman)
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                # get value of successor state by calling minimax
                val = minimax(successor, depth + 1)[1]

                # update optimal action and value
                if val > max_val:
                    max_val = val
                    opt_action = action

            return opt_action, max_val

        return minimax(gameState, 0)[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        """
            minimax with alpha beta pruning based on pseudo code in problem specification
        """
        # get number of agents
        no_agents = gameState.getNumAgents()

        def ab_minimax(state: GameState, local_depth: int, alpha=-float('inf'), beta=float('inf')):
            """
                Function to check index and call min_value, max_value accordingly
                args:
                - state: current game state
                - local_depth: keep track total turn of every agent (different from self.depth)
                - alpha: current best value for maximizing agent
                - beta: current best value for minimizing agent

                return pair of optimal action, value
            """

            # terminate when reach win or lose state or predefined depth is reached
            if local_depth == self.depth * no_agents or state.isWin() or state.isLose():
                return (None, self.evaluationFunction(state))

            if local_depth % no_agents == 0:
                # pacman (maximizing agent) move
                return max_value(state, local_depth, alpha=alpha, beta=beta)
            else:
                return min_value(state, local_depth, local_depth % no_agents, alpha=alpha, beta=beta)

        def min_value(state: GameState, depth: int, index: int, alpha=-float('inf'), beta=float('inf')):
            """
                return tuple of optimal (action, value) for minimizing agents with given index
            """

            min_val = float('inf')
            opt_action = None
            # loop through possible actions and update optimal move/ value for minimizing agent
            for action in state.getLegalActions(index):
                successor = state.generateSuccessor(index, action)
                val = ab_minimax(successor, depth + 1, alpha=alpha, beta=beta)[1]
                # update optimal move and value
                if val < min_val:
                    min_val = val
                    opt_action = action

                # pruning process
                if min_val < alpha: return (opt_action, min_val)

                # update beta value for pruning
                beta = min(beta, min_val)

            return (opt_action, min_val)

        def max_value(state: GameState, depth, alpha=-float('inf'), beta=float('inf')):
            """
            get tuple of optimal (action, value) for maximizing agent
            """

            max_val = -float('inf')
            opt_action = None

            # loop through possible actions and update optimal move/ value for maximizing agent
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                val = ab_minimax(successor, depth + 1, alpha=alpha, beta=beta)[1]
                # update optimal move and value
                if val > max_val:
                    max_val = val
                    opt_action = action

                # pruning process
                if max_val > beta: return (opt_action, max_val)

                # update beta value for pruning
                alpha = max(alpha, max_val)

            return opt_action, max_val

        return ab_minimax(gameState, 0)[0]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # get number of agents
        no_agents = gameState.getNumAgents()

        def expectimax(state: GameState, local_depth: int):
            """
                Function to check index and call value generator accordingly
                args:
                - state: current game state
                - local_depth: keep track total turn of every agent (different from self.depth)

                return pair of action, value
            """

            # terminate when reach win or lose state or predefined depth is reached
            if local_depth == self.depth * no_agents or state.isWin() or state.isLose():
                return (None, self.evaluationFunction(state))

            if local_depth % no_agents == 0:
                # get optimal move and value for pacman
                return pacman_val(state, local_depth)
            else:
                # get value for ghost
                return ghost_val(state, local_depth, local_depth % no_agents)

        def ghost_val(state: GameState, depth: int, index: int):
            """
                return average value for all actions
            """
            actions = state.getLegalActions(index)

            # get successor states
            successor_list = [state.generateSuccessor(index, action) for action in actions]

            # find average value for legal actions
            sum_values = sum([expectimax(successor, depth + 1)[1] for successor in successor_list])
            average_val = sum_values / len(actions) if len(actions) > 0 else 0

            # return tuple for consistency
            return None, average_val

        def pacman_val(state: GameState, depth):
            """
                get tuple of optimal (action, value) for maximizing agent (pacman)
            """

            max_val = -float('inf')
            opt_action = None

            # choose action that results in max estimated value
            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                val = expectimax(successor, depth + 1)[1]

                # update optimal action and value
                if val > max_val:
                    max_val = val
                    opt_action = action

            return opt_action, max_val

        return expectimax(gameState, 0)[0]

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
     build upon old score (based on how many food left) and additionally consider following
    - reduce score when food remains
    - increase score when closer to food while further from ghost
    - decrease score when not eating pellet
    """
    "*** YOUR CODE HERE ***"
    # get value of old score evalutation
    score = currentGameState.getScore() * 10

    # get some useful components including current position, food left, scared time
    position = currentGameState.getPacmanPosition()
    no_food = currentGameState.getNumFood()
    scared_times = sum([ghostState.scaredTimer for ghostState in currentGameState.getGhostStates()])

    # get list of components
    ghost_list = currentGameState.getGhostPositions()
    food_list = currentGameState.getFood().asList()
    pellet_list = currentGameState.getCapsules()

    # get nearest distance to ghost
    nearest_ghost = min([manhattanDistance(position, ghost_pos) for ghost_pos in ghost_list])
    # get distance to nearest food
    nearest_food = min([manhattanDistance(position, food_pos) for food_pos in food_list]) if len(food_list) > 0 else 1

    # distance to furthest food
    furthest_food = max([manhattanDistance(position, food_pos) for food_pos in food_list]) if len(food_list) > 0 else 1
    # decrease score when not eating pellet
    score -= len(pellet_list) * 10

    # increase score as pacman is closer to nearest food and ghost is far or is scared
    # if ghost is close, increase score if pacman is closer to furthest food
    # numerator is 9 so additional score would always be less than score for eating the food (predefined as 10)
    #                                   so pacman would try to eat food instead of standing near it
    score += 9 / nearest_food if (nearest_ghost > 2 or scared_times > 0) else 9 / furthest_food

    # decrease score while food left
    score -= no_food * 1000

    return score

# Abbreviation
better = betterEvaluationFunction
