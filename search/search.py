# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    def in_frontier(state):
        """
            Helper function to check whether state is already added in frontier
        Args:
           state (Tuple): current position
        Returns: Boolean
        """
        for f_state, path in frontier.list:
            # loop through states in frontier and check if state is already added
            if f_state == state:
                return True
        return False

    # keep track of visited nodes
    visited = set()
    # LIFO stack frontier
    frontier = util.Stack()

    # add starting state to frontier
    # state is a tuple of (current_position, list_of_actions)
    frontier.push((problem.getStartState(), []))

    # while loop for exploring
    while not frontier.isEmpty():
        # get first node in frontier
        curr_pos, curr_actions = frontier.pop()

        # add to visited to check for cycle
        visited.add(curr_pos)

        # check if goal is reach
        if problem.isGoalState(curr_pos):
            # if goal is reached, return list of actions taken
            return curr_actions

        # loop through the neighbor states and add states that
        # have NOT visited and NOT in frontier to frontier
        for successor, action, step_cost in problem.getSuccessors(curr_pos):
            if successor in visited or in_frontier(successor):
                continue
            frontier.push((successor, curr_actions + [action]))
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    def in_frontier(state):
        """
            Helper function to check whether state is already added in frontier
        Args:
           state (Tuple): current position
        Returns: Boolean
        """
        for f_state, path in frontier.list:
            # loop through states in frontier and check if state is already added
            if f_state == state:
                return True
        return False

    # keep track of visited set to handle cycle graphs
    visited = set()
    # FIFO structure as frontier for BFS
    frontier = util.Queue()

    # state is a tuple of (current_position, list_of_actions)
    frontier.push((problem.getStartState(), []))

    # while loop for exploring
    while not frontier.isEmpty():
        # get shallowest node from frontier
        curr_state, curr_actions = frontier.pop()
        # add to visited list to avoid cycles
        visited.add(curr_state)

        # check if goal is reached
        if problem.isGoalState(curr_state):
            # if yes, return actions taken to reach goal
            return curr_actions

        # loop through the neighbor states and add states that
        # have NOT visited and NOT in frontier to frontier
        for successor, action, cost in problem.getSuccessors(curr_state):
            if successor in visited or in_frontier(successor):
                continue
            frontier.push((successor, curr_actions + [action]))

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    """
        AIMA3e version of UCS
    """
    class Node:
        """
            Node structure to keep track of state, actions taken, costs
            this structure is needed to implement comparison between 2 nodes \
                                            (due to implementation of upddate in PriorityQueue)
        """
        def __init__(self, state, parent=None, action=None, cost=0):
            self.action = action
            self.parent = parent
            self.state = state
            self.cost = cost

        def __eq__(self, __o: object) -> bool:
            return isinstance(__o, Node) and self.state == __o.state

        def traceback_path(self):
            """
            Function to traceback path to reach current goal
            :return: list of path
            """
            curr_node = self
            path = []
            while curr_node.parent:
                path.append(curr_node.action)
                curr_node = curr_node.parent

            path.reverse()
            return path

    # set to keep track of visited positions
    visited = set()
    # priority queue ordered states by path cost (prioritize lowest cost)
    frontier = util.PriorityQueue()

    # add start start state with cost 0
    frontier.push(Node(problem.getStartState()), 0)

    # while loop for exploring
    while not frontier.isEmpty():
        # get node with lowest cost from frontier
        curr_node = frontier.pop()
        # add current state to visited
        visited.add(curr_node.state)

        # return action taken if goal is reached
        if problem.isGoalState(curr_node.state):
            return curr_node.traceback_path()

        # loop through successors and add states that have not in visited to frontier
        for successor, action, step_cost in problem.getSuccessors(curr_node.state):
            if successor in visited:
                continue

            # use pre-implemented update function to add new node
            # if a state is already in frontier, it would keep the one with lower cost
            frontier.update(Node(successor, action=action,parent=curr_node, cost=curr_node.cost + step_cost),
                            curr_node.cost + step_cost)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    """
        Similar to UCS but with different cost 
    """
    class Node:
        """
            Node structure to keep track of state, actions taken, costs
            this structure is needed to implement comparison between 2 nodes \
                                            (due to implementation of upddate in PriorityQueue)
        """

        def __init__(self, state, parent=None, action=None, cost=0):
            self.action = action
            self.parent = parent
            self.state = state
            self.cost = cost

        def __eq__(self, __o: object) -> bool:
            return isinstance(__o, Node) and self.state == __o.state

        def traceback_path(self):
            """
            Function to traceback path to reach current goal
            :return: list of path
            """
            curr_node = self
            path = []
            while curr_node.parent:
                path.append(curr_node.action)
                curr_node = curr_node.parent

            path.reverse()
            return path

    # keep track of visite states
    visited = set()
    # priority queue ordered states by path cost (prioritize lowest cost)
    frontier = util.PriorityQueue()

    # add start state with cost is the heuristics at start state
    frontier.push(Node(problem.getStartState()),
                  heuristic(problem.getStartState(), problem))

    # while loop for exploring
    while not frontier.isEmpty():
        # get node with least cost from priority queue
        curr_node = frontier.pop()
        # add state to visited state list
        visited.add(curr_node.state)

        # if goal is reached, return actions taken to reach goal
        if problem.isGoalState(curr_node.state):
            return curr_node.traceback_path()

        # loop through successors and add states that have not visited
        for successor, action, step_cost in problem.getSuccessors(curr_node.state):
            if successor in visited:
                continue

            # add new node with cost = heuristics + step cost
            frontier.update(Node(successor, action=action, parent=curr_node, cost=curr_node.cost + step_cost),
                            curr_node.cost + heuristic(successor, problem) + step_cost)
    util.raiseNotDefined()


#####################################################
# EXTENSIONS TO BASE PROJECT
#####################################################

# Extension Q1e
def iterativeDeepeningSearch(problem):
    """Search the deepest node in an iterative manner."""
    "*** YOUR CODE HERE ***"


    def in_frontier(state, frontier: util.Stack):
        """
            Helper function to check whether state is already added in frontier
        Args:
           state (Tuple): current position
        Returns: Boolean
        """
        for f_state, path, depth in frontier.list:
            # loop through states in frontier and check if state is already added
            if f_state == state:
                return True
        return False

    def depth_limited_search(depth):
        """
        :param depth: maximum depth to explore
        :return: Tuple of (status: Int, actions to be taken: List)
        status = -1 when it is a failure (frontier is empty)
        status = 0 when it is a cutoff
        status = 1 when a goal is found
        """
        # keep track of visited nodes
        visited = set()
        # LIFO stack frontier
        frontier = util.Stack()

        # add starting state to frontier
        # state is a tuple of (current_position, list_of_actions, current_depth)
        frontier.push((problem.getStartState(), [], 0))

        status = -1
        # while loop for exploring
        while not frontier.isEmpty():
            # get first node in frontier
            curr_pos, curr_actions, curr_depth = frontier.pop()

            # add to visited to check for cycle
            visited.add(curr_pos)

            # check if goal is reach
            if problem.isGoalState(curr_pos):
                # if goal is reached, return list of actions taken
                return (1, curr_actions)
            elif curr_depth == depth:
                # if it is a cutoff, update status and move on to next frontier
                status = 0
                continue

            # loop through the neighbor states and add states that
            # have NOT visited and NOT in frontier to frontier
            for successor, action, step_cost in problem.getSuccessors(curr_pos):
                if successor in visited or in_frontier(successor, frontier):
                    continue
                frontier.push((successor, curr_actions + [action], curr_depth+1))

        return (status, [])

    # logic for Iterative Depth First Search
    depth = 0
    while True:
        status, actions = depth_limited_search(depth)
        if status == 1:
            return actions
        elif status == -1:
            print("No solution found")
            return []

        depth+=1

    util.raiseNotDefined()


#####################################################
# Abbreviations
#####################################################
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
ids = iterativeDeepeningSearch
