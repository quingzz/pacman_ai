# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
         # loop through iterations:
        for i in range(self.iterations):
            # create a temp counter for new vs
            new_vs = util.Counter()

            # go through the states and update new values based old values
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                possible_actions = self.mdp.getPossibleActions(state)
                # compute new q value for this state
                new_qs = [self.computeQValueFromValues(state, action) for action in possible_actions]

                new_vs[state] = max(new_qs)

            # update values
            for state in self.mdp.getStates():
                self.values[state] = new_vs[state]


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return self.values[state]

        # Q value is immediate reward + sum of discounted values for all successor state
        # get all transitions and reward
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)

        # calculate q_value
        q_value = 0
        for succ_state, transition_prob in transitions:
            # T(s,a,s')[Reward + discount*V(s')]
            q_value += transition_prob*(self.mdp.getReward(state, action, succ_state) +
                                        self.discount*self.values[succ_state])

        return q_value
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None

        # get possible actions
        possible_actions = self.mdp.getPossibleActions(state)
        # get q values of each action
        q_values = [self.computeQValueFromValues(state, action) for action in possible_actions]
        # get index of optimal action
        optimal_act = q_values.index(max(q_values))

        return possible_actions[optimal_act]
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
         # loop through iterations:
        for i in range(self.iterations):
            
            # get state being updated in iteration
            update_state_i = i%len(self.mdp.getStates())
            update_state = self.mdp.getStates()[update_state_i]

            # if the state is terminal -> skip update
            if self.mdp.isTerminal(update_state):
                    continue
            
            possible_actions = self.mdp.getPossibleActions(update_state)
            # compute new q value for this state
            new_qs = [self.computeQValueFromValues(update_state, action) for action in possible_actions]

            self.values[update_state] = max(new_qs)
        

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # COMPUTE PREDECESSOR, it is possible that some states may not have predecessors
        predecessors_map = {}
        for state in self.mdp.getStates():
            # skip checking terminal states since it doesn't have successors
            if self.mdp.isTerminal(state):
                    continue
            
            # get all possible actions
            actions = self.mdp.getPossibleActions(state)
            # get transitions 
            transitions = [self.mdp.getTransitionStatesAndProbs(state, action) for action in actions]
            
            # get all transitions
            for transition_list in transitions:
                for successor, prob in transition_list:
                    if prob > 0: # if prob > 0 -> state is the predecessor of current state
                        
                        if successor not in predecessors_map:
                            predecessors_map[successor] = set()
                            predecessors_map[successor].add(state)
                        else:
                            predecessors_map[successor].add(state)
                                

        queue = util.PriorityQueue()
        # map to keep track of current best q value for state for later
        state_q = {}
        # POPULATE QUEUE
        for state in self.mdp.getStates():
            if self.mdp.isTerminal(state):
                    continue
            
            # get possible actions for this state
            actions = self.mdp.getPossibleActions(state)
            # compute qs values
            q_s = [self.computeQValueFromValues(state, action) for action in actions]
            
            # compute diff
            best_q = max(q_s)
            state_q[state] = best_q
            diff = abs(best_q - self.values[state])
            # push state and diff to priority queue
            queue.push(state, -diff)
            
            
        for i in range(self.iterations):
            if queue.isEmpty():
                break
            
            state = queue.pop()
            if self.mdp.isTerminal(state):
                continue
            self.values[state] = state_q[state]
            
            for predecessor in predecessors_map[state]:
                # get possible actions for this predecessor
                actions = self.mdp.getPossibleActions(predecessor)
                # compute qs values
                q_s = [self.computeQValueFromValues(predecessor, action) for action in actions]
                
                best_q = max(q_s)
                diff = abs(best_q - self.values[predecessor])
                
                if diff > self.theta:
                    # save best_q to later update
                    state_q[predecessor] = best_q
                    queue.update(predecessor, -diff)
                

                
                            
                         
            

