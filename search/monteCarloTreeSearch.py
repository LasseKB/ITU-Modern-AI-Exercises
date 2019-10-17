import copy

from pacman import *
from game import Agent
import numpy as np


class MCTSagent(Agent):
    def __init__(self):
        self.explored = {}  # Dictionary for storing the explored states
        self.n = 10  # Depth of search  # TODO: Play with this once the code runs
        self.c = 1  # Exploration parameter # TODO: Play with this once the code runs
        self.treeSize = 1

    def getAction(self, state):
        """ Main function for the Monte Carlo Tree Search. For as long as there
            are resources, run the main loop. Once the resources runs out, take the
            action that looks the best.
        """
        self.explored = {}

        root = [0, None, [], 0, 0, state, None]  # TODO: How will you encode the nodes and states?

        for _ in range(self.n):  # while resources are left (time, computational power, etc)
            selection = self.traverse(root)
            action = np.random.choice(selection[5].getLegalPacmanActions)
            expansion = 
                simulation_result = self.rollout(leaf)
                self.backpropagate(leaf, simulation_result)

        return self.best_action(root)

    #def all_successors(self, state):
    #    """ Returns all legal successor states."""
    #    next_pos = []
    #    for action in state.getLegalPacmanActions():
    #        next_pos.append(state.generatePacmanSuccessor(action))
    #    return next_pos

    def traverse(self, node):
        """ Returns a list of states to explore. If state is terminal the list
            has length 1.
        """

        def state_is_explored(node):
            """ Determines whether a state has been explored before.
                Returns True if the state has been explored, false otherwise
            """
            """ YOUR CODE HERE!"""
            return len(node[2]) > len(self.all_successors(node[5]))

        def best_UCT(node):
            """ Given a state, return the best action according to the UCT criterion."""
            """ YOUR CODE HERE!"""
            actionInd = 0
            bestVal = 0
            children = node[2]
            for i in range(len(children)):
                curVal = children[i][3] + (self.c * np.sqrt(2 * np.log(self.treeSize) / children[i][4]))
                if curVal > bestVal:
                    actionInd = i
                    bestVal = curVal
            return children[actionInd]

        while state_is_explored(node):
            node = best_UCT(node)

            if node[5].isWin() or node[5].isLose():
                return [copy.deepcopy(node[5])]

        return node

    def rollout(self, state):
        """ Simulate a play through, using random actions.
        """
        while not state.isWin() and not state.isLose():
            """ YOUR CODE HERE! """
            action = np.random.choice(state.getLegalPacmanActions())
            state = state.generatePacmanSuccessor(action)
        return state.getScore()

    def backpropagate(self, state, result):
        """ Backpropagate the scores, and update the value estimates."""
        """ YOUR CODE HERE! """


    def best_action(self, node):
        """ Returns the best action given a state. This will be the action with
            the highest number of visits.
        """
        """ Your Code HERE!"""
        action = None
        return action


if __name__ == '__main__':
    str_args = ['-l', 'TinyMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '10']
    str_args = ['-l', 'TestMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '10']
    args = readCommand(str_args)
    # args['display'] = textDisplay.NullGraphics()  # Disable rendering

    args['pacman'] = MCTSagent()
    out = runGames( **args)

    scores = [o.state.getScore() for o in out]
    print(scores)
