from pacman import *
from game import Agent
import numpy as np
import time
import psutil

class MCTSNode:
    def __init__(self, ID, parent, state, action):
        self.ID = ID
        self.parent = parent
        self.children = []
        self.value = 0
        self.visits = 0
        self.state = state
        self.action = action

class MCTSagent(Agent):
    def __init__(self):
        self.n = 25
        self.c = 1 / np.sqrt(2)
        self.treeSize = 1

    def bestChild(self, node):
        childInd = 0
        bestVal = 0
        for i in range(len(node.children)):
			# Check that this is correct UCT
            curVal = node.children[i].value + (self.c * np.sqrt(2 * np.log(self.treeSize) / node.children[i].visits))
            if curVal > bestVal:
                childInd = i
                bestVal = curVal
        return node.children[childInd]

    def getAction(self, state):
        # ID, parent, children, value, number of visits, state, action to get to state 
        root = MCTSNode(0, None, state, None)
        startTime = time.time()
        #for _ in range(self.n):
        while time.time() - startTime < 0.25:
            node = self.treePolicy(root)
            delta = self.defaultPolicy(node.state)
            self.backup(node, delta)
        return self.bestChild(root).action

        
    def treePolicy(self, node):

        def fullyExpanded(node):
            return len(node.children) == len(node.state.getLegalPacmanActions())

        def expand(node):
			# Check that the random action hasn't been taken before (it'll be in another child if it has)
            action = np.random.choice(node.state.getLegalPacmanActions())
            newState = node.state.generatePacmanSuccessor(action)
            newChild = MCTSNode(self.treeSize, node, newState, action)
            node.children.append(newChild)
            self.treeSize += 1
            return newChild
        

        while (not node.state.isWin()) and (not node.state.isLose()):
            if not fullyExpanded(node):
                return expand(node)
            else:
                node = self.bestChild(node)
        return node

    def defaultPolicy(self, state):
        while (not state.isWin()) and (not state.isLose()):
            action = np.random.choice(state.getLegalPacmanActions())
            state = state.generatePacmanSuccessor(action)
        return state.getScore()

    def backup(self, node, delta):
        while not node == None:
            node.visits += 1
            node.value += delta
            node = node.parent

if __name__ == '__main__':
    #str_args = ['-g', 'DirectionalGhost', '--frameTime', '0']   
    str_args = ['--frameTime', '0', "-l", "contestClassic"] 
    #str_args = ['-l', 'TinyMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '10']
    #str_args = ['-l', 'TestMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '10']
    args = readCommand(str_args)
    # args['display'] = textDisplay.NullGraphics()  # Disable rendering

    args['pacman'] = MCTSagent()
    out = runGames( **args)
    scores = [o.state.getScore() for o in out]  
    print(scores)