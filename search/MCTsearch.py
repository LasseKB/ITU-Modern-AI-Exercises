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
        self.triedActions = []

class MCTSagent(Agent):
    def __init__(self):
        self.n = 50
        self.c = 1 #/ np.sqrt(2)
        self.treeSize = 1

    def bestChild(self, node, c):
        childInd = 0
        bestVal = float("-inf")
        for i in range(len(node.children)):
			# Check that this is correct UCT
            curVal = (node.children[i].value / node.children[i].visits) + (c * np.sqrt(2 * np.log(node.visits) / node.children[i].visits))
            if curVal > bestVal:
                childInd = i
                bestVal = curVal
        return node.children[childInd]

    def getAction(self, state):
        # ID, parent, state, action to get to state 
        root = MCTSNode(0, None, state, None)
        #iterationNumber = 0
        for _ in range(self.n):
        #startTime = time.time()
        #while time.time() - startTime < 0.25:
            node = self.treePolicy(root)
            delta = self.defaultPolicy(node.state)
            self.backup(node, delta)
            #iterationNumber += 1
        #print str(iterationNumber) + " iterations"
        chosenAction = self.bestChild(root, 0).action
        #self.recurseDeleteTree(root)
        node.state.getAndResetExplored() #!
        return chosenAction

        
    def treePolicy(self, node):

        def fullyExpanded(node):
            return len(node.children) == len(node.state.getLegalPacmanActions())

        def expand(node):
			# Check that the random action hasn't been taken before (it'll be in another child if it has)
            action = np.random.choice(node.state.getLegalPacmanActions())
            while action in node.triedActions:
                action = np.random.choice(node.state.getLegalPacmanActions())
            newState = node.state.generatePacmanSuccessor(action)
            newChild = MCTSNode(self.treeSize, node, newState, action)
            node.children.append(newChild)
            node.triedActions.append(action)
            self.treeSize += 1
            return newChild
        

        while (not node.state.isWin()) and (not node.state.isLose()):
            if not fullyExpanded(node):
                return expand(node)
            else:
                node = self.bestChild(node, self.c)
        return node

    def defaultPolicy(self, state):
        i = 0
        while (not state.isWin()) and (not state.isLose()):
            action = np.random.choice(state.getLegalPacmanActions())
            state = state.generatePacmanSuccessor(action)
            i += 1
            if i > 10:
                break
        return state.getScore()
        # if state.isWin():
        #     return 1
        # elif state.isLose():
        #     return 0
        # else:
        #     print "Error: State was neither win nor lose!"
        #     return 0

    def backup(self, node, delta):
        while not node == None:
            node.visits += 1
            node.value += delta
            node = node.parent

    # def recurseDeleteTree(self, node):
    #     for child in node.children:
    #         self.recurseDeleteTree(child)
    #     del node.ID
    #     del node.parent
    #     del node.children
    #     del node.value
    #     del node.visits
    #     del node.state
    #     del node.action
    #     del node.triedActions
    #     del node
    #     return

if __name__ == '__main__':
    #str_args = ['-g', 'DirectionalGhost', '--frameTime', '0']   
    str_args = ['--frameTime', '0', "-l", "openClassic", "-n", "50"] 
    #str_args = ['-l', 'TinyMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '10']
    #str_args = ['-l', 'TestMaze', '-g', 'DirectionalGhost', '--frameTime', '0', '-n', '10']
    args = readCommand(str_args)
    # args['display'] = textDisplay.NullGraphics()  # Disable rendering

    args['pacman'] = MCTSagent()
    out = runGames( **args)
    scores = [o.state.getScore() for o in out]  
    print(scores)