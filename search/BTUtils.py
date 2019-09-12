import random
import searchAgents

class BTNode:

    def __init__(self, children):
        self.children = children

class BTSequence(BTNode):

    def evaluate(self):
        for node in self.children:
            result = node.evaluate()
            if not result:
                return False
        return True

class BTSelector(BTNode):

    def evaluate(self):
        for node in self.children:
            result = node.evaluate()
            if result:
                return True
        return False

class BTLeaf(BTNode):

    def __init__(self, function, params=None):
        self.function = function
        self.params = params

    def evaluate(self):
        if (self.params == None):
            return self.function()
        return self.function(self.params)

def checkActionLegal(params):
    action, state = params
    result = action in state.getLegalActions()
    #if (result): print "Action " + str(action) + " was legal"
    #else: print "Action " + str(action) + " was NOT legal"
    return result

def checkNoGhost(params):
    action, state = params
    newState = state.generatePacmanSuccessor(action)
    pacmanPos = newState.getPacmanPosition()
    ghostPosList = state.getGhostPositions()
    ghostPosAdjList = []
    for pos in ghostPosList:
        ghostPosAdjList.append((pos[0] - 1, pos[1]))
        ghostPosAdjList.append((pos[0] + 1, pos[1]))
        ghostPosAdjList.append((pos[0], pos[1] - 1))
        ghostPosAdjList.append((pos[0], pos[1] + 1))
    if pacmanPos in ghostPosList or pacmanPos in ghostPosAdjList:
        #print "Ghost WAS in direction " + str(action)
        return False
    #print "Ghost was NOT in direction " + str(action)
    return True

def takeAction(params):
    action, state = params
    #print "Taking action " + str(action)
    searchAgents.BTAgent.actionToTake = action
    return True

def takeRandomLegalAction(state):
    legalActions = state.getLegalActions()
    i = random.randint(0, len(legalActions) - 1)
    searchAgents.BTAgent.actionToTake = legalActions[i]
    #print "Taking random action " + str(legalActions[i])
    return True

def checkCapsule(params):
    action, state = params
    newState = state.generatePacmanSuccessor(action)
    pacmanPos = newState.getPacmanPosition()
    return pacmanPos in state.getCapsules()

def checkFood(params):
    action, state = params
    newState = state.generatePacmanSuccessor(action)
    pacmanPos = newState.getPacmanPosition()
    return state.getFood()[pacmanPos[0]][pacmanPos[1]]

def takeRandomNoGhostAction(state):
    ghostNear = True
    legalActions = state.getLegalActions()
    tries = 0
    i = 0
    while (ghostNear or legalActions[i] == "Stop") and tries < 10:
        i = random.randint(0, len(legalActions) - 1)
        ghostNear = not checkNoGhost([legalActions[i], state])
        tries += 1
    if (tries < 10):
        searchAgents.BTAgent.actionToTake = legalActions[i]
        #print "Taking random action " + str(legalActions[i])
        return True
    return False

# def parseRepresentation(rep, state):
#     if rep[0] is "SEQ":
#         children = []
#         for elem in rep[1]:
#             children.append(parseRepresentation(elem, state))
#         return BTSequence(children)
    
#     elif rep[0] is "SEL":
#         children = []
#         for elem in rep[1]:
#             children.append(parseRepresentation(elem, state))
#         return BTSelector(children)

#     elif rep[0].startswith("checkActionLegal"):
#         arg = rep[0].split(".")[1]
#         return BTLeaf(checkActionLegal, arg)

#     elif rep[0].startswith("checkNoGhost"):
#         arg = rep[0].split(".")[1]
#         return BTLeaf(checkNoGhost, arg)

#     elif rep[0].startswith("takeAction"):
#         arg = rep[0].split(".")[1]
#         return BTLeaf(takeAction, arg)

#     else:
#         raise NotImplementedError