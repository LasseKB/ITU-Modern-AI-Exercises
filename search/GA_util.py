import numpy as np
import random


opposites = {
    "North" : "South",
    "South": "North",
    "West": "East",
    "East": "West",
}


class Sequence:
    """ Continues until one failure is found."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        for node in self.children:
            result = node.__call__(state)
            if not result:
                return False
        return result


class Selector:
    """ Continues until one success is found."""
    def __init__(self, parent=None):
        self.parent = parent
        self.children = []

    def add_child(self, child):
        self.children.append(child)
        return child

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        for node in self.children:
            result = node.__call__(state)
            if result:
                return result
        return False


class CheckValid:
    """ Check whether <direction> is a valid action for PacMan
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        return self.direction in state.getLegalActions()


class CheckDanger:
    """ Check whether there is a ghost in <direction>, or any of the adjacent fields.
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        newState = state.generatePacmanSuccessor(self.direction)
        pacmanPos = newState.getPacmanPosition()
        ghostPosList = state.getGhostPositions()
        ghostPosAdjList = []
        for pos in ghostPosList:
            ghostPosAdjList.append((pos[0] - 1, pos[1]))
            ghostPosAdjList.append((pos[0] + 1, pos[1]))
            ghostPosAdjList.append((pos[0], pos[1] - 1))
            ghostPosAdjList.append((pos[0], pos[1] + 1))
        if pacmanPos in ghostPosList or pacmanPos in ghostPosAdjList:
            #print "Ghost WAS in direction " + str(self.direction)
            return True
        #print "Ghost was NOT in direction " + str(action)
        return False

    def is_dangerous(self, state):
        """ YOUR CODE HERE!"""

class ActionGo:
    """ Return <direction> as an action. If <direction> is 'Random' return a random legal action
    """
    def __init__(self, direction="Random"):
        self.direction = direction

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        if self.direction == "Random":
            legalActions = state.getLegalActions()
            i = random.randint(0, len(legalActions) - 1)
            return legalActions[i]
        #print "Returning: " + str(self.direction) + " with type: " + str(type(self.direction))
        return self.direction


class ActionGoNot:
    """ Go in a random direction that isn't <direction>
    """
    def __init__(self, direction):
        self.direction = direction

    def __call__(self, state):
        """ YOUR CODE HERE!"""
        for aDir in state.getLegalActions():
            if not aDir == self.direction and not aDir == "Stop":
                return aDir


class DecoratorInvert:
    def __call__(self, arg):
        return not arg



def parse_node(genome, parent=None):
    if len(genome) == 0:
        return

    if isinstance(genome[0], list):
        parse_node(genome[0], parent)
        parse_node(genome[1:], parent)

    elif genome[0] == "SEQ":
        if parent is not None:
            node = parent.add_child(Sequence(parent))
        else:
            node = Sequence(parent)
            parent = node
        parse_node(genome[1:], node)

    elif genome[0] == 'SEL':
        if parent is not None:
            node = parent.add_child(Selector(parent))
        else:
            node = Selector(parent)
            parent = node
        parse_node(genome[1:], node)

    elif genome[0].startswith("Valid"):
        arg = genome[0].split('.')[-1]
        parent.add_child(CheckValid(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("Danger"):
        arg = genome[0].split('.')[-1]
        parent.add_child(CheckDanger(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("GoNot"):
        arg = genome[0].split('.')[-1]
        parent.add_child(ActionGoNot(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0].startswith("Go"):
        arg = genome[0].split('.')[-1]
        parent.add_child(ActionGo(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    elif genome[0] == ("Invert"):
        arg = genome[0].split('.')[-1]
        parent.add_child(DecoratorInvert(arg))
        if len(genome) > 1:
            parse_node(genome[1:], parent)

    else:
        print("Unrecognized in ")
        raise Exception

    return parent




