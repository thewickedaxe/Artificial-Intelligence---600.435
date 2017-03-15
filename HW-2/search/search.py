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
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    visited = set()
    init_state = (problem.getStartState(), "crap", 100, [])
    visited.add(init_state[0])
    node_stack = util.Stack()
    for successor_state in problem.getSuccessors(init_state[0]):
        node_stack.push((successor_state[0], successor_state[1], successor_state[2], []))
    while not node_stack.isEmpty():
        successor = node_stack.pop()
        if successor[0] in visited:
            continue
        else:
            visited.add(successor[0])
            if problem.isGoalState(successor[0]):
                answer = successor[3][:]
                answer.append(successor[1])
                print answer
                return answer
            else:
                for subsequent_sucessor in problem.getSuccessors(successor[0]):
                    apath = successor[3][:]
                    apath.append(successor[1])
                    node_stack.push((subsequent_sucessor[0], subsequent_sucessor[1], subsequent_sucessor[2], apath))


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    init_state = (problem.getStartState(), "crap", 100)
    visited = set()
    visited.add(init_state[0])
    nodes = []
    for im_desc in problem.getSuccessors(init_state[0]):
        nodes.append((im_desc, []))
    for node in nodes:
        cur_node, cur_path_ = node
        cur_path = cur_path_[:]
        if cur_node[0] in visited:
            continue
        visited.add(cur_node[0])
        cur_path.append(cur_node[1])
        if problem.isGoalState(cur_node[0]):
            return cur_path
        else:
            for successor in problem.getSuccessors(cur_node[0]):
                nodes.append((successor, cur_path))


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    init_state = (problem.getStartState(), "init_state", float("inf"))
    visited = set()
    visited.add(init_state[0])
    nodes = util.PriorityQueue()
    for im_desc in problem.getSuccessors(init_state[0]):
        nodes.push((im_desc, []), im_desc[2])
    while not nodes.isEmpty():
        cur_node, cur_path_ = nodes.pop()
        cur_path = cur_path_[:]
        if cur_node[0] in visited:
            continue
        visited.add(cur_node[0])
        cur_path.append(cur_node[1])
        if problem.isGoalState(cur_node[0]):
            return cur_path
        else:
            successors = problem.getSuccessors(cur_node[0])
            for successor in successors:
                total_cost = successor[2] + cur_node[2]
                successor_loc, successor_dir, successor_cost = successor
                nodes.push(
                    ((successor_loc, successor_dir, total_cost), cur_path), total_cost)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    """Search the node of least total cost first."""
    init_state = (problem.getStartState(), "init_state", float("inf"))
    visited = set()
    visited.add(init_state[0])
    nodes = util.PriorityQueue()
    for im_desc in problem.getSuccessors(init_state[0]):
        nodes.push((im_desc, []), (im_desc[2] +
                                   heuristic(im_desc[0], problem)))
    while not nodes.isEmpty():
        cur_node, cur_path_ = nodes.pop()
        cur_path = cur_path_[:]
        if cur_node[0] in visited:
            continue
        visited.add(cur_node[0])
        cur_path.append(cur_node[1])
        if problem.isGoalState(cur_node[0]):
            return cur_path
        else:
            successors = problem.getSuccessors(cur_node[0])
            for successor in successors:
                successor_loc, successor_dir, successor_cost = successor
                total_cost = successor_cost + \
                    cur_node[2] + heuristic(successor_loc, problem)
                nodes.push(((successor_loc, successor_dir,
                             successor_cost + cur_node[2]), cur_path), total_cost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
