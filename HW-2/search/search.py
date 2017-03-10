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


dfs_visited = set()
dfs_ans = []
dfs_global_reached = False


def do_dfs(problem, state, answer):
    """
    Performs a recursive dfs.
    :param problem:
    :param state:
    :param answer: the path to the goal
    :return: a list of directions to the goal
    """
    global dfs_ans, dfs_visited, dfs_global_reached
    if (state[0] in dfs_visited) or dfs_global_reached:
        return
    else:
        dfs_visited.add(state[0])
        answer.append(state[1])
        if problem.isGoalState(state[0]):
            print "Goal!"
            dfs_ans = answer[1:]
            dfs_global_reached = True
        successors = problem.getSuccessors(state[0])
        successors.reverse()
        for successor in successors:
            ans_deep = answer[:]
            do_dfs(problem, successor, ans_deep)

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
    global dfs_ans
    answer = []
    visited = set()
    do_dfs(problem, (problem.getStartState(), "crap", 10), answer)
    return dfs_ans


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
            print cur_path
            return cur_path
        else:
            for successor in problem.getSuccessors(cur_node[0]):
                nodes.append((successor, cur_path))



def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
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
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
