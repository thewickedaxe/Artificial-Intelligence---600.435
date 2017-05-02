Commands to run the code -

1]

Depth First Search

For tinyMaze

python pacman.py -l tinyMaze -p SearchAgent -a fn=tinyMazeSearch

For mediumMaze

python pacman.py -l mediumMaze -p SearchAgent -a fn=tinyMazeSearch

For bigMaze

python pacman.py -l bigMaze -p SearchAgent -a fn=tinyMazeSearch

For openMaze

python pacman.py -l openMaze -p SearchAgent -a fn=tinyMazeSearch


2]

Breadth First Search

For tinyMaze

python pacman.py -l tinyMaze -p SearchAgent -a fn=bfs

For mediumMaze

python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs

For bigMaze

python pacman.py -l bigMaze -p SearchAgent -a fn=bfs

For openMaze

python pacman.py -l openMaze -p SearchAgent -a fn=bfs


3]

Uniform Cost Search

For mediumMaze with UCS Agent 

python pacman.py -l mediumMaze -p SearchAgent -a fn=ucs

For mediumDottedMaze with StayEastSearchAgent

python pacman.py -l mediumDottedMaze -p StayEastSearchAgent

For mediumScaryMaze with StayWestAgent

python pacman.py -l mediumScaryMaze -p StayWestSearchAgent


4]

A * search

Using Manhattan Heuristic with tinyMaze

python pacman.py -l tinyMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

Using Manhattan Heuristic with mediumMaze

python pacman.py -l mediumMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

Using Manhattan Heuristic with bigMaze

python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic

Using Manhattan Heuristic with openMaze

python pacman.py -l openMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic


5]

Corner problem with Breadth First Search

For tinyCorners

python pacman.py -l tinyCorners -p SearchAgent -a fn=bfs,prob=CornersProblem

For mediumCorners

python pacman.py -l mediumCorners -p SearchAgent -a fn=bfs,prob=CornersProblem


6]

Food problem with A*

For trickySearch

python pacman.py -l trickySearch -p AStarFoodSearchAgent

For testSearch 

python pacman.py -l testSearch -p AStarFoodSearchAgent


7]

Food Problem with A* with Heuristics

For trickySearch

python pacman.py -l trickySearch -p AStarFoodSearchAgent

For testSearch 

python pacman.py -l testSearch -p AStarFoodSearchAgent

