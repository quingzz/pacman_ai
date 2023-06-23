# AI for different Pacman problems


### Table of content
|  Project name | Topics |  Description |  
|---|---|---|
| search  | Search (BFS, DFS, UCS, A*)  | Find shortest path for pacman to eat all food  |  
| multiagent | Minimax | Maximize food being eaten while chased by ghosts |  
| reinforcement | Q Learning | Use reinforcement learning to maximize food being eaten while chased by ghosts|   
----

### Some sample commands for testing and GUI demo

#### Search
Run autograder  
`python autograder.py`   

Run pacman with A* agent, in bigMaze, using manhattan distance as heuristic  
`python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic`

#### Multiagent
Run autograder  
`python autograder.py`  

Run pacman with minimax agent, in smallClassic maze
`python pacman.py -p AlphaBetaAgent -l smallClassic`

#### Reinforcement
Run autograder    
`python autograder.py`  

Run pacman with Q Learning agent, in smallGrid maze, after 2000 trainings  
`python pacman.py -p PacmanQAgent -x 2000 -n 2010 -l smallGrid `

----
### Acknowledgement
Pacman project structure and autograder was provided by [UC Berkeley](http://ai.berkeley.edu/home.html)