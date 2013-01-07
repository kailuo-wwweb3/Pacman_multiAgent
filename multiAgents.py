# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util
import math

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


  def getAction(self, gameState):
    """
    You do not need to change this method, but you're welcome to.

    getAction chooses among the best options according to the evaluation function.

    Just like in the previous project, getAction takes a GameState and returns
    some Directions.X for some X in the set {North, South, West, East, Stop}
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    "Add more of your code here if you want to"

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    Design a better evaluation function here.

    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (newFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    newFood = successorGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    heuristic = 0
    
    for st in newScaredTimes:
        heuristic += st
    ghostDistances = []
    for gs in newGhostStates:
        ghostDistances += [manhattanDistance(gs.getPosition(),newPos)]

    foodList = newFood.asList()

    wallList = currentGameState.getWalls().asList()

    emptyFoodNeighbors = 0
    foodDistances = []

    for f in foodList:
        neighbors = foodNeighbors(f)
        for fn in neighbors:
            if fn not in wallList and fn not in foodList:
                emptyFoodNeighbors += 1
        foodDistances += [manhattanDistance(newPos,f)]
    inverseFoodDist = 0
    if len(foodDistances) > 0:
        inverseFoodDist = 1.0/(min(foodDistances))
     
    heuristic += (min(ghostDistances)*((inverseFoodDist**4)))
    heuristic += successorGameState.getScore()-(float(emptyFoodNeighbors)*4.5)
    return heuristic

def foodNeighbors(foodPos):
    foodNeighbors = []
    foodNeighbors.append((foodPos[0]-1,foodPos[1]))
    foodNeighbors.append((foodPos[0],foodPos[1]-1))
    foodNeighbors.append((foodPos[0],foodPos[1]+1))
    foodNeighbors.append((foodPos[0]+1,foodPos[1]))
    return foodNeighbors
    

def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)
    
    

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """

  def max_value(self, gameState, agentIndex, depth):
    legalActions = gameState.getLegalActions(agentIndex)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    v = float("-inf")
    for action in legalActions:
        successor = gameState.generateSuccessor(agentIndex, action)
        v = max(v, self.minimax_Decision(successor, agentIndex + 1, depth + 1))
    return v
  
  def min_value(self, gameState, agentIndex, depth):
    legalActions = gameState.getLegalActions(agentIndex)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    v = float("inf")
    for action in legalActions:
        successor = gameState.generateSuccessor(agentIndex, action)
        v = min(v, self.minimax_Decision(successor, agentIndex + 1, depth + 1))
    return v



  def minimax_Decision(self, gameState, agentIndex, depth):
    if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
    if agentIndex == gameState.getNumAgents():
        agentIndex = 0
    if agentIndex == 0:
        return self.max_value(gameState, agentIndex, depth)
    else:
        return self.min_value(gameState, agentIndex, depth)
      
    
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    state = gameState
    self.agentCount = state.getNumAgents()
    depth = 0
    agent = self.index
    actionDict = {}
    actions = state.getLegalActions(agent)
    actions.remove(Directions.STOP)
    for action in actions:
        val = self.minimax_Decision(state.generateSuccessor(agent, action),agent+1,depth+1)
        actionDict[val] = action
    return actionDict[max(actionDict)]



class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def ab_max_value(self, gameState, alpha, beta, agentIndex, depth):
    legalActions = gameState.getLegalActions(agentIndex)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    v = float("-inf")
    for action in legalActions:
        successor = gameState.generateSuccessor(agentIndex, action)
        v = max(v, self.ab_minimax_Decision(successor, alpha, beta, agentIndex + 1, depth + 1))
        if v >= beta:
            return v
        alpha = max(alpha, v)
    return v
  
  def ab_min_value(self, gameState, alpha, beta, agentIndex, depth):
    legalActions = gameState.getLegalActions(agentIndex)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    v = float("inf")
    for action in legalActions:
        successor = gameState.generateSuccessor(agentIndex, action)
        v = min(v, self.ab_minimax_Decision(successor, alpha, beta, agentIndex + 1, depth + 1))
        if v <= alpha:
            return v
        beta = min(beta, v)
    return v



  def ab_minimax_Decision(self, gameState, alpha, beta, agentIndex, depth):
    if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
    if agentIndex == gameState.getNumAgents():
        agentIndex = 0
    if agentIndex == 0:
        return self.ab_max_value(gameState, alpha, beta, agentIndex, depth)
    else:
        return self.ab_min_value(gameState, alpha, beta, agentIndex, depth)


  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    state = gameState
    self.agentCount = state.getNumAgents()
    depth = 0
    agent = self.index
    actionDict = {}
    actions = state.getLegalActions(agent)
    actions.remove(Directions.STOP)
    for action in actions:
        val = self.ab_minimax_Decision(state.generateSuccessor(agent, action), float("-inf"), float("inf"), agent+1,depth+1)
        actionDict[val] = action
    return actionDict[max(actionDict)]

    
class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """
  def ex_max_value(self, gameState, agentIndex, depth):
    legalActions = gameState.getLegalActions(agentIndex)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    v = float("-inf")
    for action in legalActions:
        successor = gameState.generateSuccessor(agentIndex, action)
        v = max(v, self.ex_minimax_Decision(successor, agentIndex + 1, depth + 1))
    return v
  
  def ex_min_value(self, gameState, agentIndex, depth):
    legalActions = gameState.getLegalActions(agentIndex)
    if Directions.STOP in legalActions:
        legalActions.remove(Directions.STOP)
    v = 0
    for action in legalActions:
        successor = gameState.generateSuccessor(agentIndex, action)
        v = v + self.ex_minimax_Decision(successor, agentIndex + 1, depth + 1)
    return v * 1.0 / len(legalActions)



  def ex_minimax_Decision(self, gameState, agentIndex, depth):
    if depth == self.depth * gameState.getNumAgents() or gameState.isWin() or gameState.isLose():
        return self.evaluationFunction(gameState)
    if agentIndex == gameState.getNumAgents():
        agentIndex = 0
    if agentIndex == 0:
        return self.ex_max_value(gameState, agentIndex, depth)
    else:
        return self.ex_min_value(gameState, agentIndex, depth)



  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"
    state = gameState
    self.agentCount = state.getNumAgents()
    depth = 0
    agent = self.index
    actionDict = {}
    actions = state.getLegalActions(agent)
    actions.remove(Directions.STOP)
    for action in actions:
        val = self.ex_minimax_Decision(state.generateSuccessor(agent, action),agent+1,depth+1)
        actionDict[val] = action
    return actionDict[max(actionDict)]
    
    

def minDistanceToGoal(pacmanPosition, foodList):
 
  
  xy1 = pacmanPosition
  distance = 0.0

  if(len(foodList) == 0):
      return distance
  
  #calculate which food is the nearest one to the pacman
  minDistanceFood = foodList[0]
  minDistanceToFood =  abs(xy1[0] - minDistanceFood[0]) + abs(xy1[1] - minDistanceFood[1]) 
  
  for food in foodList:
      xy2 = food
      tmpDistance =  abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
      if(tmpDistance < minDistanceToFood):
          minDistanceToFood = tmpDistance
          minDistanceFood = xy2
  
  foodList.remove(minDistanceFood)

  
  #calculate which food is the nearest to the minDistanceFood,for example, FoodA
  #then, calculate which food is the nearest food to FoodA, and so on...
  while len(foodList)>0:
      dist = 999999
      fd = []
      for food in foodList:
          tmpDist = abs(food[0]-minDistanceFood[0]) + abs(food[1]-minDistanceFood[1])
          if(dist > tmpDist):
              dist = tmpDist
              fd = food
      minDistanceToFood += dist
      minDistanceFood = fd
      foodList.remove(fd)

      
  return minDistanceToFood

     

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: My evaluation funciton is to take consideration of the score given by the current game state, the total distance to the 4 (Did some experients here) nearest
    food dots, whether the pacman ate pellet or not and the sum of distance to the nearest ghost and the second nearest ghost to the nearest ghost.
  """
  "*** YOUR CODE HERE ***"
  ghostStates = [gameState for gameState in currentGameState.getGhostStates()]
  ghostPositionList = [ gameState.getPosition() for gameState in ghostStates]
  foodList = currentGameState.getFood().asList()
  pacmanPosition = currentGameState.getPacmanPosition()
  scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

  
  distToFood = 0.0001
  foodWeight = 20
  ghostWeight = -0.5
  currentScore = currentGameState.getScore()
  powerPellet = 0
  
  distToGhost = minDistanceToGoal(pacmanPosition,ghostPositionList)
  
  foodToPacmanDistList = []
  for food in foodList:
     foodToPacmanDistList.append(manhattanDistance(food,pacmanPosition))
  
  foodToPacmanDistList.sort()

  p = 4
  
  if len(foodToPacmanDistList) < p:
      p = len(foodToPacmanDistList)
                
  for i in range(p):
      distToFood += foodToPacmanDistList[i]
     
     

      # The pacman the most powerful man!
  if scaredTimes[0] != 0 :
      powerPellet = float("inf")
  
  return currentScore + 1.0/distToFood * foodWeight + powerPellet + distToGhost * ghostWeight

  




# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"


def evaluationFunctionForContest(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  