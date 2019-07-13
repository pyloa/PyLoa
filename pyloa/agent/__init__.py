# define abstract alias imports for other modules
from .agent import Agent
from .rlagent import RLAgent
from .qagent import QAgent
from .pagingagent import PagingAgent
from .vertexagent import VertexAgent
# define alias imports for other modules
from .qmlp import QMLPAgent
from .qlstm import QLSTMAgent
from .qtable import QTableAgent
from .qmlpvertex import QMLPVertexAgent
from .qlstmvertex import QLSTMVertexAgent
from .pagingagent import FIFOAgent, LIFOAgent, BeladysOptimalAgent, LeastFrequentlyUsedAgent, MostFrequentlyUsedAgent, \
    LeastRecentlyUsedAgent, MostRecentlyUsedAgent, RandomAgent
from .vertexagent import FIFOVertexAgent, RandomVertexAgent
