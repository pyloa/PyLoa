import os
from pyloa.agent import QLSTMVertexAgent, QMLPVertexAgent
from pyloa.environment import VertexEnvironment
from pyloa.instance import DeltaRandomGraphGenerator
from pyloa.utils import NumpyGraph

# vars
max_color = 4
max_delta = 3
number_of_vertices = 8
episodes = 50000

# hyperparams
params = {
    'checkpoint_step': episodes//10,
    'instance': {
        'type': DeltaRandomGraphGenerator,
        'graph_type': NumpyGraph,
        'graph_number': episodes,
        'graph_size': number_of_vertices,
        'graph_color_number': max_color,
        'max_delta': max_delta,
        'seed': None,
    },
    'environment': {
        'type': VertexEnvironment,
        'max_color': max_color,
        'max_delta': max_delta,
        'include_vertex': False,
        'include_delta': True,
        'include_delta_squared': False,
        'include_color_flag': False,
        'include_num_steps': False,
        'include_embedding': False,
        'include_vertex_coloring': False,
    },
    'agent': {
        'type': QMLPVertexAgent,
        'name': QMLPVertexAgent.__name__+'-001',
        'units': 256,
        'discount_factor': 0.9,
        'learning_rate': 0.0001,
        'memory_size': 2**10 * number_of_vertices,
        'sample_size': 2**4 * number_of_vertices,
        'batch_size': 2**6,
        'epsilon': 0.0,
        'epsilon_delta': 13 / (episodes * 10),
        'epsilon_max': 0.99,
        'replace_threshold': 8,
        'save_file': os.path.abspath(os.path.relpath(__file__)),
    },
}