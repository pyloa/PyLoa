import os
from pyloa.agent import QMLPAgent
from pyloa.environment import DefaultPagingEnvironment
from pyloa.instance import RandomSequenceGenerator

# vars
sequence_size = 1000
max_page = 6
min_page = 1
episodes = 250

# hyperparams
params = {
    'checkpoint_step': episodes//10,
    'instance': {
        'type': RandomSequenceGenerator,
        'sequence_size': sequence_size,
        'sequence_number': episodes,
        'min_page': min_page,
        'max_page': max_page,
        'seed': None,
    },
    'environment': {
        'type': DefaultPagingEnvironment,
        'sequence_size': sequence_size,
        'cache_size': 5,
        'num_pages': max_page - min_page + 1,
        'include_request': True,
        'include_pagefault': True,
        'include_cache': True,
        'include_time_in_cache': False,
        'include_time_since_touched': False,
        'include_histogram': False,
    },
    'agent': {
        'type': QMLPAgent,
        'name': QMLPAgent.__name__+'-001',
        'units': 256,
        'discount_factor': 0.9,
        'learning_rate': 0.0001,
        'memory_size': 2**6 * sequence_size,
        'sample_size': 2**4 * sequence_size,
        'batch_size': 2**6,
        'epsilon': 0.0,
        'epsilon_delta': 13 / (episodes * 10),
        'epsilon_max': 0.99,
        'replace_threshold': 8,
        'save_file': os.path.abspath(os.path.relpath(__file__)),
    },
}