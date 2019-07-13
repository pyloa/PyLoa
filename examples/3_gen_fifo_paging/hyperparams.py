import os
from pyloa.agent import FIFOAgent
from pyloa.environment import DefaultPagingEnvironment
from pyloa.instance import RandomSequenceGenerator

# vars
sequence_size = 100
max_page = 6
min_page = 1
episodes = 250
cache_size = 5

# hyperparams
params = {
    'checkpoint_step': 0,
    'diversification_size': 20,
    'window_size': 3,
    'start_size': cache_size * 4,
    'selection_size': 8,
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
        'cache_size': cache_size,
        'num_pages': max_page - min_page + 1,
        'include_request': True,
        'include_pagefault': True,
        'include_cache': True,
        'include_time_in_cache': False,
        'include_time_since_touched': False,
        'include_histogram': False,
    },
    'agent': {
        'type': FIFOAgent,
    },
}