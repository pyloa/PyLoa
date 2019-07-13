from pyloa.environment import DefaultPagingEnvironment
from pyloa.instance import RandomSequenceGenerator
from pyloa.agent import *

# vars
sequence_size = 1000
max_page = 6
min_page = 1
episodes = 100

# hyperparams
params = {
    'checkpoint_step': 0,
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
    },
    # recursively scans root_dir and loads all trained agent nested within root_dir with their respective hyperparams
    'root_dir': '/home/mr702783/Workspace/paging/examples/',
    # list of deterministic agent definitions, each a dictionary
    'agent': [
        {'type': FIFOAgent},
        {'type': BeladysOptimalAgent},
        {'type': LIFOAgent},
        {'type': MostFrequentlyUsedAgent},
        {'type': MostRecentlyUsedAgent},
        {'type': RandomAgent},
        {'type': LeastRecentlyUsedAgent},
        {'type': LeastFrequentlyUsedAgent},
    ]
}
