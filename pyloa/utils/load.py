import os
import sys
import logging
from importlib.util import find_spec, module_from_spec

from .format import str_iterable
from .exception import HyperparamsConfigurationException
from pyloa.instance import InstanceGenerator, SequenceGenerator
from pyloa.agent import Agent, RLAgent, PagingAgent
from pyloa.environment import Environment, PagingEnvironment


def load_hyperparameters(config_file_path, run_mode):
    """A helper function that imports a hyperparameter python-file and checks its dictionary 'params' for validity.

    Args:
        config_file_path (str): path to python-file
        run_mode (str): run mode of project

    Returns:
        dict: syntactically correct hyperparameters for run mode

    Raises:
        ModuleNotFoundError: if file can't be imported
        HyperparamsConfigurationException: if syntax of config-file is invalid (missing values, incorrect types, etc.)
    """
    config_dir, config_file = os.path.split(config_file_path)
    config_file_name, config_file_type = config_file.rsplit(".", 1)

    # update sys path, load from spec
    sys.path.append(config_dir)
    spec = find_spec(config_file_name)
    if spec is None:
        raise ModuleNotFoundError("Can not import module '{}' via spec")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.remove(config_dir)
    params = module.params

    # check params for completeness and validity
    if not isinstance(params, dict):
        raise HyperparamsConfigurationException("params in config_file='{}' must be of type='{}'".format(
            config_file_path, dict))
    if 'instance' not in params:
        raise HyperparamsConfigurationException("instance must be specified in params")
    if 'type' not in params['instance']:
        raise HyperparamsConfigurationException("type of instance must be specified in params")
    if not issubclass(params['instance']['type'], InstanceGenerator):
         raise HyperparamsConfigurationException("instance must be subclassed of '{}'".format(InstanceGenerator))
    if 'environment' not in params:
        raise HyperparamsConfigurationException("environment must be specified in params")
    if 'type' not in params['environment']:
        raise HyperparamsConfigurationException("type of environment must be specified in params")
    if not issubclass(params['environment']['type'], Environment):
        raise HyperparamsConfigurationException("environment must be subclassed of '{}'".format(Environment))
    if 'agent' not in params:
        raise HyperparamsConfigurationException("agent must be specified in params")

    # run mode specific entries
    if run_mode == 'train':
        if 'type' not in params['agent']:
            raise HyperparamsConfigurationException("type of agent must be specified in params")
        if not issubclass(params['agent']['type'], RLAgent):
            raise HyperparamsConfigurationException("agent must be subclassed of '{}'".format(RLAgent))
        if 'checkpoint_step' not in params:
            raise HyperparamsConfigurationException("checkpoint_step must be specified in params")
        if not isinstance(params['checkpoint_step'], int) or not params['checkpoint_step'] >= 0:
            raise HyperparamsConfigurationException("checkpoint_step must be unsigned int")
    if run_mode == 'eval':
        if not isinstance(params['agent'], list):
            raise HyperparamsConfigurationException("agent must be a list of agents to evaluate")
        if len(params['agent']) == 0:
            raise HyperparamsConfigurationException("list of agents to evaluate must contain at least one item")
        for agent in params['agent']:
            if 'type' not in agent:
                raise HyperparamsConfigurationException("type of agent='{}' must be specified in params".format(agent))
            if not issubclass(agent['type'], Agent):
                raise HyperparamsConfigurationException("agent must be subclassed of '{}'".format(Agent))
        if 'root_dir' not in params:
            raise HyperparamsConfigurationException("root_dir must be specified to evaluate learned agents")
    if run_mode == 'gen':
        if 'type' not in params['agent']:
            raise HyperparamsConfigurationException("type of agent must be specified in params")
        if not issubclass(params['agent']['type'], PagingAgent):
            raise HyperparamsConfigurationException("agent must be subclassed of '{}'".format(PagingAgent))
        if not issubclass(params['environment']['type'], PagingEnvironment):
            raise HyperparamsConfigurationException("environment must be subclassed of '{}'".format(PagingEnvironment))
        if not issubclass(params['instance']['type'], SequenceGenerator):
            raise HyperparamsConfigurationException("instance must be subclassed of '{}'".format(SequenceGenerator))

    # log and ship
    logging.info("params from {} successfully loaded".format(config_file_path))
    logging.debug("params={}".format(str_iterable(params)))
    return params
