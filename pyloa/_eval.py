import os
import logging
from shutil import copyfile
from collections import OrderedDict
from pathlib import Path

import tensorflow as tf
from tqdm import tqdm

from pyloa.utils import str_iterable, tensorboard_text, load_hyperparameters


def run(args, params, dirs):
    """Sets up all necessary objects and containers before training.

    Firstly, instantiates the necessary objects defined in params (agent, instance_generator, environment); secondly.
    backs up the configuration file and starts to eval all agents. Agents are loaded both from pre-trained agents (all
    hyperparams definitons nested in root_dir) and from deterministic (non-trained) agents defined within params.

    Args:
         args (argparse.Namespace): namespace object of parsed arguments via argparse
         params (Dict[str, any]): dictionary defined in config python file
         dirs (OrderedDict[str, str]): dictionary of file/path locations
    """
    # instantiate objects
    instance_generator = params['instance']['type'](**params['instance'])
    eval_writer = tf.summary.FileWriter(os.path.join(dirs['log_dir'], 'eval')) if not args.nlog else None
    if eval_writer is not None:
        tensorboard_text(writer=eval_writer, tag='hyperparams', value=str_iterable(params, 1, markdown_newline=True))

    # scan for backed up hyperparams in root_dir (these are the trained agents to be evaluated)
    filepaths = [os.path.abspath(os.path.realpath(str(f))) for f in Path(params['root_dir']).glob('**/hyperparams*.py')]
    trained_params = []
    # iterate over params, and load each param instance
    for filepath in filepaths:
        _all_files = os.listdir(os.path.dirname(filepath))
        for _file in _all_files:
            if any(_file.endswith(ext) for ext in [".meta", ".pkl", ".pickle"]):
                trained_params.append(load_hyperparameters(config_file_path=filepath, run_mode='train'))
                break
    logging.info("loading following pre-trained models: {}".format(trained_params))

    # instantiate objects
    environments, tf_graphs, agents = OrderedDict(), OrderedDict(), OrderedDict()

    # load pre-trained agents
    for param in trained_params:
        _env = param['environment']['type'](**param['environment'])
        _graph = tf.Graph()
        with _graph.as_default():
            _agent = param['agent']['type'](**param['agent'], state_dim=_env.state_dim, action_dim=_env.action_dim)
            _agent.load()
            _agent.epsilon = 1.0
        _key = id(_agent)
        environments[_key], agents[_key], tf_graphs[_key] = _env, _agent, _graph

    # load default deterministic agents
    for agent_dict in params['agent']:
        _env = params['environment']['type'](**params['environment'])
        _agent = agent_dict['type'](**agent_dict, state_dim=_env.state_dim, action_dim=_env.action_dim)
        _key = id(_agent)
        environments[_key], agents[_key] = _env, _agent

    # prepare outputs
    if not args.nsave:
        _backup_file_path = os.path.join(dirs['output_dir'], dirs['config_file_name']+'.py')        # backup target
        copyfile(src=dirs['config_file_path'], dst=_backup_file_path)                               # copy and log
        logging.info("backed up '{}' as '{}'".format(dirs['config_file_path'], _backup_file_path))

    # let's evaluate
    evaluate(
        instance_generator=instance_generator,
        environments=environments,
        agents=agents,
        eval_writer=eval_writer
    )


def evaluate(instance_generator, environments, agents, eval_writer):
    """Agents are evaluated on their environments by playing instances.

    instance_generator yields an instance for each environment that its respective agent plays on. Each instance
    constitutes as one episode; each environment is being reset for each instance.

    Args:
        instance_generator (pyloa.instance.InstanceGenerator): generator, which iterates instances to evaluate on
        environments (OrdererdDict[int, pyloa.environment.Environment]): envs, which consume instances and return states
            & rewards
        agents (OrdererdDict[int, pyloa.agent.Agent]): agents that play on their respective environments via actions
        eval_writer (tf.summary.FileWriter): tensorboard logging for evaluation
    """
    logging.info("begin evaluating {} on {} instances from {}".format(str(list(agents.values())),
                                                                      len(instance_generator), str(instance_generator)))

    # iterate over instances
    t = tqdm(instance_generator, desc=str(instance_generator), unit='instance')
    for episode, instance in enumerate(t):
        for _, environment in environments.items():
            environment.reset()
            environment.instance = instance

        for key, agent in agents.items():
            agent.play_instance(environment=environments[key], episode=episode, is_train=False, writer=eval_writer)
        logging.info("finished evaluating episode={}".format(episode))
