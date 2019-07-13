import os
import logging
from shutil import copyfile

import tensorflow as tf
from tqdm import tqdm

from pyloa.utils import str_iterable, tensorboard_text


def run(args, params, dirs):
    """Sets up all necessary objects and containers before training.

    Firstly, instantiates the necessary objects defined in params (agent, instance_generator, environment); secondly.
    backs up the configuration file and starts to train agent.

    Args:
         args (argparse.Namespace): namespace object of parsed arguments via argparse
         params (Dict[str, any]): dictionary defined in config python file
         dirs (OrderedDict[str, str]): dictionary of file/path locations
    """
    # instantiate objects
    instance_generator = params['instance']['type'](**params['instance'])
    environment = params['environment']['type'](**params['environment'])
    eval_writer = tf.summary.FileWriter(os.path.join(dirs['log_dir'], 'eval')) if not args.nlog else None
    train_writer = tf.summary.FileWriter(os.path.join(dirs['log_dir'], 'train')) if not args.nlog else None
    if eval_writer is not None:
        tensorboard_text(writer=eval_writer, tag='hyperparams', value=str_iterable(params, 1, markdown_newline=True))
    agent = params['agent']['type'](**params['agent'], state_dim=environment.state_dim,
                                    action_dim=environment.action_dim)
    _agent_file = os.path.join(dirs['output_dir'], str(agent))

    # prepare outputs
    if not args.nsave:
        _backup_file_path = os.path.join(dirs['output_dir'], dirs['config_file_name']+'.py')        # backup target
        copyfile(src=dirs['config_file_path'], dst=_backup_file_path)                               # copy and log
        logging.info("backed up '{}' as '{}'".format(dirs['config_file_path'], _backup_file_path))

    # let's train
    train(
        instance_generator=instance_generator,
        environment=environment,
        agent=agent,
        eval_writer=eval_writer,
        train_writer=train_writer,
        checkpoint_step=params['checkpoint_step'] if not args.nsave else 0,
        agent_file=_agent_file
    )


def train(instance_generator, environment, agent, eval_writer, train_writer, checkpoint_step, agent_file):
    """Agent learns his environment by playing instances.

    instance_generator yields an instance for the environment that the agent plays on. Each instance constitutes as one
    episode; the environment is being reset for each instance. The agent is evaluated and afterwards trained on each
    instance. A checkpoint is saved every checkpoint_step-many episodes.

    Args:
        instance_generator (pyloa.instance.InstanceGenerator): iterates instances to train on
        environment (pyloa.environment.Environment): consumes instances and returns states & rewards
        agent (pyloa.agent.Agent): plays/trains on environment via actions
        eval_writer (tf.summary.FileWriter): tensorboard logging for evaluation
        train_writer (tf.summary.FileWriter): tensorboard logging for training
        checkpoint_step (int): defines after how many episodes a checkpoint of agent is saved
        agent_file (str): file location of agent to be stored at
    """
    logging.info("begin training {} on {} instances from {}".format(agent, len(instance_generator), instance_generator))
    with tqdm(instance_generator, desc=str(agent), unit='instance') as t:
        for episode, instance in enumerate(t):
            # start a new episode
            environment.reset()
            environment.instance = instance

            # eval on new instance
            agent.play_instance(environment=environment, is_train=False, episode=episode, writer=eval_writer)

            # reset, train on same instance
            environment.reset()
            environment.instance = instance

            # train on new instance
            agent.play_instance(environment=environment, is_train=True, episode=episode, writer=train_writer)

            # save step
            if checkpoint_step and (episode + 1) % checkpoint_step == 0:
                agent.save(agent_file)

    logging.info("finished training {}".format(agent))
