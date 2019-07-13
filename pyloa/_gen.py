import os
import logging
from collections import OrderedDict
from shutil import copyfile

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from pyloa.agent import BeladysOptimalAgent, RLAgent
from pyloa.utils import tensorboard_array

count_index = 0


def run(args, params, dirs):
    """Sets up all necessary objects and containers before genetic algorithm is envoked to evaluate agent.

    Firstly, instantiates the necessary objects defined in params (agent, sequence_generator, environment); secondly.
    backs up the configuration file and starts to eval agent. Agent, Environment and Generator have to be a "paging
    problem instance" for genetic evaluation (vertex problem is NOT applicable for this runmode).

    Args:
        args (argparse.Namespace): namespace object of parsed arguments via argparse
        params (Dict[str, any]): dictionary defined in config python file
        dirs (OrderedDict[str, str]): dictionary of file/path locations
    """
    # instantiate objects
    sequence_generator = params['instance']['type'](**params['instance'])
    environment = params['environment']['type'](**params['environment'])
    eval_writer = tf.summary.FileWriter(os.path.join(dirs['log_dir'], 'eval')) if not args.nlog else None
    agent = params['agent']['type'](**params['agent'], state_dim=environment.state_dim,
                                    action_dim=environment.action_dim)

    # load agent from save_file, set epsilon to 1.0 (pure evaluation)
    if isinstance(agent, RLAgent):
        agent.load()
        agent.epsilon = 1.0

    # agent container
    agents = OrderedDict()
    belady_agent = BeladysOptimalAgent(state_dim=environment.state_dim, action_dim=environment.action_dim)
    agents[id(belady_agent)] = belady_agent
    agents[id(agent)] = agent

    # prepare outputs
    if not args.nsave:
        _backup_file_path = os.path.join(dirs['output_dir'], dirs['config_file_name']+'.py')        # backup target
        copyfile(src=dirs['config_file_path'], dst=_backup_file_path)                               # copy and log
        logging.info("backed up '{}' as '{}'".format(dirs['config_file_path'], _backup_file_path))

    # evaluate
    diversification_size = params['diversification_size'] if 'diversification_size' in params else \
        len(sequence_generator.pages)**2
    evaluate(
        sequence_size=sequence_generator.sequence_size,
        pages=sequence_generator.pages,
        cache_size=environment.cache_size,
        window_size=params['window_size'] if 'window_size' in params else environment.cache_size,
        agents=agents,
        environment=environment,
        eval_writer=eval_writer,
        diversification_size=diversification_size,
        start_size=params['start_size'] if 'start_size' in params else 20,
        selection_size=params['selection_size'] if 'selection_size' in params else 5,
    )


def generate_start_individuum(pages, sequence_size, window_size):
    # Todo @schoenitz documentation
    seq = np.full(sequence_size, np.random.choice(pages, 1))
    seq[0:window_size] = create_window_mutation(pages, window_size)
    return seq


def create_window_mutation(pages, w_size, replace=False, ignore_as_first_page=None):
    # Todo @schoenitz documentation
    if not replace and w_size > len(pages):
        raise ValueError("window size must not exceed the number of pages")
    if ignore_as_first_page is None:
        return np.random.choice(pages, w_size, replace=replace)
    else:
        return np.append(
            np.random.choice([page for page in pages if page != ignore_as_first_page], 1),
            np.random.choice(pages, w_size - 1, replace=replace)
        )


def index_branching(sequence, pages, window_size, index):
    # Todo @schoenitz documentation
    branched_sequences = []
    for relative_idx in range(window_size):
        for page in pages:
            branched_sequences.append(np.copy(sequence))
            branched_sequences[-1][index + relative_idx] = page
    return branched_sequences


def genetic_mutation(population, pages, diversification_size, window_size, index):
    # Todo @schoenitz documentation
    _new_population = []
    for individuum in population:
        for _ in range(diversification_size):
            _tmp = np.copy(individuum)
            _new_window = create_window_mutation(pages, window_size, ignore_as_first_page=_tmp[index - 1])
            _tmp[index:index + window_size] = _new_window
            _new_population.append(_tmp)
    return _new_population


def genetic_branching(population, pages, window_size, index):
    # Todo @schoenitz documentation
    _new_population = []
    for individuum in population:
        _new_population.extend(index_branching(sequence=individuum, pages=pages, window_size=window_size, index=index))
    return _new_population


def genetic_selection(sequences, scores, selection_size):
    # Todo @schoenitz documentation
    _args_top_scores = np.argsort(scores)[-selection_size:][::-1]
    _best_individuums = [sequences[arg] for arg in _args_top_scores]
    _best_scores = [scores[arg] for arg in _args_top_scores]
    return _best_individuums, _best_scores


def evaluate_population(agents, environment, population, eval_writer):
    # Todo @schoenitz documentation
    global count_index
    _approximations = []
    _len = len(population)

    for idx, sequence in enumerate(population):
        _agent_steps, _belady_steps = 0, 0
        _agent_faults, _belady_faults = 0, 0
        _agent_costs, _belady_costs = 0, 0
        # reset envs, set instance to each env
        for key, agent in agents.items():
            environment.reset()
            environment.instance = sequence

            # evaluate agent on env
            _step, _cost, _fault, = agent.play_instance(environment=environment, writer=eval_writer,
                                                        episode=count_index, is_train=False)
            if isinstance(agent, BeladysOptimalAgent):
                _belady_steps += _step
                _belady_costs += _cost
                _belady_faults += _fault
            else:
                _agent_steps += _step
                _agent_costs += _cost
                _agent_faults += _fault
                _approx = 1.0 if _belady_faults == 0 else _agent_faults / _belady_faults
                _approximations.append(_approx)
                tensorboard_array(eval_writer, 'approx/'+str(agent), _approximations, count_index)
        count_index += 1
    return _approximations


def evaluate(sequence_size, pages, cache_size, window_size, agents, environment, eval_writer, diversification_size,
             start_size=25, selection_size=5):
    """Performs a genetic algorithm on agents to calculate the competitive ratio of agent vs. BeladysOptimalAgent.

    #Todo @schoenitz LONG DESCRIPTION

    Args:
        sequence_size (int): size of sequences
        pages (np.ndarray): pages that each sequence may contain
        cache_size (int): cache_size of paging problem
        window_size (int): window_size that genetic may use to alter a sequence
        agents (OrderedDict[int, pyloa.agent.PagingAgent]): two agent instances (Belady and testee) to evaluate
        environment (pyloa.environment.PagingEnvironment): environment to evluate on
        eval_writer (tf.summary.FileWriter): tensorboard logging for evaluation
        diversification_size (int): number of diversifications per individuum during mutation step
        start_size (int): size of initial start population
        selection_size (int): number of "best performing individuums" to keep during selection steo
    """
    # vars
    episodes = int((sequence_size-cache_size)/window_size)
    gen = (cache_size + window_size * i for i in range(episodes))
    current_population = [generate_start_individuum(pages, sequence_size, window_size) for _ in range(start_size)]

    logging.info("starting genetic algorithm evaluation with window_size={}, diversification_size={}, start_size={}, "
                 "selection_size={}".format(window_size, diversification_size, start_size, selection_size))

    t = tqdm(iterable=enumerate(gen), desc='GeneticAlgorithm', total=episodes+1)
    for episode, window_index in t:
        # eval step
        scores = evaluate_population(agents, environment, current_population, eval_writer)
        logging.info("metrics after branch-{}: population_size={}, mean={:3.3f}, max={:3.3f}, min={:3.3f}, std={:3.3f}"
                     .format(episode, len(scores), np.mean(scores), np.max(scores), np.min(scores), np.std(scores)))

        # select best performing individuums, wipe the rest
        current_population, scores = genetic_selection(current_population, scores, selection_size)
        logging.info("metrics after select-{}: population_size={}, mean={:3.3f}, max={:3.3f}, min={:3.3f}, std={:3.3f}"
                     .format(episode, len(scores), np.mean(scores), np.max(scores), np.min(scores), np.std(scores)))

        # mutation step
        current_population = genetic_mutation(current_population, pages, diversification_size, window_size, window_index)

        # eval step
        scores = evaluate_population(agents, environment, current_population, eval_writer)
        logging.info("metrics after mutate-{}: population_size={}, mean={:3.3f}, max={:3.3f}, min={:3.3f}, std={:3.3f}"
                     .format(episode, len(scores), np.mean(scores), np.max(scores), np.min(scores), np.std(scores)))

        # select best performing individuums, wipe the rest
        current_population, scores = genetic_selection(current_population, scores, selection_size)
        logging.info("metrics after select-{}: population_size={}, mean={:3.3f}, max={:3.3f}, min={:3.3f}, std={:3.3f}"
                     .format(episode, len(scores), np.mean(scores), np.max(scores), np.min(scores), np.std(scores)))

        # branching step
        current_population = genetic_branching(current_population, pages, window_size, window_index)

    # final eval step
    scores = evaluate_population(agents, environment, current_population, eval_writer)
    final_population, scores = genetic_selection(current_population, scores, selection_size)
    logging.info("final average genetic score: {:3.3f}".format(np.mean(scores)))
    logging.info("Top-{} performing individuums:".format(selection_size))
    for individuum in final_population:
        logging.info("{}".format(individuum))
    logging.info("genetic algorithm finished!")
