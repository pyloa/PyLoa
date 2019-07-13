#!/usr/bin/env python

"""PyLoa: A research repository for analyzing the performance of classic on-line algorithms vs. modern Machine
Learning, specifically Reinforcement Learning, approaches."""

__author__ = "Matthias Rozanski, Sebastian Schoenitz"
__copyright__ = "Copyright 2019, The PyLoa Project"
__credits__ = []
__license__ = "MIT"
__maintainer__ = "PyLoa Developers"
__email__ = "pyloadevelopers@gmail.com"
__status__ = "Prototype"

import os
import sys
import logging
import argparse
from datetime import datetime
from collections import OrderedDict

sys.path.append(os.path.dirname(sys.path[0]))

from pyloa.utils.load import load_hyperparameters
from pyloa.utils.format import str_iterable
from pyloa.utils.exception import DirectoryNotFoundError, InvalidFileType


def run(args, params, dirs):
    """Invokes the correct run mode.

    Args:
        args (argparse.Namespace): namespace object of parsed arguments via argparse
        params (Dict[str, any]): dictionary defined in config python file
        dirs (OrderedDict[str, str]): dictionary of file/path locations
    """
    if args.runmode == 'train':
        import pyloa._train
        pyloa._train.run(args=args, params=params, dirs=dirs)
    if args.runmode == 'eval':
        import pyloa._eval
        pyloa._eval.run(args=args, params=params, dirs=dirs)
    if args.runmode == 'gen':
        import pyloa._gen
        pyloa._gen.run(args=args, params=params, dirs=dirs)


def main():
    """Generic plumbing: resolve directories, files and paths, parse arguments and set up logging

    Raises:
        DirectoryNotFoundError: if specified config directory does not exist
        FileNotFoundError: if specified config file does not exist
        InvalidFileType: if specified config file is not a python-file
        NotADirectoryError: if specified output directory already exists as a file
    """
    # dirs, files and paths
    _project_directory = os.path.abspath(os.path.realpath(sys.path[0]))     # where python file is located
    _working_directory = os.path.abspath(os.path.realpath(os.getcwd()))     # where user calls python file from
    _config_directory = os.path.join(_working_directory, os.path.join('examples', '0_train_qtable_paging'))
    _config_file = 'hyperparams.py'
    _config_file_path = os.path.join(_config_directory, _config_file)
    _log_directory = _config_directory
    _log_file = 'default.log'
    _log_file_path = os.path.join(_log_directory, _log_file)
    _iso_time = 'E' + datetime.now().replace().isoformat().replace('-', '').replace(':', '').replace('.', '')
    _output_directory = os.path.join(_config_directory, _iso_time)

    # set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('runmode', type=str, choices=['train', 'eval', 'gen'],
                        help="depicts the run-mode of _main.py")
    parser.add_argument('-nl', '--no-log',  dest='nlog', default=False, action='store_true',
                        help="deactivates logger (default: '%(default)s')")
    parser.add_argument('-ns', '--no-save', dest='nsave', default=False, action='store_true',
                        help="deactivates saving model (default: '%(default)s')")
    parser.add_argument('-ll', '--loglevel', dest='loglevel', type=str, default='INFO',
                        choices=['CRITICAL', 'ERROR',  'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
                        help="depicts logging level (default: '%(default)s')")
    parser.add_argument('-lf', '--logfile', dest='logfile', type=str, default=_log_file_path,
                        help="destination of log file (default: '%(default)s')")
    parser.add_argument('-lfmt', '--logformat', dest='logformat', type=str,
                        default='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        help="defines logger message format (default: '%(default)s')")
    parser.add_argument('-ldfmt', '--logdateformat', dest='logdateformat', type=str, default='%m/%d/%Y %I:%M:%S %p',
                        help="defines logger timestamp format (default: '%(default)s')")
    parser.add_argument('-c', '--config',  dest='config', type=str, default=_config_file_path,
                        help="destination of config file (default: '%(default)s')")
    parser.add_argument('-o', '--output-dir', dest='outdir', type=str, default=_output_directory,
                        help="target output directory (default: '%(default)s')")
    args = parser.parse_args()

    # resolve input directories
    _config_directory, _config_file = os.path.split(os.path.expanduser(args.config))
    _config_file_name, _config_file_type = _config_file.rsplit('.', 1) if '.' in _config_file else (_config_file, None)
    _config_directory = os.path.abspath(os.path.realpath(_config_directory))
    # if user specified config directory, place outputs there aswell if not specified otherwise, else use default
    if _config_file_path != args.config and _output_directory == args.outdir:
        _output_directory = os.path.join(_config_directory, _iso_time)
    else:
        _output_directory = os.path.abspath(os.path.realpath(args.outdir))
    _config_file_path = os.path.join(_config_directory, _config_file)
    # if user specified logfile_path, use it, else put logfile in output_dir
    if _log_file_path != args.logfile:
        _log_directory, _log_file = os.path.split(os.path.expanduser(args.logfile))
    else:
        _log_directory, _log_file = _output_directory, os.path.split(os.path.expanduser(args.logfile))[-1]
    _log_directory = os.path.abspath(os.path.realpath(_log_directory))
    _log_file_path = os.path.join(_log_directory, _log_file)

    # check dirs, paths and files for OSErrors
    if not os.path.exists(_config_directory):
        raise DirectoryNotFoundError("config_directory='{}' for config_file='{}' does not exist".format(
                                     _config_directory, _config_file))
    if not os.path.isfile(_config_file_path):
        raise FileNotFoundError("config_file='{}' in config_directory='{}' does not exist or is not a file".format(
                                _config_file, _config_directory))
    if not _config_file_type == 'py':
        raise InvalidFileType("config has to be a '.py' file but found '{}'".format(_config_file_type))
    if not args.nsave or (not args.nlog and _log_directory.startswith(_output_directory)):
        if not os.path.exists(_output_directory):
            os.mkdir(_output_directory)
        elif not os.path.isdir(_output_directory):
            raise NotADirectoryError("can't create directory for output_dir='{}' since file with same path already "
                                     "exists".format(_output_directory))

    # set up logger
    if not args.nlog:
        if not os.path.exists(_log_directory):
            raise DirectoryNotFoundError("log_directory='{}' for log_file='{}' does not exist".format(_log_directory,
                                                                                                      _log_file))
        logging.basicConfig(format=args.logformat, filename=_log_file_path, level=args.loglevel,
                            datefmt=args.logdateformat)
    logging.info("successfully set up logging")
    logging.debug("called {} with following args={}".format(__file__, str_iterable(vars(args))))

    # load params from configuration file
    params = load_hyperparameters(config_file_path=_config_file_path, run_mode=args.runmode)

    # store dirs, paths and files; clean up args before shipping
    dirs = OrderedDict()
    dirs['config_dir'] = _config_directory
    dirs['config_file_path'] = _config_file_path
    dirs['config_file'] = _config_file
    dirs['config_file_name'] = _config_file_name
    dirs['config_file_type'] = _config_file_type
    dirs['log_dir'] = _log_directory
    dirs['log_file_path'] = _log_file_path
    dirs['log_file'] = _log_file
    dirs['project_dir'] = _project_directory
    dirs['working_dir'] = _working_directory
    dirs['output_dir'] = _output_directory
    del args.outdir, args.config, args.logfile, args.loglevel, args.logdateformat, args.logformat
    logging.debug("initialized with the following file/path/directory variables: dirs={}".format(str_iterable(dirs)))
    # start working
    run(args=args, params=params, dirs=dirs)

    # clean up, say fare well
    logging.info("{} terminated".format(__file__))


if __name__ == '__main__':
    main()
