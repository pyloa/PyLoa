## PyLoa - Learning on-line Algorithms with Python

`pyloa` is a research repository for analyzing the performance of classic on-line algorithms vs. modern Machine
Learning, specifically Reinforcement Learning, approaches. PyLoa ships with an implementation of two commonly known 
on-line problems as `environments`:

* `(k,n)-paging-problem` with a `cache_size k` and `n pages` for a sequence of page-requests
* `(k,n)-coloring-problem` with `k colors` for a graph with `n vertices` 

PyLoa allows for `agents` to be 

* trained on such `enviroments` (problem definitions) that require on-line solutions, 
* evaluated against commonly used heuristics or any state-of-the-art algorithm,
* exploited (extrapolation of a *potentially* worst case problem instances) to determine a solution's [competitve ratio](https://en.wikipedia.org/wiki/Competitive_analysis_(online_algorithm)).

---

#### Dependencies

`pyloa` is developed for Python 3.5+ and has the following package dependencies:

```python
matplotlib==3.0.3  
scipy==1.2.1  
tensorflow==1.13.1  
tqdm==4.31.1  
numpy==1.16.2
```

---
#### Installation

We recommend using `pyloa` within a [virtual environment](https://docs.python.org/3.5/library/venv.html):

    mkdir myproject
    cd myproject
    python3 -m venv virtualenv/
    source virtualenv/bin/activate
 
Update [`pip`](https://pypi.org/project/pip/) and [`setuptools`](https://pypi.org/project/setuptools/) before continuing:

    pip install --upgrade pip setuptools

Afterwards you can install `pyloa` either from its latest [PyPI stable](https://google.com) release

    pip install pyloa
    
**or** from its latest [development release](https://github.com/pyloa/PyLoa/tree/master/pyloa) on GitHub

    pip install git+https://github.com/pyloa/PyLoa.git

---
#### General Usage

`pyloa` can be used in three different ways to analyze an on-line problem; each depicted via a so called runmode 
(`train`, `eval`, `gen`). Any runemode can be invoked via its positional argument and requires a python-configuration-file.

    pyloa {train,gen,eval} --config path/to/hyperparams.py

hyperparams depicts the setting of the experiment at hand; it must hold a dictionary named `params`, which moreover **must 
contain** dictionaries for the keys `instance`, `environment` and `agent`. 

* `params["ìnstance"]`: Must define a configuration of a subclass implementation of [`pyloa.instance.InstanceGenerator`](https://github.com/pyloa/PyLoa/blob/master/pyloa/instance/instancegenerator.py), 
which generates problem instances for the domain. As an example, for the `(k,n)-paging-problem` a simple generator could 
randomly generate a sequence of requests of length `sequence_size`, whereas each request is within [1, n].   
* `params["agent"]`: Must define a configuration of a subclass implementation of [`pyloa.agent.Agent`](https://github.com/pyloa/PyLoa/blob/master/pyloa/agent/agent.py), 
which observes a state `s` of its environment, acts with action `a` accordingly, receives reward `r` and observes 
transitioned state `s'`. For toy problem instances a simple Q-learning table implementation would suffice. 
* `params["environment"]`: Must define a configuration of a subclass implementation of [`pyloa.environment.Environment`](https://github.com/pyloa/PyLoa/blob/master/pyloa/environment/environment.py), 
which consumes a problem instance and let's the agent *play* until it terminates. An `environment` constitutes as a problem
definition.

A minimal example for learning the `(5,6)-paging-problem` with a [`QTableAgent`](https://github.com/pyloa/PyLoa/blob/master/pyloa/agent/qtable.py) on a 
[`PagingEnvironment`](https://github.com/pyloa/PyLoa/blob/master/pyloa/environment/environment.py#L105) can be invoked with

    pyloa train --config hyperparams.py

and the hyperparams.py as following:
    
```python
from pyloa.instance import RandomSequenceGenerator
from pyloa.environment import DefaultPagingEnvironment
from pyloa.agent import QTableAgent

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
    },
    'environment': {
        'type': DefaultPagingEnvironment,
        'sequence_size': sequence_size,
        'cache_size': 5,
        'num_pages': max_page - min_page + 1,
    },
    'agent': {
        'type': QTableAgent,
        'discount_factor': 0.55,
        'learning_rate': 0.001,
        'epsilon': 0.0,
        'epsilon_delta': 13 / (episodes * 10),
        'epsilon_max': 0.99,
        'save_file': "/home/me/models/",
    },
}
```

This example is defined in [examples/0_train_qtable_paging/hyperparams.py](https://github.com/pyloa/PyLoa/tree/master/examples/0_train_qtable_paging) and can be run with

    pyloa train --config examples/0_train_qtable_paging/hyperparams.py
    
The resulting run can be seen [here](http://google.com). In total there are five toy examples, which can be run on any system, 
defined in the examples directory.

#### Runmodes

PyLoa has three different runmodes: `train` ,`eval` and `gen`. There are slight adaptions to be made for the configuration file 
depending on the selected runmode; we encourage checking the examples for reference (on a site note: hyperparams are loaded and validated
in [pyloa.utils.load](https://github.com/pyloa/PyLoa/blob/master/pyloa/utils/load.py#L13)). Semantically the three different runmodes stand for: 

* train: An `RLAgent` will be trained for `episode`-many instances, generated by an `InstanceGenerator`, on his `environment`. 
Every `checkpoint_step`-many instances a checkpoint of `RLAgent` will be saved.  
* eval: All trained `RLagents` nested within `root_dir` will be evaluated on `episode`-many instances, generated by an `InstanceGenerator`.
Additionally non-trainable agents may be defined and evaluated alongside.
* gen: Currently **only applicable** for the `(k,n)-paging-problem`. A genetic algorithm empirically determines a `PagingAgent`'s 
(approximate) competitive ratio. 

Each runmode will create TFEvent-files for TensorBoard in its experiment's output directory. 

