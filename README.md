# 🐝FBEE (Forward Backward Epistemic Exploration)


Code for the paper "Epistemically guided Foward Backward exploration", accepted to RLC 2025.

## Installation

FBEE can be installed by unzipping the code file, accessing the folder and doing the following (I recommend creating a separate virtual environment for it).

```
cd fbee
pyenv install 3.9.7
pyenv virtualenv 3.9.7 fbee_env
pyenv activate fbee_env
pip install -r requirements.txt 
```


## Training 
The main entry point is the `pretrain.py` script.
To train the FBEE agent run
``` 
python url_benchmark/pretrain.py --config-name {config_name}
````
where `{config_name}` can be 
`'pretrain_fb'` + `{'hopper/maze/cheetah/walker/quadruped'}`

for example to train an FBEE agent on the maze task run:

```
python url_benchmark/pretrain.py --config-name pretrain_fb_maze
```

All the configuration files can be found in the `configs` folder and they contain all the parameters that we used to run the experiments in the paper.


As the command-line interface is provided through [`hydra`](https://github.com/facebookresearch/hydra), you can easily modify the default parameters.

For example the following command sets the `f_uncertainty` variable to True, which corresponds to the variant $FBEE^F$ in the paper (default configuration corresponds to run $FBEE^Q$) and evaluates every 10000 environment steps (default is every 100k).

```
python url_benchmark/pretrain.py --config-name pretrain_fb_maze f_uncertainty=True eval_every_frames=10000
```

## Evaluation

Trained models can be easily evaluated by adding the `eval` flag:

```
python url_benchmark/pretrain.py --config-name pretrain_fb_maze eval=True load_model={path_to_trained_model} use_wandb=False save_video=True
```

The agent is evaluated on a set of tasks:
| Domain | Tasks |
|---|---|
| `walker` | `stand`, `walk`, `run`, `flip` |
| `quadruped` | `walk`, `run`, `stand`, `jump` |
|`cheetah` | `walk`, `walk_backward`, `run`, `run_backward`, `flip`, `flip_backward`|
|hopper| `hop`, `stand`, `hop_backward`, `flip`, `flip_backward`


## Temporary manual modification required
At present, training of the Cheetah agent requires a minor manual modification to the dm-control source code. Specifically, users must insert an additional method into the `Physics` class defined in the following file `{path_to_env}/fbee_env/lib/python3.9/site-packages/dm_control/suite/cheetah.py` at line ~52:

```
def angmomentum(self):
    """Returns the angular momentum of torso of the Cheetah about Y axis."""
    return self.named.data.subtree_angmom['torso'][1]
```

# Citation
If you use our work or some of our ideas, please consider citing us :)

```
@article{urpi2025epistemically,
  title={Epistemically-guided forward-backward exploration},
  author={Urp{\'\i}, N{\'u}ria Armengol and Vlastelica, Marin and Martius, Georg and Coros, Stelian},
  journal={arXiv preprint arXiv:2507.05477},
  year={2025}
}
```
## References
This codebase was adapted from the original FB codebase based on the papers:
- A. Touati, J. Rapin, Y. Ollivier, [Does Zero-Shot Reinforcement Learning Exist?](https://arxiv.org/abs/2209.14935)
- A. Touati, Y. Ollivier, [Learning One Representation to Optimize All Rewards (Neurips 2021)](https://arxiv.org/abs/2103.07945)

Their codebase can be found [here](https://github.com/facebookresearch/controllable_agent).

