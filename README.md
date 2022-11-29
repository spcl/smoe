# Spatial Mixture-of-Experts

This is the official repository for [Spatial Mixture-of-Experts](https://arxiv.org/abs/2211.13491).
A Spatial Mixture-of-Experts (SMoE) layer learns the underlying location dependence of a dataset.

If you find this useful, please cite:

```
@inproceedings{
  title={Spatial Mixture-of-Experts},
  author={Nikoli Dryden and Torsten Hoefler},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2022}
}
```

This currently contains code for our heat diffusion experiments and several baselines.
Code for additional experiments (WeatherBench, ENS-10, etc.) will be released soon.

## Location-Dependent Heat Diffusion

Please see Section 3.1 of the paper for full details.

### Generating Heat Diffusion Data

Note: Due to differences in random number generation, we do not guarantee an identical dataset will be generated even with the same random seed.

We include the region map we used (`heat/mask.npy`). A different region map can be generated using the `generate_mask.py` script (see documentation therein).

The heat diffusion data is generated using the `generate_data.py` script. (Run it with `--help` for full options.) To replicate our dataset, run as follows:
```
# Training data:
python generate_data.py --diffusivity 0.0025 0.025 0.25 --num-runs 1000 train
# Validation data:
python generate_data.py --diffusivity 0.0025 0.025 0.25 --num-runs 20 --seed 546981 val
# Test data:
python generate_data.py --diffusivity 0.0025 0.025 0.25 --num-runs 20 --seed 865124 test
# Move data:
mv {train.npy,val.npy,test.npy} heat
```

### Running Experiments

Experiments can be run using the `train.py` script (`--help` gives full options).

A basic run of an SMoE using our configuration is as follows:
```
python train.py --output-dir out --data-path heat --data-no-norm --job-id smoe --fp16 --epochs 150 --schedule plateau --plateau-epochs 15 --early-stop 30 --optimizer adam --lr 0.001 --loss mse --initialization default --metric mse mask-mse prcntclose mask-prcntclose --mask heat/mask.npy --prcntclose 0.01 --opt-metric prcntclose1.0 --opt-metric-max --prcntclose-tol-scale 0.01 0.1 1 --save-on-best --stop-on-metric-level 100 --model smoe --last-layer-experts 3 --gate-type latent --unweighted-smoe --rc-loss --dampen-expert-error --routing-error-quantile 0.3
```

Additional models can be selected using the `--model` argument. For example, to train a CNN:
```
python train.py --output-dir out --data-path heat --data-no-norm --job-id smoe --fp16 --epochs 150 --schedule plateau --plateau-epochs 15 --early-stop 30 --optimizer adam --lr 0.001 --loss mse --initialization default --metric mse mask-mse prcntclose mask-prcntclose --mask heat/mask.npy --prcntclose 0.01 --opt-metric prcntclose1.0 --opt-metric-max --prcntclose-tol-scale 0.01 0.1 1 --save-on-best --stop-on-metric-level 100 --model cnn --layers conv conv --conv-filters 4
```

