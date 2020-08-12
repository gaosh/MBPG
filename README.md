# Momentum-Based Policy Gradient Methods
Authors: Feihu Huang, Shangqian Gao, Pei Jian and Huang Heng

PyTorch Implementation of [Momentum-Based Policy Gradient Methods](https://arxiv.org/pdf/2007.06680.pdf) (ICML 2020).

Code uploaded.
# Requirements
pytorch 1.1.0  
[garage](https://github.com/rlworkgroup/garage) 2019.10.1\
[mujuco](http://www.mujoco.org/)  
[gym](https://github.com/openai/gym)  
If you do not install mujuco, then only CartPole environment is available.
# Usage
To run IS-MBPG
```
python MBPG_test.py --env CartPole
```
To run IS-MBPG*
```
python MBPG_test.py --env CartPole --IS_MBPG_star True
```
To run HA-MBPG
```
python MBPG_HA_test.py --env CartPole
```
To run different environments change --env to one of the followings: "CartPole", "Walker", "Hopper" or "HalfCheetah". If you want to use our algorithms on different enviroment, you need to implement it by yourself, but it should be pretty straightforward.
# Citation
```
@InProceedings{huang2020accelerated,
  author    = {Huang, Feihu and Gao, Shangqian and Pei, Jian and Huang, Heng},
  title     = {Momentum-Based Policy Gradient Methods},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning},
  year      = {2020},}
```
