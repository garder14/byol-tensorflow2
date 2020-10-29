# BYOL - TensorFlow 2

This is an unofficial implementation of ["Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning"](https://arxiv.org/pdf/2006.07733.pdf) (BYOL) for self-supervised representation learning on the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

<img src="https://i.imgur.com/PnwlSmw.png" width="90%">

## Results

The linear evaluation accuracy of a ResNet-18 encoder pretrained for 100 and 200 epochs is shown below.

|           | 100 epochs | 200 epochs |
|:---------:|:----------:|:----------:|
| **ResNet-18** |   82.03%   |   86.92%   |

## Software installation

Clone this repository:

```bash
git clone https://github.com/garder14/byol-tensorflow2.git
cd byol-tensorflow2/
```

Install the dependencies:

```bash
conda create -n tf-gpu tensorflow-gpu cudatoolkit=10.1
conda activate tf-gpu
```

## Pretraining (representation learning)
 
To pretrain a ResNet-18 base encoder for 200 epochs (weights saved every 100 epochs), try the following command:

```bash
python pretraining.py --encoder resnet18 --num_epochs 200 --batch_size 512
```

It takes around 12 hours on a Tesla P100 GPU. 

## Linear evaluation

To evaluate the quality of representations extracted by a certain base encoder, try the following command:

```bash
python linearevaluation.py --encoder resnet18 --encoder_weights f_online_200.h5
```

It takes around 1 hour on a Tesla P100 GPU.

## References

* [Jean-Bastien Grill et al., "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning", 2020](https://arxiv.org/pdf/2006.07733.pdf)
