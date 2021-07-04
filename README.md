# User guide 
This README will explain how to set-up and run the implementation code for Conditional Generation of Aerial Images for Imbalanced Learning using Generative Adversarial Networks.

# Installation
The required modules can be installed via: <br>

```
pip install -r requirements.txt
```

# Quick start
The following code can be run to execute the training of the GAN. You can choose your own hyperparameters with the argparser. An example for a model with name `my_first_gan` and learning rate 0.0001 is shown below. If parameter settings are not chosen, the default parameters will be used. <br>
```
python train.py --model_name 'my_first_gan' --lr 0.0001
```
The file automatically creates the directories for the checkpoints, loss plots and sample of generates images. Make sure to choose the right directory to store the loss plots and grid of images `result_path`, the directory where the data is stores `dataset_path` and the path to the csv file with the cluster labels `path_to_csv` in line 79 to 81 in `train.py`. <br>

# Data
In order to access the data, please send a request to:
- itzelbelderbos@hotmail.com
- tjadejong@thinkpractice.nl
- mirela.popa@maastrichtuniversity.nl

# Citation
Please use the following BibTeX reference when citing this code:
```
@article{belderbos2021conditional,
  title={Conditional generation of aerial Images for imbalanced learning using generative adversarial networks},
  author={Belderbos, Itzel and De Jong, Tim and Popa, Mirela},
  journal={arXiv preprint arXiv: to do},
  year={2021}
}
```


