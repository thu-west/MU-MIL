Instance Explainable Multi-Instance Learning for ROI of Various Data
================================================

by XuZhao (<zhaoxu18@mails.tsinghua.edu.cn>) and ZihaoWang (<wzh17@mails.tsinghua.edu.cn>)

Overview
--------

PyTorch implementation of our paper "Instance Explainable Multi-Instance Learning for ROI of Various Data" accepted by DASFAA 2020.


Dependencies
------------

Install Pytorch 1.3.0, using pip or conda, should resolve all dependencies.
Install the package imgaug by pip.
Tested with Python 3.6, but should work with 3.x as well.
Tested on GPU.

Dataset
-------

You can download the datasets we introduced in our paper from following links:
* [MUSK1, MUSK2, Fox, Elephant, Tiger](http://www.cs.columbia.edu/~andrews/mil/data/MIL-Data-2002-Musk-Corel-Trec9-MATLAB.tgz)
* [MNIST](http://yann.lecun.com/exdb/mnist/), Notice that the dataset MNISTBag for MIL task is generated from MNIST.Code for generation could be found in mnist_dataloader.py.
* [Colon Cancer dataset](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/crchistolabelednucleihe/crchistophenotypes_2016_04_28.zip)


How to Use
----------
`cc_src/*`: Evaluate our model specifically on the colon cancer dataset.

`classic_src/*`: Evaluate our model specifically on the MUSK1, MUSK2, Fox, Elephant, Tiger datasets.

`mnist_src/*`: Evaluate our model specifically on the MNIST-Bags dataset. 
__NOTE__: The codes will automatically download the original MNIST dataset and generate the MNIST-Bags dataset. It can handle any bag length without the dataset becoming unbalanced. It is most probably not the most efficient way to create the bags. Furthermore it is only test for the case that the target number is ‘9’.

`src/*`: We implement the uniform interface for various datasets introduced above. You can specify the dataset by setting hte parameter `--dataset`and simply run the main.py as follows:
<br>
<br>
`python main.py --dataset cc`
<br>
<br>
A part of settable parameters are listed as follows:

Parameter | Options | Usage
--------- | ------- | -----
--dataset | [cc, bc, mb, musk1, musk2, fox, tiger, elephant] | Specify the dataset for evaluation
--attention | [att, mu, datt, mhatt] | Specify the attention type 
--epochs | | Specify the epoch nums for training
--lr | | Specify the learning rate
--dim | | Sepcify the dimension for the attention

Some other settable parameters could be found in the `src/main.py` file.
