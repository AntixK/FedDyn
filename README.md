# FedDyn (WIP)

<p align="center">
  <img src="https://github.com/AntixK/FedDyn/blob/main/assets/alg.png" width="550" title="FedDyn Algorithm">
</p>

## Getting Started 
Downloading datasets through torchvision can be quite slow or can even [fail](https://stackoverflow.com/a/66816284). Therefore, it is recommended that you download the datasets prior to preparing the data.
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)
- [EMNIST](http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/gzip.zip)

## Results

## Custom Datasets

## ToDo:
- [x] Add MLP 
- [x] Add CNN 
- [ ] Write client side training 
- [x] Server side update 
    - [ ] Update only parameters & not entire state dict?
- [x] Main FedDyn loop 
- [x] Proper communication between server and clients 
- [x] Prepare dataset
    - [x] IID 
    - [x] Non-IID Balanced
    - [x] None-IID Unbalanced
- [x] Download datasets 
- [ ] Metrics (bits transferred, accuracy etc)
- [ ] Parallel training 

### References
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/pdf/1602.05629)
- [Bayesian Nonparametric Federated Learning of Neural Networks](https://arxiv.org/abs/1905.12022)