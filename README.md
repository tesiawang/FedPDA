# FedPDA
The official source codes for "FedPDA: Collaborative Learning to Reduce Online-Adaptation Frequency of Neural Receivers".

## Prepare the environment
To execute the code, a server equipped with at least one functional GPU is required to run `TensorFlow` version 2.11.1. Additionally, `Sionna` version 0.14.0 is necessary. To enable the `sharpfed` package in our codes, both `joblib` and `paramiko` are required.

## License and Citation
FedPDA is Apache-2.0 licensed, as found in the LICENSE file.
If you use our codes in your papers or projects, please cite it as:
```bibtex
@INPROCEEDINGS{6567033,
  author={Shuo Wang, Tianxin Wang, and Xudong Wang},
  booktitle={2025 Proceedings IEEE INFOCOM}, 
  title={FedPDA: Collaborative Learning to Reduce Online-Adaptation Frequency of Neural Receivers}, 
  year={2025},
  volume={},
  number={},
  pages={}
```
Besides, you should cite the reference of the `Sionna` package as mentioned in [Sionna: An Open-Source Library for Next-Generation Physical Layer Research](https://github.com/NVlabs/sionna), and the reference of the [TensorFlow](https://github.com/tensorflow/tensorflow) package.