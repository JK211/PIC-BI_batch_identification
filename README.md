# PIC-BI: Practical and Intelligent Combinatorial Batch Identification for UAV assisted IoT Networks
***Authors**: [Zhe Ren](https://github.com/JK211), Xinghua Li, Yinbin Miao, [Zhuowen Li](https://github.com/lzwgiter), [Zihao Wang](https://github.com/wangzihao318)* 

***Keywords**: UAV-aided IoT Networks, Combinatorial Framework, Batch Identification, Kalman Filter, Reinforcement Learning* 

## Requirements

- Python 3.7

## Description

The code in the repository implements the combinatorial batch identification framework mentioned in the paper, which specifically contains two algorithms, adaptive estimation algorithm for batch illegal ratio (AEABIR) and combinatorial strategy learning algorithm (CSLA), respectively.

- Estimation_V5.py: The code corresponds to adaptive estimation algorithm for batch illegal ratio (AEABIR), which can be run alone to realize all the functions of the algorithm, and is carried out to give experimental plots.
- Main.py: This code, together with Batch_Signatures_*actions.py and DQN.py, implements combinatorial strategy learning algorithm (CSLA), which can be used to learn combinatorial algorithmic recognition of batch requests based on this code.
- batch_requests_generator.py: This code is noted for generating batch requests containing illegal requests.

The above 3 codes form the main body of our experiments, and the rest of the codes contain comparison schemes, dependencies, cryptographic schemes, and cover free family compositions.

## Installation

*Python Dependencies:*

- pytorch 1.10.1
- gym 0.24.0
- tate_bilinear_pairing 0.6
- statsmodels 0.14.1

We recommend installing strictly the above version and utilizing conda and pip for unified management.

## Usage

- Specify the attacker's attack behavior and evaluate the estimation effect of the AEABIR.

```shell
$ python Estimation_V5.py
```

- Train a model

```shell
$ python Main.py
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
