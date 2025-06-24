# SparseSequentialLearning: Exploring Stochastic Contextual Linear Bandit (SCLB) and Feature Selection Combinations for Fixed Reduced Dimensions

This repository contains the code implemented for the course CSE3000 in BSc Computer Science and Engineering, TU Delft.
https://github.com/TU-Delft-CSE/Research-Project
## Project Structure

The project consists of several key components:
- Synthetic Data Generator
- SCLB-FS Algorithms
- Simulation Framework
- Configuration System

## Getting Started

### Data Generation
To obtain a data file for your experiments:
1. Navigate to the synthetic data generator
2. Adjust the parameters as needed for your use case
3. Run the generator to create your dataset

### Running Experiments
To execute a simulation:
1. Ensure you have a configuration file for your experiment
   - Use or modify an existing config file
2. In the execution file main.py:
   - Specify the relative path to a config file
   - Specify the relative path to a data file
   - Select the appropriate simulator
3. Run the main execution file

## References
- P. Auer, N. Cesa-Bianchi, and P. Fischer. Finite-Time Analysis of the Multiarmed Bandit Problem. Machine Learning, 47(2–3):235–256, 2002
- Tor Lattimore and Csaba Szepesvari. Bandit Algorithms. Cambridge University Press, Cambridge, UK, 2020.
- Botao Hao, Tor Lattimore, and Mengdi Wang. High-dimensional sparse linear bandits. arXiv preprint arXiv:2011.04020, 2021.
- Yasin Abbasi-Yadkori, David Pal, and Csaba Szepesvari. Improved algorithms for linear stochastic bandits. In Advances in Neural Information Processing Systems, volume 24, pages 2312–2320, 2011.
- Wei Chu, Lihong Li, Lev Reyzin, and Robert Schapire. Contextual bandits with linear payoff functions. In Proceedings of the 14th International Conference on Artificial Intelligence and Statistics (AISTATS), 2011.
- Shilpa Agarwal and Navin Goyal. Thompson sampling for contextual bandits with linear payoffs. In Proceedings of the 30th International Conference on Machine Learning (ICML), 2013.