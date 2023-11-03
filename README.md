[![Python 3.7](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/) [![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE) 

## Multi-Objective Bayesian Optimization with physics-informed descriptors for 3D Printing of Thermoplastics

### Overview
Welcome to our Multi-Objective Bayesian Optimization (MOBO) project for enhancing the 3D printing of thermoplastics. This repository is an adaptation of the original code from AutoOED, available at AutoOED GitHub, and has been tailored by Kianoosh Sattari to accommodate MOBO for optimizing the 3D printing process of thermoplastics using mixed inks containing six monomers:
R1 (HA)
R2 (IA)
R3 (NVP)
R4 (AA)
R5 (HEAA)
R6 (IBOA)

Our research results are publicly available on ChemRxiv via the following link. Additionally, we are under review for publication in Nature Communications.
Modifications and Objectives

Our codebase has been adjusted to address the complex process of 3D printing thermoplastics using mixed inks comprising the six aforementioned monomers. We have introduced three key constraints:

1) Constraint on Monomer Ratios: Ensures that the sum of monomer ratios (R1 to R5) does not exceed 1. The last ratio for experimental evaluation, R6, is calculated as 1 - ΣRi.
2) Printability Constraint: We employ a Random Forest (RF) classifier with inputs of structural and physics-informed descriptors of the input monomers, producing a binary (0/1) output to determine printability.
3) Tg (Glass Transition Temperature) Range Constraint: This constraint employs a RF classifier with similar inputs to evaluate the Tg range of the suggested monomers. Class 1 indicates that the Tg is within the acceptable range of [10-60] °C.

Our primary optimization objectives revolve around improving two critical mechanical properties: Tensile Strength and Toughness, both assessed via tensile tests. Notably, these objectives inherently conflict with each other, and no explicit function can establish a direct relationship between the monomer ratios and the final mechanical properties.
Gaussian Process and Experiment Setup

Hyperparameters for the Gaussian Process models were established based on 30 initial samples. The 3D printing experiments, involving the thermoplastics, were conducted by Alireza Mahjoubnia. The initial 30 samples were optimized to identify the optimal ratios that yield the most favorable mechanical properties.
Utilizing Multi-Objective Bayesian Optimization

Our approach is grounded in the Thompson sampling multi-objective Bayesian optimization method, as detailed in the article "Bradford et al., Journal of Global Optimization volume 71, pages 407–438 (2018)." Our overarching goal is to leverage the capabilities of MOBO to fine-tune the ratios of the six monomers, ultimately achieving an optimized balance between Tensile Strength and Toughness in the 3D printed thermoplastics.


## Installation
### How-to
To install, just clone our repo:

```
git clone https://github.com/aspuru-guzik-group/ORGANIC.git
```

And, it is done!

### Requirements

- tensorflow==1.2
- future==0.16.0
- rdkit
- keras
- numpy
- scipy
- pandas
- tqdm
- pymatgen

## How to use

ORGANIC has been carefully designed to be simple, while still allowing full customization of every parameter in the models. Have a look at a minimal example of our code:

```python
model = ORGANIC('OPVs')                     # Loads a ORGANIC with name 'OPVs'
model.load_training_set('opv.smi')          # Loads the training set (molecules encoded as SMILES)
model.set_training_program(['PCE'], [50])   # Sets the training program as 50 epochs with the PCE metric
model.load_metrics()                        # Loads all the metrics
model.train()  
