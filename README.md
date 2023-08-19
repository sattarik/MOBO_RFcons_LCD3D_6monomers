![image](https://github.com/sattarik/MOBO_RFcons_LCD3D_6monomers/assets/54645299/214488c8-d91f-4092-9e11-5283b283909b)# Multi-Objective Bayesian Optimization for 3D Printing of Thermoplastics
This repository is an adaptation of the original code from AutoOED available at: AutoOED GitHub. The code has been tailored by Kianoosh Sattari to accommodate MOBO (Multi-Objective Bayesian Optimization) for optimizing the 3D printing process of thermoplastics using mixed inks containing 6 monomers: R1(HA)	R2(IA)	R3(NVP)	R4(AA)	R5(HEAA)	R6(IBOA)

# Modifications and Objectives
The codebase was adjusted to handle the intricate process of 3D printing thermoplastics with mixed inks of the 6 aforementioned monomers. 
Three constraints were added to handle Sum(R1-R5)<=1, the printability, and Tg range of the suggested monomers. 

The primary objectives of this optimization were to enhance two crucial mechanical properties: Tensile Strength and Toughness, both evaluated under a tensile test. Notably, these objectives are inherently in conflict with each other, and no explicit function exists that establishes a direct relationship between the monomer ratios and the final mechanical properties.

# Gaussian Process and Experiment Setup
Hyperparameters for the Gaussian Process models were established based on 30 initial samples. Alireza Mahjoubnia conducted the 3D printing experiments involving the thermoplastics. The initial 30 samples were optimized in an attempt to identify the optimal ratios that yield the most favorable mechanical properties.

# Utilizing Multi-Objective Bayesian Optimization
The adopted approach is based on the Thompson sampling multi-objective Bayesian optimization method, as detailed in the article "Bradford et al., Journal of Global Optimization volume 71, pages 407–438 (2018)". The overarching objective is to leverage the capabilities of MOBO to fine-tune the ratios of the three monomers, ultimately achieving an optimized balance between Tensile Strength and Toughness in the 3D printed thermoplastics.
