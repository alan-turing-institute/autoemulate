# Training an emulator

Emulators are drop-in replacements for complex simulations, and can be orders of magnitude faster. Any model could be an emulator in principle, from a simple linear regression to Gaussian Processes to Neural Networks. Evaluating all these models requires time and machine learning expertise. AutoEmulate is designed to automate the process of finding a good emulator model for a simulation.

In the background, `AutoEmulate` does input processing, cross-validation, hyperparameter optimization and model selection. It’s different from typical AutoML packages, as the choice of models and hyperparameter search spaces is optimised for typical emulation problems.