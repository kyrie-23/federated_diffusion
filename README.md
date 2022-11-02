# Federated Diffusion Model with Non-IID Data
## Setup
    pip install labml-nn
    pip install avalanche-lib
## Training on federated setting
    python federated.py
## Training on centralized setting
    python centralized.py

### Set configurations. 
You can override the defaults by passing the values in the function main().

For example:

    experiment.configs(configs, {
        'dataset': 'MNIST',  # 'MNIST','CelebA'
        'image_channels': 1,  # 1,3
        'epochs': 5,  # 5,100
    })

## Sampling
This file can generate samples and interpolations from a trained model (You must run this evaluating after training a model).
    
    python evaluate.py
Specify an uuid from trained logs in the function main().

    run_uuid = "858546e54c6c11ed9b3764d69af93a2f"
