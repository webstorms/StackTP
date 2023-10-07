Part of the accompanying code repository to the "Hierarchical temporal prediction captures motion processing along the visual pathway" paper. This repo contains the code for the modified hierarchical temporal prediction model and the slowness and sparse autoencoder control models.

## Installation
```bash
conda env create -f environment.yml
conda activate stacktp
```

Training data can be downloaded from: https://figshare.com/articles/dataset/Natural_movies/24265498

## Training
See ```scripts/local.py``` to train the models.

## Inspecting models
See ```notebooks/Inspection.ipynb``` to view model RFs and predictions.

## License
Code released under the MIT license.