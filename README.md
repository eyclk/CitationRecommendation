# CiteBART: Learning to Generate Citation Tokens for Local Citation Recommendation

## Steps to reproduce our results:
1. After cloning the project, create the following folders inside the main project folder: "checkpoints" and "models".
2. Download our preprocessed datasets for both base and global technique from the Google Drive links below. Alternatively, follow the steps shown in "Preprocessing the Datasets from Scratch" section below to recreate our preprocessed datasets.
3. Place all preprocessed datasets inside the "cit_data" folder.
4. Create a new conda environment and install the dependencies shown in "Dependencies" section.
5. To perform continual pre-training on any one of the datasets, run the corresponding script inside the "train/scripts" folder. You can also modify the parameters inside the scripts beforehand.

## Dataset Download Links:
- Link to our preprocessed base datasets: https://drive.google.com/drive/folders/1WlqlTkSj8LwihbrQvBX5F9_0uZAGGhiE?usp=drive_link

- Link to our preprocessed global datasets: https://drive.google.com/drive/folders/1JH34nEXt8_p-0P9A--aQHK4yBXQfJe4v?usp=drive_link

- Link to the original datasets: https://drive.google.com/drive/folders/11n4YVHgUPfzetJi-y5voFpmRIjiBM0lQ

## Dependencies:

- conda create --name "env_name" python=3.8
- conda activate "env_name"
- pip3 install torch torchvision torchaudio   # Use an appropriate version from https://pytorch.org/get-started/locally/
- pip install transformers transformers[torch] datasets

## Preprocessing the Datasets from Scratch:

- 
