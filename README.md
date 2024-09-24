# CiteBART: Learning to Generate Citation Tokens for Local Citation Recommendation

## Steps to reproduce our results:
1. After cloning the project, create the following folders inside the main project folder: "checkpoints" and "models".
2. Download our preprocessed datasets for both base and global technique from the Google Drive links below.
3. (Optional) Alternatively, follow the steps shown in "Preprocessing the Datasets from Scratch" section below to recreate our preprocessed datasets.
4. Place all preprocessed datasets inside the "cit_data" folder.
5. Create a new conda environment and install the dependencies shown in "Dependencies" section.
6. To perform continual pre-training on any one of the datasets, run the corresponding script inside the "train/scripts" folder. You can also modify the parameters inside the scripts beforehand.

## Dataset Download Links:
- Link to our preprocessed base datasets: https://drive.google.com/drive/folders/1WlqlTkSj8LwihbrQvBX5F9_0uZAGGhiE?usp=drive_link

- Link to our preprocessed global datasets: https://drive.google.com/drive/folders/1JH34nEXt8_p-0P9A--aQHK4yBXQfJe4v?usp=drive_link

- (Optional) Link to the original datasets: https://drive.google.com/drive/folders/11n4YVHgUPfzetJi-y5voFpmRIjiBM0lQ

## Dependencies:

- `conda create --name "env_name" python=3.8`
- `conda activate "env_name"`
- `pip3 install torch torchvision torchaudio`   # Use an appropriate PyTorch version from https://pytorch.org/get-started/locally/ according to your CUDA version.
- `pip install transformers transformers[torch] datasets`

## Preprocessing the Datasets from Scratch (Optional):

1. Download the original datasets from the above link. Place them inside the "preprocessing/original_datasets" folder. For example, the two files downloaded for ACL200 dataset should be placed inside a folder named "acl200_original" under the "preprocessing/original_datasets" folder.
2. You can preprocess each dataset for both base and global techniques using their corresponding code in the "preprocessing" folder.
3. Select the code for your chosen dataset. Modify its first few lines to provide the input and output path for the code. Inputs should be the path of two files that belong to the original dataset. Outputs are going be the paths and the names of the preprocessed dataset files.
4. After the chosen preprocessinf code is complete, there should be 4 new files generated inside the given output path. One of these files is the complete version of the preprocessed dataset. Training and evaluation splits of this complete dataset file are also created. Lastly, a complete list of unique author-date citations has been provided in another file as well.
