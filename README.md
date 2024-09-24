# CiteBART: Learning to Generate Citation Tokens for Local Citation Recommendation

## Dataset Download Links:
- Link to our preprocessed base datasets: https://drive.google.com/drive/folders/1WlqlTkSj8LwihbrQvBX5F9_0uZAGGhiE?usp=drive_link

- Link to our preprocessed global datasets: https://drive.google.com/drive/folders/1JH34nEXt8_p-0P9A--aQHK4yBXQfJe4v?usp=drive_link

- (Optional) Link to the original datasets: https://drive.google.com/drive/folders/11n4YVHgUPfzetJi-y5voFpmRIjiBM0lQ

## Dependencies:

- `conda create --name "env_name" python=3.8`
- `conda activate "env_name"`
- `pip3 install torch torchvision torchaudio`   # Use an appropriate PyTorch version from https://pytorch.org/get-started/locally/ according to your CUDA version.
- `pip install transformers transformers[torch] datasets`

## Preprocessing the Datasets from Scratch (Optional, Not Recommended):

1. Download the original datasets from the above link. Place them inside the "preprocessing/original_datasets" folder. For example, the two files downloaded for ACL200 dataset should be placed inside a folder named "acl200_original" under the "preprocessing/original_datasets" folder.
2. You can preprocess each dataset for both base and global techniques using their corresponding code in the "preprocessing" folder.
3. Select the code for your chosen dataset. Modify its first few lines to provide the input and output path for the code. Inputs should be the path of two files that belong to the original dataset. Outputs are going be the paths and the names of the preprocessed dataset files.
4. After the chosen preprocessinf code is complete, there should be 4 new files generated inside the given output path. One of these files is the complete version of the preprocessed dataset. Training and evaluation splits of this complete dataset file are also created. Lastly, a complete list of unique author-date citations has been provided in another file as well.

## Steps to reproduce our results:
1. After cloning the project, create the following folders inside the main project folder: "checkpoints" and "models".
2. Create a new conda environment and install the dependencies shown in "Dependencies" section.
3. Download our preprocessed datasets for both base and global technique from the Google Drive links above.
4. (Optional) Alternatively, follow the steps shown in "Preprocessing the Datasets from Scratch" section above to recreate our preprocessed datasets.
5. Place each preprocessed dataset inside its corresponding folder in the "cit_data" folder.
6. To run the code, use the provided scripts inside the "train/scripts" folder. 
7. (Optional) You can modify the parameters inside the scripts beforehand.
8. Directly run the corresponding script for the chosen dataset inside the "train/scripts" folder. 

## Example Run Scenario for Peerread Base:
1. Clone the project, and install the dependencies.
2. Download "peerread_base" dataset from Google Drive.
3. Place the three downloaded files inside "cit_data/peerread_base" folder.
4. Go inside the "train/scripts" folder and open the "run_CiteBART_peerread_base.sh" in order to modify its parameters. For example, you can change "num_epochs" parameter to 1, for a quick validation trial.
5. Run the "run_CiteBART_peerread_base.sh" script to perform training on the peerread base dataset. The results will be printed on the terminal after the training.
