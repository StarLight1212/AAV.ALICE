# Mapping AAV capsid sequences to functions through in silico function-guided evolution


####  Please access and download the relevant data from the following Google Drive link:

  > [ALICE Data Download](https://drive.google.com/drive/folders/1vHAZe2p-mq58dbaS-pdC5AtfR12eYMDX?usp=sharing)

Upon downloading, ensure that the acquired "input", "output", and "checkpoint" directories are placed at the same hierarchical level as the "step1_pretrain", "step2_semantic_tuning", and other directories within the ALICE project folder structure. This directory organization is crucial for maintaining the correct relative paths and enabling seamless execution of the ALICE pipeline.

## Required python packages
- python 3.8-3.10
- pytorch torch-deploy-1.8 (GPU 3090, CUDA 11.2)
- torch = 1.10.1+cu113
- torchvision = 0.11.2+cu113
- rdkit (2021.09.4)
- bio = 1.3.3
- biopython = 1.79
- sklearn = 1.0.2
- numpy = 1.21.5
- pandas = 0.24.2
- scipy = 1.7.1
- openbabel = 3.1.1
- argparse



**News:**    
```yaml
The improvement directions of FeatNN mainly include the following 5 points:  
#### This is a repo to train and deploy the ALICE System.

#### Step 1 Pretrain Module used for 

#### Step 2 Pretrain Module used for 

#### Step 3 Pretrain Module used for 

#### Step 4 Pretrain Module used for 

#### Step 5 Pretrain Module used for 

#### Evaluation (directory: evaluation) used to predict (train and inference) the binding ability of ly6a, ly6c1 and production fitness of AAV capsid sequences. 
