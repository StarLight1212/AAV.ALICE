# ALICE System
## Mapping AAV capsid sequences to functions through in silico function-guided evolution

✨✨✨ALICE offers a novel design paradigm for designing protein complexes: AAV capsids, helping researchers to explore the potential AAV capsid space more accurately. It provides new ideas for AI to design protein complexes under limited data conditions. More importantly, it accelerates the development of new serotype capsids in gene therapy.


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
- jupyter-client==7.4.9
- jupyter-core==4.12.0
- jupyterlab-pygments==0.2.2
- kiwisolver==1.4.4
- scikit-learn==1.0.2
- scipy==1.7.3
- tokenizers==0.13.3
- traitlets==5.9.0
- transformers==4.30.2
- xgboost==1.6.2
- zipp==3.15.0
- huggingface-hub==0.14.1
- bleach==6.0.0
- safetensors==0.3.1

(for details see the file of 'environment.yml')

**News:**    
```yaml
Key Innovations
```
ALICE (AI-driven Ligand-Informed Capsid Engineering) represents a groundbreaking approach to AAV capsid engineering, offering several key innovations:
- [x] 1. Direct in silico design of multifunctional capsids: ALICE achieves direct computational engineering of AAV capsids with multiple functions, independent of in vivo selection processes. While AI has been used to construct capsid libraries with enhanced fitness or diversity, previous approaches still relied on unpredictable in vivo selection. ALICE pioneers the direct in silico design of multifunctional capsids, potentially accelerating AAV-based gene therapy development more efficiently.
- [x] 2. Function-guided evolutionary approach: ALICE overcomes the challenge of fusing multiple functions into a single sequence, a limitation often faced by conventional generative language models. By employing a novel function-guided evolutionary approach, ALICE successfully integrates multiple functions into a single capsid sequence, bridging the gap in AI-driven design of multifunctional AAV capsids and providing insights for other multifunctional protein complexes.
- [x] 3. Interpretable computational mapping of capsid sequence to function: ALICE demonstrates an interpretable approach to mapping capsid sequences to multiple functions. By illustrating multi-level relationships between capsid sequences and functions, ALICE enables knowledge-driven AAV design. This approach offers insights into the interpretability, controllability, and safety of viral capsid protein engineering, addressing crucial bio-security concerns in the field.

------

## Usage 
### Please access and download the relevant data from the following Google Drive link:

  > [ALICE Data Download](https://drive.google.com/drive/folders/1vHAZe2p-mq58dbaS-pdC5AtfR12eYMDX?usp=sharing)

    https://drive.google.com/drive/folders/1vHAZe2p-mq58dbaS-pdC5AtfR12eYMDX?usp=sharing

Upon downloading, ensure that the acquired "input", "output", and "checkpoint" directories are placed at the same hierarchical level as the "step1_pretrain", "step2_semantic_tuning", and other directories within the ALICE project folder structure. This directory organization is crucial for maintaining the correct relative paths and enabling seamless execution of the ALICE pipeline.


#### Step 1 Pretrain Module (directory: step1_pretrain)
used for train and inference the RoBeERTa pretrain Module with on the uniprot datasets.

#### Step 2 Semantic Tuning Module (step2_semantic_tuning)
used for train and inference the RoBERTa-based SeqGAN module on the AAV capsid sequences .

#### Step 3 Ranking and Filtrition Process (step3_rankfiltpro)
used for rank and filter the capsid sequences generated by RoBERTa-based SeqGAN with high binding ability to the targets of Ly6a and Ly6c1.

#### Step 4 Function-guided Evolution (FE) (step4_function_guided_evolution) 
used for evolving and fusing the multiple function into the capsid sequences with the contrastive and heuristic strategy. 

#### Step 5 Elite Ascendancy (step5_elite_ascendancy)
used for select the best capsid sequences with multiple functions for in vivo test.


#### Evaluation (directory: evaluation)
This module is applied to predict (train and inference) the binding ability of Ly6a, Ly6c1, and production fitness of AAV capsid sequences. It provides crucial feedback for the entire pipeline.
Key components:
- [x] Model architectures for binding and fitness prediction
- [x] Training pipelines for each prediction task
- [x] Inference scripts for applying trained models

Each module in this pipeline builds upon the previous one, creating a comprehensive framework for designing and optimizing AAV capsid sequences with desired functional properties. The FE module serves as a critical component for fusing and generating the capsid sequences with multiple functions throughout the process.

## License  
This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, and scientific publications. Permission is granted to use ALICE given that you agree to my licensing terms. Regarding the request for commercial use, please contact us via email to help you obtain the authorization letter.  


