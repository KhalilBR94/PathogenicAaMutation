# Description
This application is a model inspired by the work BertVS of Li chuan et al. Its purpose is to predict whether the mutation is benign or whether it is pathogenic from mutated protein sequences. For this model we will use two representations.

1) The first representation corresponds to the hydrophobocity index of each AminoAcid.
2) The second representation corresponds to the representation of Amino acids in the form of embeddings.

Embeddings are generated using the biotransformer with ProtBert-bfd as a backend.
In this model, and for the same gene (PTEN or BRCA1 C-terminal domain) we use as input 1 csv file containing the mutated sequences with their labels (0: Benign / 1: pathogen) which you will find in the "data" folder. of the application. These files were obtained and filtered from the ClinVar database.
For the binary classification, we opted for a CNN/BiLSTM architecture with one Conv1D layer, one MaxPooling1D layer and one BiLSTM layer and a fully connected output layer. This classifier will use the concatenation of the two representations as input.

# Tags
Fill the tags.json file in this folder:

- tasks: Probability
         Binary classification
- libraries: bio-transformers  0.1.15
             tensorflow  2.3.4 
             
- embeddings: Rostlab/prot_bert_bfd.
- datasets: BRCA1-C-Terminal-Domain
            PTEN
- device: ["GPU"]: str -> By default on cpu on deepchain, put device to "gpu" to benefit GPU
                          and accelerate the optimization process during score computation.


## libraries and requirements:
- bio-transformers  0.1.15
- tensorflow  >= 2
- python 3.7
- deepchain-apps

## tasks
- Probability
- Binary classification of protein sequences
- Transformers

## embeddings
- Rostlab/prot_bert_bfd
## datasets
- BRCA1-C-Terminal-Domain
- PTEN
## App structure
  +- checkpoint  #saves the "best" model or weights from train.py. Model checkpoint can be loaded from the state saved to continue the evaluation after running evaluation.py
  +- data
  |  +- BRCA1_CTerDom.csv : #BRCA1 dataset
  |  +- PTEN_dataset.csv :  #PTEN dataset
  +- examples : #Templates of applications from Deepchain  	
      |  +-  app_with_checkpoint.py # example: app example with checkpoint
      |  +-  torch_classifier.py # example: show how to train a neural network with pre-trained embeddings
  +- src: #Code to run the application
      |  +- Results : #folder that contains the results PR_curves, ROC_curve and confusion matrix plots in png format
      |  +-   __init__.py  # __init__ file to create python module
      |  +- main_model.py : #Train and evaluation of the model
      |  +- evaluation.py : #Evaluation of the model from the checkpoint generated from train.py (to save time).
      |  +- Desc.md : #Description of the application and how to run it
      |  +- train.py : #Training of the model and generation of the best model checkpoint
      |  +- utilis.py : #a script that includes the functions used by our model
  +- App drawing.png : #Representation of the application
  +- Readme.md : #How to create an application from DeepChain platform

## How to run the application 
 1) Open a terminal and install CUDA and torch stable version: pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
A guide to install Conda is available here: https://docs.conda.io/projects/conda/en/latest/user-guide/install/. 
1) Download folder PathogenicAaMutations.
2) Open a terminal from this folder.
3) Create a new  virtual environnement and install the required packages:
- conda create --name paam-env python=3.7 -y
  conda activate paam-env
- TensorFlow: pip install tensorflow    https://www.tensorflow.org/install
- pip install bio-transformers 0.1.15   https://pypi.org/project/bio-transformers/
- imbalanced learn: pip install -U imbalanced-learn https://imbalanced-learn.org/stable/install.html#getting-started
- deepchain models:  pip install deepchain-apps

4) Run the application as follow: 
evaluation.py [-h help] [-d DATASET] [-i INPUT] [-o OUTPUT_DIR] 

where the arguments are:
 -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        [Required] Choose PTEN or BRCA1 dataset.
  -i INPUT, --input INPUT
                        [Required] File path for PTEN or BRCA1 dataset.
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        [Optional] Folder for output results (ROC_curve and
                        PR_curve). (default="src/Results")
Example 1: Launch the application with BRCA1 dataset: 

python src/evaluation.py -d BRCA1 -i data/BRCA1_CTerDom.csv

Example 2: Launch the application with PTEN dataset: 

python src/evaluation2.py -d PTEN -i data/Pten_dataset.csv
