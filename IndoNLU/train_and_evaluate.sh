#!/bin/bash -l

#$ -P statnlp
#$ -l h_rt=99:00:00   # Specify the hard time limit for the job
#$ -N train_and_evaluate # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -V
#$ -m e

python finetune_wrete.py
python finetune_emot.py
python finetune_smsa.py
