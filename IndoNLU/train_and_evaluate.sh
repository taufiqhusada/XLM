#!/bin/bash -l

#$ -P statnlp
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N finetune_smsa # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -V
#$ -m e

python finetune_emot.py
python finetune_smsa.py
