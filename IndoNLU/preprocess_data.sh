#!/bin/bash -l

#$ -P statnlp
#$ -l h_rt=8:00:00   # Specify the hard time limit for the job
#$ -N preprocess_data # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -V
#$ -m e

python preprocess_data.py
