#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --time=02:30:00
#SBATCH --mem=8000M

module load python
python move-files.py

cd partitioned-data
tar cvf data.tar data/
mv data.tar ../../tars
