#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --time=0-12:00
#SBATCH --output=test.out


module load StdEnv/2020 python/3.7
module load gurobi/9.1.0

SCRIPT_DIR=$SCRATCH/learn2price/experiments/exp21/test
DATA_DIR=$SCRATCH/learn2price/experiments/exp21/tars


# Copy the train and validation data
cp $DATA_DIR/test.tar $SLURM_TMPDIR
cd $SLURM_TMPDIR
tar xf test.tar

mkdir data
mv test.zarr data
########################################

# setting up the environment
virtualenv --no-download l2p-env
source l2p-env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index torch
pip install --no-index numpy
pip install --no-index scipy
pip install --no-index networkx
pip install --no-index numcodecs
pip install --no-index zarr

mkdir gurobipy
cd gurobipy
cp /cvmfs/restricted.computecanada.ca/easybuild/software/2020/Core/gurobi/9.1.0/setup.py .
cp -r /cvmfs/restricted.computecanada.ca/easybuild/software/2020/Core/gurobi/9.1.0/lib .
python setup.py install
##########################################

# running the test
cd $SCRIPT_DIR
for model in baseline mlp se gcn gcn-se mse kld
do
    python src/run.py \
        --test-data $SLURM_TMPDIR/data/test.zarr \
        --model-type ${model} \
        --model-file models/${model}.pkl \
        --result-file results/${model}.txt
done
###########################################

