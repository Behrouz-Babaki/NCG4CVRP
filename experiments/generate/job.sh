#!/bin/bash
#SBATCH --account=rrg-bengioy-ad
#SBATCH --cpus-per-task=4
#SBATCH --time=02:30:00
#SBATCH --mem=8000M

module load StdEnv/2020 python/3.7
module load gurobi

SCRIPT_DIR=/$SCRATCH/learn2price/experiments/exp21/generate

cd $SLURM_TMPDIR

virtualenv --no-download gen-env
source gen-env/bin/activate

pip install --no-index --upgrade pip
pip install --no-index numpy
pip install --no-index networkx

mkdir gurobipy
cd gurobipy
cp /cvmfs/restricted.computecanada.ca/easybuild/software/2020/Core/gurobi/9.1.0/setup.py .
cp -r /cvmfs/restricted.computecanada.ca/easybuild/software/2020/Core/gurobi/9.1.0/lib .
python setup.py install

cd $SCRIPT_DIR

for i in {1..100}
do
    output_name=$(cat /dev/urandom | tr -cd 'a-f0-9' | head -c 15)
    ./generate_traces.py             \
        --num-customers 21             \
        --r 6                            \
        --output generated_data/${output_name}.pkl
done
