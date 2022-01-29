Repeating the experiments of the ML4OR paper with a better setup. 
The training experiments are copied from `exp18`. 

## directory structure
- generate: generating the instances
- create-zarr: creating the zarr files `train.zarr`, `valid.zarr`, and `test.zarr`, compressing them, and putting them in the `tars` file
- train: training 7 models
- test: testing the models in terms of cross-entropy loss and top-k accuracy
- evaluate: comparing policies when used in CG
