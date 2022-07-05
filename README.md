# StageCNN
paper: Simple and effective multi-scale fusion strategy

# Describe
Different datasets and models can be used for training, provided that the dataset needs to be downloaded in a specific location.

# Train
python --model {Model_Name} --dataset {Dataset_Name} --epochs 30 --device 0

# Use Stage
python --model {Model_Name} --dataset {Dataset_Name} --epochs 30 --device 0 --reweight

# Use Stage with weight redistribution
python --model {Model_Name} --dataset {Dataset_Name} --epochs 30 --device 0 --lrweight
