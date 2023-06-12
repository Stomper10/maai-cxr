# Single model inference for 1 output

model=densenet121_1005.tflite
name=Card

python3 classify_cxr_sin.py \
     --model models/$model \
     --labels CheXpert-v1.0/test_project.csv \
     --name $name \
     > logs/single_${name}_${model}.txt
