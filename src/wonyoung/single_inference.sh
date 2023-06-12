# Single model inference for 1 output

model=plef_densenet121_7613_test_edgetpu.tflite
name=Pleu

python3 classify_cxr_sin.py \
     --model models/$model \
     --labels CheXpert-v1.0/test_project.csv \
     --name $name \
     > logs/single_${name}_${model}.txt
