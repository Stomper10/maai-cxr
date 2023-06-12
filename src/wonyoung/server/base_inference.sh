# Single model inference for 5 outputs

model=densenet121_1005.tflite

python3 classify_cxr_bas.py \
     --model models/$model \
     --labels CheXpert-v1.0/test_project.csv \
     > logs/$model.txt
