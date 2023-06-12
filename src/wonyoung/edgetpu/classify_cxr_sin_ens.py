# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Example using TF Lite to classify a given image using an Edge TPU.

   To run this code, you must attach an Edge TPU attached to the host and
   install the Edge TPU runtime (`libedgetpu.so`) and `tflite_runtime`. For
   device setup instructions, see g.co/coral/setup.

   Example usage (use `install_requirements.sh` to get these files):
   ```
   python3 classify_cxr.py \
     --model models/DenseNet201_edgetpu.tflite  \
     --labels CheXpert-v1.0/test_project.csv \
   ```
"""
import os
import sys
import csv
import uuid
import json
import time
import argparse
import numpy as np
from PIL import Image
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, f1_score, accuracy_score
from config_sin import Config

import classify
import tflite_runtime.interpreter as tflite
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(model_file):
  model_file, *device = model_file.split('@')
  return tflite.Interpreter(
      model_path=model_file,
      experimental_delegates=[
          tflite.load_delegate(EDGETPU_SHARED_LIB,
                               {'device': device[0]} if device else {})
      ])

def CXR_test_label(test_label_path):
  labels = dict()
  with open(test_label_path, "r") as f:
    csv_reader = csv.reader(f)
    next(csv_reader, None) # skip the header
    for line in csv_reader:
        labels[line[0]] = [[float(line[13]), 
                            float(line[7]), 
                            float(line[11]), 
                            float(line[10]), 
                            float(line[15])]]
  return labels

def softmax(x):
  f_x = np.exp(x) / np.sum(np.exp(x))
  return f_x

def main():
  config = Config()
  stdoutOrigin = sys.stdout 
  sys.stdout = open(f"./logs/{uuid.uuid4().hex}.txt", "w")
  print(vars(config))
  
  label_idx = {"Atel": 0, "Card": 1, "Cons": 2, "Edem": 3, "Pleu": 4}
  target_idx = label_idx[config.name]
  labels_dict = CXR_test_label(config.labels)

  for model in config.model_list:
    print("Model name: ", model)
    interpreter = make_interpreter("./models/" + model)
    interpreter.allocate_tensors()

    inference_time_list = []
    count = 0
    for key in labels_dict:
      count += 1
      image = Image.open(key).convert("L") #.resize((320, 320), Image.ANTIALIAS)
      image = np.expand_dims(image, axis=-1)
      classify.set_input(interpreter, image)

      print(f"----INFERENCING {count:3d} : {key[20:]}----")
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      inference_time_list.append(inference_time * 1000)

      data_pred_cat = classify.output_tensor(interpreter)
      labels_dict[key].append([data_pred_cat[f"{i}"][1] if len(data_pred_cat[f"{i}"]) == 2 else softmax(data_pred_cat[f"{i}"][:2])[1] for i in range(len(data_pred_cat))])
      print(labels_dict[key])

    print(f"Avg. {sum(inference_time_list) / len(labels_dict):.1f}ms per image.")
    print("End inferencing:", model)

  labels = np.array([value[0] for value in labels_dict.values()])
  output = np.array([value[1:] for value in labels_dict.values()]).mean(axis=1)
  output_bin = (output > 0.5).astype(int)

  print("[ Ensemble Evaluation Results ]")
  print("Model names:", config.model_list)
  print("# of Models:", len(config.model_list))
  print("      AUROC /  F1   /  Acc")
  fpr, tpr, _ = roc_curve(labels[:, target_idx], output[:, 0])
  roc_auc = metrics.auc(fpr, tpr)
  f1 = f1_score(labels[:, target_idx], output_bin[:, 0])
  acc = accuracy_score(labels[:, target_idx], output_bin[:, 0])
  print(f"{config.name}: {roc_auc:.3f} / {f1:.3f} / {acc:.3f}")

  sys.stdout.close()
  sys.stdout = stdoutOrigin

if __name__ == '__main__':
  main()
