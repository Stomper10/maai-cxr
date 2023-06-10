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
import csv
import time
import argparse
import numpy as np
from PIL import Image
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve

import classify
#import tflite_runtime.interpreter as tflite
#import platform

import tensorflow as tf

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

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '-m', '--model', required=True, help='File path of .tflite file.')
  parser.add_argument(
      '-l', '--labels', help='File path of labels file.')
  parser.add_argument(
      '-k', '--top_k', type=int, default=1,
      help='Max number of classification results')
  parser.add_argument(
      '-t', '--threshold', type=float, default=0.0,
      help='Classification score threshold')
  parser.add_argument(
      '-c', '--count', type=int, default=5,
      help='Number of times to run inference')
  args = parser.parse_args()

  labels_dict = CXR_test_label(args.labels)
  label_names = ["Atel", "Card", "Cons", "Edem", "Pleu"]

  print("Model name:", args.model)
  interpreter = tf.lite.Interpreter(args.model)
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
    labels_dict[key].append([data_pred_cat[f"{i}"][1] for i in range(5)])
    print(labels_dict[key])
    
  print(f"Avg. {sum(inference_time_list) / len(labels_dict):.1f}ms per image.")
  
  labels = np.array([value[0] for value in labels_dict.values()])
  output = np.array([value[1] for value in labels_dict.values()])

  print("[ AUROC score ]")
  roc_cum = 0
  for i in range(5):
    fpr, tpr, _ = roc_curve(labels[:, i], output[:, i])
    roc_auc = metrics.auc(fpr, tpr)
    roc_cum += roc_auc
    print(f"{label_names[i]}: {roc_auc:.3f}")
  print(f"*Avg: {roc_cum / 5:.3f}")

if __name__ == '__main__':
  main()
