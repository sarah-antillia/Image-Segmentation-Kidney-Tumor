# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2023/08/10 to-arai 
# create_mini_eval.py

import os
import sys
import shutil
import random

import glob
import traceback

def create_mini_eval(images_dir, masks_dir, output_images_dir, output_masks_dir, num=100):
  image_files = glob.glob(images_dir + "/*.jpg")
  random.seed = 137
  random.shuffle(image_files)
  print(image_files)

  image_files = random.sample(image_files, num)
  for image_file in image_files:
    basename = os.path.basename(image_file)

    shutil.copy2(image_file, output_images_dir)
    print("Copied {} to {}".format(image_file, output_images_dir))

    mask_file = os.path.join(masks_dir, basename)
    shutil.copy2(mask_file,  output_masks_dir)
    print("Copied {} to {}".format(mask_file, output_masks_dir))


if __name__ == "__main__":
  try:
    images_dir = "./Kidney-Tumor/valid/images/"
    masks_dir  = "./Kidney-Tumor/valid/masks/"
    output_images_dir = "./mini_eval/images/"
    output_masks_dir  = "./mini_eval/masks/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    create_mini_eval(images_dir, masks_dir, output_images_dir, output_masks_dir, num=40)
  
  except:
    traceback.print_exc()
