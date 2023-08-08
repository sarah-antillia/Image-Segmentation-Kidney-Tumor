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
# ImageMaskDatasetGenerator.py

import os
import sys
import shutil
from xml.dom import registerDOMImplementation
import cv2
from PIL import Image, ImageOps

import glob
import numpy as np
import math
import traceback

class ImageMaskDatasetGenerator:
  def __init__(self, mask_size=40, angles=[90, 180, 270], debug=False):
    self.MASK_SIZE= mask_size
    self.DEBUG    = debug
    self.ANGLES   = angles

  def augment(self, image, output_dir, filename):
    image = Image.fromarray(image)
    if len(self.ANGLES) > 0:
      for angle in self.ANGLES:
        rotated_image = image.rotate(angle)
        output_filename = "rotated_" + str(angle) + "_" + filename
        rotated_image_file = os.path.join(output_dir, output_filename)
        rotated_image.save(rotated_image_file)
        print("=== Saved {}".format(rotated_image_file))
      
    # Create mirrored image
    mirrored = ImageOps.mirror(image)
    output_filename = "mirrored_" + filename
    image_filepath = os.path.join(output_dir, output_filename)
    
    mirrored.save(image_filepath)
    print("=== Saved {}".format(image_filepath))
        
    # Create flipped image
    flipped = ImageOps.flip(image)
    output_filename = "flipped_" + filename

    image_filepath = os.path.join(output_dir, output_filename)

    flipped.save(image_filepath)
    print("=== Saved {}".format(image_filepath))


  def resize_to_square(self, image, RESIZE=512):
     image = Image.fromarray(image)
     w, h = image.size
     bigger = w
     if h >bigger:
       bigger = h
     background = Image.new("RGB", (bigger, bigger))
     x = (bigger - w)//2
     y = (bigger - h)//2

     background.paste(image, (x, y))
     background = background.resize((RESIZE, RESIZE))
     return background

  def generate(self, images_dir, masks_dir, output_images_dir, output_masks_dir):
    image_files = glob.glob(images_dir + "/*.jpg")
    mask_files  = glob.glob(masks_dir  + "/*.jpg")
    num_image_files = len(image_files)
    num_mask_files  = len(mask_files)
    print("=== num_image_files {}".format(num_image_files))
    print("=== num_mask_files {}".format(num_mask_files))

    for mask_file in mask_files:
      print("=== mask_file {}".format(mask_file))
      mask = cv2.imread(mask_file)
      mask = cv2.cvtColor(mask, cv2.COLOR_RGB2BGR)
      basename = os.path.basename(mask_file)
      image_file = os.path.join(images_dir, basename)
      image = cv2.imread(image_file)
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

      rect_output_maskfile = os.path.join(output_masks_dir, "rect_" + basename)

      rect  = self.getBoundingBox(mask, rect_output_maskfile)
      print("--- rect {}".format(rect))
      (x, y, w, h) = rect
 
      output_maskfile  = os.path.join(output_masks_dir, basename)
      output_imagefile = os.path.join(output_images_dir, basename)
 
      if w < self.MASK_SIZE or h < self.MASK_SIZE:
        # Exclude a too small mask file 
        print("--- Skipped a too small mask file {}".format(output_maskfile))
        continue

      shutil.copy2(mask_file, output_masks_dir)
      print("=== Copied mask file to  {}".format(output_maskfile))  

      shutil.copy2(image_file, output_images_dir)
      print("=== Copied image file to  {}".format(output_imagefile))  

      self.augment(mask, output_masks_dir, basename)
      self.augment(image, output_images_dir, basename)


  def getBoundingBox(self, mask, output_file):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    ret, binarized = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
    points = np.array(contours[0])

    x, y, w, h = cv2.boundingRect(points)
    rect = (x, y, w, h)
    if self.DEBUG:
      color= (0, 255, 0)
      cv2.rectangle(mask, rect, color=color, thickness=2)
      cv2.imwrite(output_file, mask)
    return rect
  

if __name__ == "__main__":
  try:
    images_dir = "./Kits19-base/images/"
    masks_dir  = "./Kits19-base/masks/"
  
    output_images_dir = "./Kits19-master/images/"
    output_masks_dir  = "./Kits19-master/masks/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    generator = ImageMaskDatasetGenerator(mask_size=40, angles=[], debug=False)
    generator.generate(images_dir, masks_dir, output_images_dir, output_masks_dir)

  except:
    traceback.print_exc()


