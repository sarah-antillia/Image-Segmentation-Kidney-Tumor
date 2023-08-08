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

# 2023/08/10 
# create_base_dataset.py

import os
import sys
import shutil
import cv2

import glob
import numpy as np
import math
import nibabel as nib
import traceback

# Read file
"""
scan = nib.load('/path/to/stackOfimages.nii.gz')
# Get raw data
scan = scan.get_fdata()
print(scan.shape)
(num, width, height)

"""


# See : https://github.com/neheller/kits19/blob/master/starter_code/visualize.py


KIDNEY_COLOR = [255, 0, 0]
TUMOR_COLOR  = [0, 0, 255]



# This function has been taken from visualize.py
# https://github.com/neheller/kits19/blob/master/starter_code/visualize.py
#
def class_to_color(segmentation, kidney_color, tumor_color, tumor_only=True):
    # initialize output to zeros
    shp = segmentation.shape
    seg_color = np.zeros((shp[0], shp[1], 3), dtype=np.float32)

    # set output to appropriate color at each location
    # 2023/08/10 antillia.com
    if tumor_only:
      # Set a kidney mask color to be black 
      kidney_color = [0, 0, 0]
    seg_color[np.equal(segmentation, 1)] = kidney_color
    seg_color[np.equal(segmentation, 2)] = tumor_color
    return seg_color
  
"""
def get_mask_boundingbox( mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    ret, bin_img = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
       bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #contours = list(filter(lambda x: cv2.contourArea(x) > 100, contours))
    points = np.array(contours[0])
    #print(points)
    x, y, w, h = cv2.boundingRect(points)
    rect = (x, y, w, h)
    return rect
"""


def create_mask_files(niigz, output_dir, index):
    print("--- niigz {}".format(niigz))
    nii = nib.load(niigz)

    print("--- nii {}".format(nii))
    data = nii.get_fdata()
    print("---data shape {} ".format(data.shape))
    #data = np.asanyarray(nii.dataobj)
    num_images = data.shape[0] # math.floor(data.shape[2]/2)
    print("--- num_images {}".format(num_images))
    num = 0
    for i in range(num_images):
      img = data[i, :, :]
      img = self.class_to_color(img, KIDNEY_COLOR, TUMOR_COLOR) 
      img = np.array(img)
      if np.any(img > 0):
        filepath = os.path.join(output_dir, str(index) + "_" + str(i) + ".jpg")
        cv2.imwrite(filepath, img)
        print("Saved {}".format(filepath))
        num += 1
    return num
  
def create_image_files(niigz, output_masks_dir, output_images_dir, index):
   
    print("--- create_image_files niigz {}".format(niigz))
    nii = nib.load(niigz)

    print("--- nii {}".format(nii))
    data = nii.get_fdata()
    print("---data shape {} ".format(data.shape))
    #data = np.asanyarray(nii.dataobj)
    num_images = data.shape[0] # math.floor(data.shape[2]/2)
    print("--- num_images {}".format(num_images))
    num = 0
    for i in range(num_images):
      img = data[i, :, :]
   
      filename = str(index) + "_" + str(i) + ".jpg"
      mask_filepath = os.path.join(output_masks_dir, filename)
      if os.path.exists(mask_filepath):
        filepath = os.path.join(output_images_dir, filename)
   
        cv2.imwrite(filepath, img)
        print("Saved {}".format(filepath))
        num += 1
    return num
  

def create_base_dataset(data_dir, output_images_dir, output_masks_dir):
    dirs = glob.glob(data_dir)

    print("--- num dirs {}".format(len(dirs)))

    index = 10000
    for dir in dirs:
      print("== dir {}".format(dir))
      image_nii_gz_file = os.path.join(dir, "imaging.nii.gz")
      seg_nii_gz_file   = os.path.join(dir, "segmentation.nii.gz")
      index += 1
      if os.path.exists(image_nii_gz_file) and os.path.exists(seg_nii_gz_file):
        num_segmentations = create_mask_files(seg_nii_gz_file,   output_masks_dir,  index)
        num_images        = create_image_files(image_nii_gz_file, output_masks_dir, output_images_dir, index)
        print(" image_nii_gz_file: {}  seg_nii_gz_file: {}".format(num_images, num_segmentations))

        if num_images != num_segmentations:
          raise Exception("Num images and segmentations are different ")
      else:
        print("Not found segmentation file {} corresponding to {}".format(seg_nii_gz_file, image_nii_gz_file))


if __name__ == "__main__":
  try:
    data_dir          = "./data/case_*"
    output_images_dir = "./Kits19-base/images/"
    output_masks_dir  = "./Kits19-base/masks/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    # Create jpg image and mask files from nii.gz files under data_dir.
    create_base_dataset(data_dir, output_images_dir, output_masks_dir)

  except:
    traceback.print_exc()


