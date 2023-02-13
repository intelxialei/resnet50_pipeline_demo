# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Data loading and processing.

Defines dataset class that exports input functions of Mask-RCNN
for training and evaluation using Estimator API.

The train_fn includes training data for category classification,
bounding box regression, and number of positive examples to normalize
the loss during training.

"""
import glob
import logging
import os

import cv2
import numpy as np
from PIL import Image
import threading
import tensorflow as tf
from tqdm import tqdm
import copy 
class Dataset:
    """ Load and preprocess the imagenet dataset. """

    def __init__(self, params):
        """ Configures dataset. """
        self._params = params

        self._logger = logging.getLogger('dataset')

        logging.info("reading dataset to memory (may take a while)... ")
        self._size = 1536
        self._eval_samples = []
        if params.preprocess_method==3:
            '''
            for f in tqdm(os.listdir(self._params.data_dir)[:self._size]):
                f = os.path.join(self._params.data_dir, f)
                if os.path.isfile(f):
                    self._eval_samples.append(copy.deepcopy(Image.open(f)))
            #exit()
            '''
            self._eval_samples = [copy.deepcopy(Image.open(os.path.join(self._params.data_dir, f))) for f in os.listdir(self._params.data_dir)[:self._size] if os.path.isfile(os.path.join(self._params.data_dir, f))]
        else:
            self._eval_samples = [cv2.imread(os.path.join(self._params.data_dir, f)) for f in os.listdir(self._params.data_dir)[:self._size] if os.path.isfile(os.path.join(self._params.data_dir, f))]
        self._size = len(self._eval_samples )
        
        self.lock = threading.Lock()
        self._i = 0
        self._chunks = (self._size - 1) // self._params.eval_batch_size + 1

    def __iter__(self):
        return self
    def reset(self):
        self._i = 0

    def __len__(self):
        return self._size

    def chunks(self):
        return self._chunks

    def __next__(self):
        with self.lock:
            if self._params.preprocess_method==0: # list
                while self._i< self._chunks:
                    raw_batch = self._eval_samples[self._i*self._params.eval_batch_size:(self._i+1)* self._params.eval_batch_size]
                    im_list = []
                    for j, im in enumerate(raw_batch):
                        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im_rgb = cv2.resize(im_rgb, self._params.image_size)
                        im_list.append(im_rgb)
                    preprocessed_batch = np.array(im_list)
                    #print(self._i, preprocessed_batch.shape)
                    self._i +=1
                    return preprocessed_batch
                else:
                    raise StopIteration

            if self._params.preprocess_method==1: # numpy
                preprocessed_batch = np.empty((self._params.eval_batch_size, *self._params.image_size, 3))
                while self._i< self._chunks:
                    raw_batch = self._eval_samples[self._i*self._params.eval_batch_size:(self._i+1)* self._params.eval_batch_size]
                    for j, im in enumerate(raw_batch):
                        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        preprocessed_batch[j] = cv2.resize(im_rgb, self._params.image_size)
                    self._i +=1
                    return preprocessed_batch
                else:
                    raise StopIteration

            if self._params.preprocess_method==2: # list + tf.image.resize
                while self._i< self._chunks:
                    raw_batch = self._eval_samples[self._i*self._params.eval_batch_size:(self._i+1)* self._params.eval_batch_size]
                    im_list = []
                    for j, im in enumerate(raw_batch):
                        im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im_tensor = tf.image.resize(np.asarray(im_rgb), self._params.image_size, antialias=True)
                        im_list.append(im_tensor)
                    preprocessed_batch = tf.stack(im_list)
                    print(self._i, tf.shape(preprocessed_batch).numpy().tolist())
                    #exit()
                    self._i +=1
                    return preprocessed_batch
                else:
                    raise StopIteration

            if self._params.preprocess_method==3: # list + pillow_simd
                while self._i< self._chunks:
                    raw_batch = self._eval_samples[self._i*self._params.eval_batch_size:(self._i+1)* self._params.eval_batch_size]
                    im_list = []
                    for j, im in enumerate(raw_batch):
                        
                        #w_scale = self._params.image_size[1]/im.width
                        #h_scale = self._params.image_size[0]/im.height

                        im =im.resize(self._params.image_size)
                        im = im.convert('RGB')
                        #im= im.resize(self._params.image_size[1] / max(im.width, im.height)， kernel='linear')
                        #im.write_to_file("test2.jpg")
                        #exit()
                        #im =im.thumbnail_image(width=self._params.image_size[1], height=self._params.image_size[0], size="force")
                        tmp = np.array(im)
                        im_list.append(np.array(im))
                    preprocessed_batch = np.array(im_list)
                    #print(preprocessed_batch.shape)
                    #print(self._i, tf.shape(preprocessed_batch).numpy().tolist())
                    #exit()
                    self._i +=1
                    return preprocessed_batch
                else:
                    raise StopIteration


if __name__=="__main__":
    from argparse import Namespace
    params = Namespace(**dict(
        image_size=(224, 224),
        eval_batch_size =  512,
        data_dir ='/mnt/nfs/share/LSVRC2012/val',
        preprocess_method = 3,

        ))
    dataset = Dataset(params)
    for x in dataset:
        print(x.shape)
