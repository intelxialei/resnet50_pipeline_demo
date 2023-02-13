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
"""Training script for Mask-RCNN."""
import os
from argparse import Namespace
from run import run_inference

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["TF_CPP_VMODULE"] = 'non_max_suppression_op=0,generate_box_proposals_op=0,executor=0'

from arguments import PARSER
from config import CONFIG
from dataset import Dataset


def main():

    # setup params
    arguments = PARSER.parse_args()
    params = Namespace(**{**vars(CONFIG), **vars(arguments)})

    # setup dataset
    dataset = Dataset(params)

    if params.mode == 'infer':
        run_inference(dataset, params)
    


if __name__ == '__main__':
    main()
