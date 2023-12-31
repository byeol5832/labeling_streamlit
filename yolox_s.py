#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.num_classes = 20
        self.data_num_workers = 2

        self.data_dir = '/home/armstrong/Desktop/hboh/streamlit/testing_streamlit'

        self.input_size = (640, 640)
        self.test_size = (640, 640)

        self.name = 'train'
