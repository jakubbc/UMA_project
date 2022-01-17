#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Learning -- Project
Jakub Ciemięga, Klaudia Stpiczyńska 2022
Warsaw University of Technology
"""

import numpy as np
from data_manager import create_test_table, create_prepared_table, save_ttt_table, save_spect_train_table, \
    load_spect_test_table
from aq_algorithm import induce_rules, predict_table, save_rules_to_file, load_rules_from_file
from tests import *
from cross_validation import *

# program parameters
# pick one of three datasets
dataset_ind = 1
dataset_list = ['alcohol', 'ttt', 'spect']
dataset = dataset_list[dataset_ind]
rules_from_file = True
# table before split used for training, after split for testing
split = 900  # does not apply to spect data since train and test sets specified

# params for aq algorithm:
num_best = 2
quality_index_type = 0

if __name__ == "__main__":



    # tests:
    # print(f'test covers(): {test_covers()}')
    # print(f'test delete_non_general(): {test_delete_non_general()}')

    # cn2

    for element in dataset_list:
        accuracy = cross_validation_cn2(rules_from_file, element)
        print(element +" cn2 accuracy: ", accuracy)

    # #aq

    for element in dataset_list:
        accuracy = cross_validation_aq(rules_from_file, element, num_best, quality_index_type)
        print(element + " aq accuracy: ", accuracy)


