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
# dataset_list=['spect', 'alcohol']
dataset = dataset_list[dataset_ind]
rules_from_file = True
# table before split used for training, after split for testing
split = 900  # does not apply to spect data since train and test sets specified

# params for aq algorithm:
num_best = 2
quality_index_type = 0

if __name__ == "__main__":

    # df = create_prepared_table('alcohol')
    # # print(df.shape)
    # print("klasa", df.iloc[1,-1])
    # print("klasa", df.iloc[2, -1])


    # tests:
    # print(f'test covers(): {test_covers()}')
    # print(f'test delete_non_general(): {test_delete_non_general()}')

    # cn2

    for element in dataset_list:
        # path = 'data/'
        # sufix = 'spect'
        # cn2_fit = CN2algorithm(path + element + '-train.csv', path + element + '-test.csv')
        # cn2_rules = load_rules_from_file(path + element + 'rules-cn2' + sufix)
        accuracy = cross_validation_cn2(rules_from_file, element)
        print(element +" cn2 accuracy: ", accuracy)
        # df = create_prepared_table(element)
        # for i in range(len(df)):
        #     print(df.loc[i, 'class'])
        #     row = df.loc[i, :]
        #     print(type(row))
        #     result = cn2_fit.check(row, cn2_rules)
        #     print(result)

    # #aq
    #
    # for element in dataset_list:
    #     accuracy = cross_validation_aq(rules_from_file, element, num_best, quality_index_type)
    #     print(element + " aq accuracy: ", accuracy)


