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
dataset_ind = 2
dataset_list = ['alcohol', 'ttt', 'spect']
dataset = dataset_list[dataset_ind]
# decide what fraction of dataset used for training, the rest used for testing, has to be smaller than 1
split = 0.9  # does not apply to spect data since train and test sets specified

# decide if to read rules from file, otherwise rules are induced (takes longer)
rules_from_file = False

# params for aq algorithm:
num_best = 2
quality_index_type = 0
# parameter for cross validation
max_k = 3

if __name__ == "__main__":

    # simple example to use the aq algorithm on a single data frame
    """
    # df must have column names, target column (class) needs to be the last
    # rows should be shuffled for better training, e.g. df = df.sample(frac=1).reset_index(drop=True)
    df = create_prepared_table(dataset)

    # simple test table for debugging purposes
    # df = create_test_table()

    if not rules_from_file:
        rules = induce_rules(df.loc[:int(df.shape[0]*split)], num_best, quality_index_type)
        save_rules_to_file(rules, 'rules')
    else:
        rules = load_rules_from_file(f'rules_{dataset}')

    # load test data set for spect data
    if 'spect' == dataset:
        df = load_spect_test_table()
        split = 0
    df_pred = predict_table(rules, df.loc[int(df.shape[0]*split):], 'pred')
    print(f'Accuracy: {np.round(np.sum(df_pred.iloc[:, -2] == df_pred.iloc[:, -1]) / df_pred.shape[0] * 100, 3)}%')
    """
    # tests:
    # print(f'test covers(): {test_covers()}')
    # print(f'test delete_non_general(): {test_delete_non_general()}')

    # time measure

    # for element in dataset_list:
    #     time = count_times_cn2(element)
    #     print(element + "  cn2 time: ")
    #     print(time)
    #
    #     time = count_times_aq(element, num_best, quality_index_type)
    #     print(element + "  aq time: ")
    #     print(time)

    # # cn2
    # for element in dataset_list:
    #     accuracy = cross_validation_cn2(rules_from_file, element, max_k)
    #     print(element + " cn2 accuracy: ", accuracy)

    # # aq
    # for element in dataset_list:
    #     accuracy = cross_validation_aq(rules_from_file, element, num_best, quality_index_type, max_k)
    #     print(element + " aq accuracy: ", accuracy)
