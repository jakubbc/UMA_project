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

    # when working on a project, execute this once to create a tic-tac-toe or spect table from raw data
    # save_ttt_table()
    # save_spect_train_table()


    #aq

    # df = create_prepared_table(dataset)
    # # df = create_test_table()
    # print(df)
    #
    # if not rules_from_file:
    #     rules = induce_rules(df.loc[:split], num_best, quality_index_type)
    #     save_rules_to_file(rules, 'rules')
    # else:
    #     rules = load_rules_from_file(f'rules_{dataset}')
    # print(f'rules number: {len(rules)}\n{rules}')
    #
    # # load test data set for spect data
    # if 'spect' == dataset:
    #     df = load_spect_test_table()
    #     split = 0
    #
    # df_pred = predict_table(rules, df.loc[split:], 'pred')
    #
    # print(df_pred)
    # print(f'Accuracy: {np.round(np.sum(df_pred.iloc[:, -2] == df_pred.iloc[:, -1]) / df_pred.shape[0] * 100, 3)}%')


    for element in dataset_list:
        accuracy = cross_validation(rules_from_file, element, Algorithm.CN2)
        print(element +" accuracy: ", accuracy)


