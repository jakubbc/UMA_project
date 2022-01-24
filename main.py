#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Learning -- Project
Jakub Ciemięga, Klaudia Stpiczyńska 2022
Warsaw University of Technology
"""


from aq_algorithm import use_aq_simple
from cross_validation import cross_validation_aq, cross_validation_cn2, count_times_aq, count_times_cn2
from tests import *

dataset_list = ['alcohol', 'ttt', 'spect']

# params for simple use
fname = 'data/student-mat.csv'  # file containing data
target_col_num = -6  # number of target column, starting from 0, negative values start from the back (-1 is last column)
headers = True  # decides whether file under fname contains column names (True) or not (False)
split = 0.8  # fraction of dataset in simple use used for training, the rest used for testing, has to be smaller than 1

# params for aq algorithm:
num_best = 2
quality_index_type = 0  # 0 or 1

# parameter for cross validation
max_k = 10
rules_from_file = False

if __name__ == "__main__":

    # simple example to use the aq algorithm on a single data frame from file
    use_aq_simple(fname, target_col_num, headers, split, num_best, quality_index_type)

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

    # cn2 experiment
    # for element in dataset_list:
    #     accuracy = cross_validation_cn2(rules_from_file, element, max_k)
    #     print(element + " cn2 accuracy: ", accuracy)

    # aq experiment
    # for element in dataset_list:
    #     accuracy = cross_validation_aq(rules_from_file, element, num_best, quality_index_type, max_k)
    #     print(element + " aq accuracy: ", accuracy)
