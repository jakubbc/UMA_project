""" cross_validation.py
Authors: Jakub Ciemięga, Klaudia Stpiczyńska
"""

import pandas as pd
import os
from cn2_algorithm import *
from aq_algorithm import *
from data_manager import *

from enum import Enum

from timeit import default_timer as timer


def cross_validation_cn2(rules_from_files, filename, max_k):
    """
    cross validation for CN2 algorithm for tic-tac-toe and student alcohol consumption dataset
    and experiment for spect data set
    return overall accuracy
    """
    path = 'data/'

    df = create_prepared_table(filename)
    overall_accuracy=0
    if filename != 'spect':
        for i in range(0, max_k):
            accuracy = cn2_test( df, filename, rules_from_files, i+1, max_k)
            overall_accuracy = overall_accuracy+accuracy
        overall_accuracy = overall_accuracy/max_k
        os.remove(path + filename + '-train.csv')
        os.remove(path + filename + '-test.csv')
    else:
        overall_accuracy = cn2_test( df, filename, rules_from_files, 1, 1)

    return overall_accuracy


def cross_validation_aq(rules_from_files, filename, num_best, quality_index_type, max_k):
    """
    cross validation for AQ algorithm for tic-tac-toe and student alcohol consumption dataset
    and experiment for spect data set
    return overall accuracy
    """

    path = 'data/'

    df = create_prepared_table(filename)
    overall_accuracy = 0

    if filename != 'spect':
        for i in range(0, max_k):
            accuracy = aq_test(df, filename, rules_from_files, num_best, quality_index_type, i+1, max_k)
            print(accuracy)
            overall_accuracy = overall_accuracy + accuracy
        overall_accuracy = overall_accuracy / max_k
    else:
        overall_accuracy = aq_test(df, filename, rules_from_files, num_best, quality_index_type, 1, 1)

    return overall_accuracy


def cn2_test(data, filename, rules_from_file, current_k, max_k):
    '''
    test CN2 method for one train and test data set
    return accuracy of alogrithm for test dataset
    '''
    print("something")
    test_length = int(len(data) / max_k)

    path = 'data/'

    if filename=='spect':
        save_spect_train_table()
        load_spect_test_table()
        sufix = 'spect'
        possible_values=['1', '0']
    else:

        test_begin = ((max_k - 1) - (current_k - 1))*test_length +1
        test_end= (max_k - current_k+1)*test_length
        if test_end > len(data):
            test_end = len(data)
        if test_begin == 1:
            test_begin = 0
        df_test =data[test_begin:test_end]
        df_train = data.drop(range(test_begin, test_end))
        sufix = str(current_k)


        if filename == 'alcohol':
            possible_values = ['1', '2', '3', '4', '5']
        elif filename == 'ttt':
            possible_values = ['positive', 'negative']

        df_train.to_csv(path + filename + '-train.csv', index=False)
        df_test.to_csv(path + filename + '-test.csv', index=False)

    cn2_fit = CN2algorithm(path + filename + '-train.csv', path + filename + '-test.csv')

    if not rules_from_file:
        cn2_rules = cn2_fit.fit_CN2()
        save_rules_to_file(cn2_rules, path + filename + 'rules-cn2' + sufix)
    else:
        cn2_rules = load_rules_from_file(path + filename + 'rules-cn2' + sufix)

    cn2_test = cn2_fit.test_fitted_model(possible_values, cn2_rules, cn2_fit.test_set)[0]
    # print("Accuracy: " + filename + " " + sufix, cn2_fit.accuracy)
    cn2_fit.check_for_all(cn2_rules, possible_values)
    #uncomment for precise result
    # cn2_test.to_csv(path + filename + 'result-cn2' + sufix + '.csv')
    return cn2_fit.accuracy


def aq_test(data, filename, rules_from_file, num_best, quality_index_type, current_k, max_k):
    """
    test AQ method for one train and test data set
    return accuracy of algorithm for test dataset
    """
    test_length = int(len(data) / 3)

    path = 'data/'

    if filename == 'spect':
        df_train = data
        df_test = load_spect_test_table()
        sufix = 'spect'
        possible_values = ['0', '1']
    else:

        test_begin = ((max_k - 1) - (current_k - 1)) * test_length + 1
        test_end = (max_k - current_k + 1) * test_length
        if test_end > len(data):
            test_end = len(data)
        if test_begin == 1:
            test_begin = 0
        df_test = data[test_begin:test_end]
        df_train = data.drop(range(test_begin, test_end))
        sufix = str(current_k)

        if filename == 'alcohol':
            possible_values = ['1', '2', '3', '4', '5']
        elif filename == 'ttt':
            possible_values = ['positive', 'negative']

    if not rules_from_file:
        rules = induce_rules(df_train, num_best, quality_index_type)
        save_rules_to_file(rules, path + filename + 'rules-aq' + sufix)
    else:
        rules = load_rules_from_file(path + filename + 'rules-aq' + sufix)
    df_pred = predict_table(possible_values, rules, df_test, 'pred')
    accuracy = np.round(np.sum(df_pred.iloc[:, -2] == df_pred.iloc[:, -1]) / df_pred.shape[0] * 100, 3)
    # print("Accuracy: " + filename + " " + sufix, accuracy)
    return accuracy


def count_times_cn2(filename):
    path = 'data/'
    data = create_prepared_table(filename)
    test_length = int(len(data) / 3)
    if filename=='spect':
        save_spect_train_table()
        load_spect_test_table()
    else:
        df_train = data[0:2 * test_length]
        df_test = data[2 * test_length + 1:]

        df_train.to_csv(path + filename + '-train.csv', index=False)
        df_test.to_csv(path + filename + '-test.csv', index=False)

    cn2_fit = CN2algorithm(path + filename + '-train.csv', path + filename + '-test.csv')
    start_time = timer()
    cn2_fit.fit_CN2()
    end_time = timer()
    return end_time - start_time


def count_times_aq(filename, num_best, quality_index_type):
    path = 'data/'
    data = create_prepared_table(filename)
    test_length = int(len(data) / 3)
    if filename=='spect':
        df_train=data
    else:
        df_train = data[0:2 * test_length]


    start_time = timer()
    rules = induce_rules(df_train, num_best, quality_index_type)
    end_time = timer()
    return end_time - start_time










