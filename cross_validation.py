""" cross_validation.py
Authors: Jakub Ciemięga, Klaudia Stpiczyńska
"""

import pandas as pd
import os
from cn2_algorithm import *
from aq_algorithm import *
from data_manager import *

from enum import Enum




def cross_validation_cn2(rules_from_files, filename):
    '''
    cross validation for CN2 algorithm for tic-tac-toe and student alcohol consumption dataset
    and experiment for spect data set
    return overall accuracy
    '''
    path = 'data/'

    df = create_prepared_table(filename)

    if filename != 'spect':
        accuracy1 = cn2_test(1, df, filename, rules_from_files)
        accuracy2 = cn2_test(2, df, filename, rules_from_files)
        accuracy3 = cn2_test(3, df, filename, rules_from_files)
        overall_accuracy = (accuracy1 + accuracy2 + accuracy3)/3
        overall_accuracy=overall_accuracy*100
        os.remove(path + filename +'-train.csv')
        os.remove(path + filename + '-test.csv')
    else:
        overall_accuracy = cn2_test(1, df, filename, rules_from_files)
        overall_accuracy=overall_accuracy*100

    return overall_accuracy

def cross_validation_aq(rules_from_files, filename, num_best, quality_index_type):
    '''
    cross validation for AQ algorithm for tic-tac-toe and student alcohol consumption dataset
    and experiment for spect data set
    return overall accuracy
    '''

    path = 'data/'

    df = create_prepared_table(filename)

    if filename != 'spect':
        accuracy1 = aq_test(1, df, filename, rules_from_files, num_best, quality_index_type)
        accuracy2 = aq_test(2, df, filename, rules_from_files, num_best, quality_index_type)
        accuracy3 = aq_test(3, df, filename, rules_from_files, num_best, quality_index_type)
        overall_accuracy = (accuracy1 + accuracy2 + accuracy3)/3
    else:
        overall_accuracy = aq_test(1, df, filename, rules_from_files, num_best, quality_index_type)


    return overall_accuracy


def cn2_test(data_set_number, data, filename, rules_from_file):
    '''
    test CN2 method for one train and test data set
    return accuracy of alogrithm for test dataset
    '''
    print("something")
    test_length = int(len(data) / 3)

    path = 'data/'

    if filename=='spect':
        save_spect_train_table()
        load_spect_test_table()
        sufix = 'spect'
        possible_values=['1', '0']
    else:

        if data_set_number == 1:
            df_train = data[0:2 * test_length]
            df_test = data[2 * test_length + 1:]
            sufix='one'
        elif data_set_number == 2:
            df_train=data
            df_train = df_train.drop(range(test_length+1, 2*test_length))
            df_test=data[test_length+1:2*test_length]
            sufix = 'two'
        elif data_set_number == 3:
            df_train = data[test_length + 1:]
            df_test = data[0:test_length]
            sufix = 'three'
        else:
            print("Only k=3 for cross validation")
            return 0

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


def aq_test(data_set_number, data, filename, rules_from_file, num_best, quality_index_type):
    '''
    test AQ method for one train and test data set
    return accuracy of alogrithm for test dataset
    '''
    test_length = int(len(data) / 3)

    path = 'data/'

    if filename == 'spect':
        df_train = data
        df_test = load_spect_test_table()
        sufix = 'spect'
        possible_values = ['0', '1']
    else:

        if data_set_number == 1:
            df_train = data[0:2 * test_length]
            df_test = data[2 * test_length + 1:]
            sufix = 'one'
        elif data_set_number == 2:
            df_train = data
            df_train = df_train.drop(range(test_length + 1, 2 * test_length))
            df_test = data[test_length + 1:2 * test_length]
            sufix = 'two'
        elif data_set_number == 3:
            df_train = data[test_length + 1:]
            df_test = data[0:test_length]
            sufix = 'three'
        else:
            print("Only k=3 for cross validation")
            return 0

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








