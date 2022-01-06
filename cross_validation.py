""" cross_validation.py
Authors: Jakub Ciemięga, Klaudia Stpiczyńska
"""

import pandas as pd
import os
from cn2_algorithm import *
from aq_algorithm import *
from data_manager import *

from enum import Enum

class Algorithm (Enum):
    AQ = 1
    CN2 = 2


def cross_validation(rules_from_files, filename, algorithm):
    path = 'data/'

    if algorithm == Algorithm.CN2:
        df=create_prepared_table(filename)

        if filename != 'spect':
            accuracy1 = cn2_test(1, df, filename, rules_from_files)
            accuracy2 = cn2_test(2, df, filename, rules_from_files)
            accuracy3 = cn2_test(3, df, filename, rules_from_files)
            overall_accuracy = (accuracy1 + accuracy2 + accuracy3)/3
            os.remove(path + filename +'-train.csv')
            os.remove(path + filename + '-test.csv')
        else:
            overall_accuracy = cn2_test(1, df, filename, rules_from_files)

    return overall_accuracy


def cn2_test(data_set_number, data, filename, rules_from_file):
    test_length = int(len(data) / 3)

    path = 'data/'

    if filename=='spect':
        save_spect_train_table()
        load_spect_test_table()
        sufix = 'spect'
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

        df_train.to_csv(path + filename + '-train.csv', index=False)
        df_test.to_csv(path + filename + '-test.csv', index=False)

    cn2_fit = CN2algorithm(path + filename + '-train.csv', path + filename + '-test.csv')

    if not rules_from_file:
        cn2_rules = cn2_fit.fit_CN2()
        save_rules_to_file(cn2_rules, path + filename + 'rules-cn2' + sufix)
    else:
        cn2_rules = load_rules_from_file(path + filename + 'rules-cn2' + sufix)

    cn2_test = cn2_fit.test_fitted_model(cn2_rules, cn2_fit.test_set)[0]
    print("Accuracy: " + filename + " " + sufix, cn2_fit.accuracy)
    #uncomment for precise result
    # cn2_test.to_csv(path + filename + 'result-cn2' + sufix + '.csv')
    return cn2_fit.accuracy




