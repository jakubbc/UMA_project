#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" data_manager.py
Authors: Jakub Ciemięga, Klaudia Stpiczyńska
"""
import pandas as pd


def create_test_table() -> pd.DataFrame:
    """ create simple data set for tests

    :return df: test data set
    :rtype df: pd.DataFrame
    """
    outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny,sunny,' \
              'sunny'.split(',')
    temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild,mild,mild'.split(',')
    # humidity = 'high,normal,high,normal,high,normal,high,normal,high,high,high,high,normal,normal,normal,normal' \
    humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal,normal,normal' \
        .split(',')
    # windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE,TRUE,TRUE'.split(',')
    windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE,TRUE,TRUE'.split(',')
    play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes,no,yes'.split(',')

    dataset = {'outlook': outlook, 'temp': temp, 'humidity': humidity, 'windy': windy, 'play': play}
    df = pd.DataFrame(dataset, columns=['outlook', 'temp', 'humidity', 'windy', 'play'])

    return df


def create_prepared_table(dataset: str) -> pd.DataFrame:
    """ create and return one of 3 available datasets

    :param dataset: name of the desired dataset, must be in ['alcohol', 'ttt', 'spect']
    :type dataset: str

    :return df: merged and prepared dataframe to apply AQ algorithm to
    :rtype df: pd.DataFrame
    """
    if 'alcohol' == dataset:
        df1 = pd.read_csv('data/student-mat.csv')
        df2 = pd.read_csv('data/student-por.csv')
        df = pd.concat([df1, df2], ignore_index=True)  # ignore index because there are index duplicates
        columns = list(df.columns)
        # delete Walc and Dalc from the middle
        del columns[26:28]
        # add column to predict
        columns.append('Walc')
        df = df[columns]
    elif 'ttt' == dataset:
        df = pd.read_csv('data/ttt.csv')
    elif 'spect' == dataset:
        df = pd.read_csv('data/spect_train.csv')

    # since 'class_value' is used in the induced rules, rename such columns
    df = df.rename(columns={'class_value': 'class_value_1'})

    return df


def save_ttt_table() -> None:
    """ reads the tic-tat-toe data and saves it in one file for later use
        Don't directly read from file since it's sorted by class which affects rule induction (no negative kernel
        available)
    """
    df = pd.read_csv('data/tic-tac-toe.data', header=None)
    # assign column names
    # df.columns = ['top-left', 'top-middle', 'top-right', 'middle-left', 'middle-middle', 'middle-right',
    #               'bottom-left', 'bottom-middle', 'bottom-right', 'x_win']
    df.columns = ['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'bl', 'bm', 'br', 'x_win']
    # shuffle for better learning
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('ttt.csv', index=False)


def save_spect_train_table() -> None:
    """ reads the spect train data and saves it in one file for later use
        Don't directly read from file since it's sorted by class which affects rule induction (no negative kernel
        available)
    """
    df = pd.read_csv(f'data/SPECT.train', header=None)
    # assign column names
    columns = [f'F{num}' for num in range(1, 23)]
    columns.insert(0, 'overall')
    df.columns = columns
    # move class column to last column
    del columns[0]
    columns.append('overall')
    df = df[columns]
    # shuffle for better learning
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(f'data/spect_train.csv', index=False)


def load_spect_test_table() -> pd.DataFrame:
    """ returns the spect test data
        Test data can be sorted by class since it only affects rule induction
    """
    df = pd.read_csv(f'data/SPECT.test', header=None)
    # assign column names
    columns = [f'F{num}' for num in range(1, 23)]
    columns.insert(0, 'overall')
    df.columns = columns
    # move class column to last column
    del columns[0]
    columns.append('overall')
    df = df[columns]

    return df
