#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" test.py
Authors: Jakub Ciemięga, Klaudia Stpiczyńska
"""
from data_manager import create_test_table
from aq_algorithm import covers, delete_non_general


def test_covers() -> bool:
    """ test the covers() function from aq_algorithm.py. Returns True if function correct
    """
    df = create_test_table()
    rule = {'temp': ['hot', 'cool'], 'windy': ['TRUE']}
    if covers(rule, df.loc[0]):
        return False
    if not covers(rule, df.loc[1]):
        return False
    if covers(rule, df.loc[2]):
        return False

    rule = {}
    if not covers(rule, df.loc[3]):
        return False
    if not covers(rule, df.loc[4]):
        return False
    if not covers(rule, df.loc[5]):
        return False

    return True


def test_delete_non_general() -> bool:
    """ test the delete_non_general() function from aq_algorithm.py. Returns True if function correct
    """
    complexes = [{'key1': ['val1']}, {}]
    complexes = delete_non_general(complexes)
    if complexes != [{}]:
        return False

    complexes = [{'key1': ['val1']}, {'key1': ['val1']}]
    complexes = delete_non_general(complexes)
    if complexes != [{'key1': ['val1']}]:
        return False

    complexes = [{'key1': ['val1', 'val2']}, {'key1': ['val1']}]
    complexes = delete_non_general(complexes)
    if complexes != [{'key1': ['val1', 'val2']}]:
        return False

    complexes = [{'key1': ['val1', 'val2'], 'key2': ['val1']}, {'key1': ['val2'], 'key2': ['val1', 'val2']}]
    complexes = delete_non_general(complexes)
    if complexes != [{'key1': ['val1', 'val2'], 'key2': ['val1']}, {'key1': ['val2'], 'key2': ['val1', 'val2']}]:
        return False

    return True
