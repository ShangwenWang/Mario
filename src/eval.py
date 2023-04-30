import re
from typing import List

def camel_case_split(identifier:str) -> List[str]:
    temp = re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', identifier)).strip().split()
    tmpRes = [x.lower() for x in temp if x!=""]
    return tmpRes

def f_score(precison: float, recall: float) -> float:
    return 2*precison*recall/(precison + recall)

def calRecall(oracle_set, pred_set, tokenize=False):
    if isinstance(oracle_set, list):
        oracle_set = set(oracle_set)
    if isinstance(pred_set, list):
        pred_set = set(pred_set)

    if tokenize is False:
        unionSet = oracle_set | pred_set
        if unionSet.__len__() == 0 or oracle_set.__len__() == 0:
            return 0
        return (oracle_set & pred_set).__len__() / oracle_set.__len__()
    else:
        tmpSet1 = set()
        tmpSet2 = set()
        for ele in oracle_set:
            for x in camel_case_split(ele):
                tmpSet1.add(x)
        for ele in pred_set:
            for x in camel_case_split(ele):
                tmpSet2.add(x)
        unionSet = tmpSet1 | tmpSet2
        if unionSet.__len__() == 0 or tmpSet1.__len__() == 0:
            return 0
        return (tmpSet1 & tmpSet2).__len__() / tmpSet1.__len__()


def calPrecision(oracle_set, pred_set, tokenize=False):
    if isinstance(oracle_set, list):
        oracle_set = set(oracle_set)
    if isinstance(pred_set, list):
        pred_set = set(pred_set)
    if tokenize is False:
        unionSet = oracle_set | pred_set
        if unionSet.__len__() == 0 or oracle_set.__len__() == 0:
            return 0
        return (oracle_set & pred_set).__len__() / pred_set.__len__()
    else:
        tmpSet1 = set()
        tmpSet2 = set()
        for ele in oracle_set:
            for x in camel_case_split(ele):
                tmpSet1.add(x)
        for ele in pred_set:
            for x in camel_case_split(ele):
                tmpSet2.add(x)
        unionSet = tmpSet1 | tmpSet2
        if unionSet.__len__() == 0 or tmpSet1.__len__() == 0 or tmpSet2.__len__() == 0:
            return 0
        return (tmpSet1 & tmpSet2).__len__() / tmpSet2.__len__()


def evaluate_beam(opt, model):
    pass