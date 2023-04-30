import json
import os
import pickle
import sys
from collections import defaultdict
import re
import numpy as np
from tqdm import tqdm
from typing import List


handledIdentifier = {}
def camel_case_split(identifier):
    if identifier in handledIdentifier:
        return handledIdentifier[identifier]
    temp = re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', identifier)).strip().split()
    tmpRes = [x.lower() for x in temp if x!=""]
    handledIdentifier[identifier] = tmpRes
    return tmpRes


def constrcuGraph(fieldIrMethods, fileGraph=None):
    if fileGraph is None:
        fileGraph = defaultdict(list)
    for curCls in fieldIrMethods:
        tag = camel_case_split(os.path.splitext(os.path.basename(curCls))[0])[-1]
        fileGraph[tag].append(curCls)
    return fileGraph


def listRfind(listA:List, targetStr):

    for i in range(listA.__len__()):
        curIdx = 0 - i - 1
        if listA[curIdx] == targetStr:
            return listA.__len__() + curIdx
    return -1

def cutoffSubtokens(methods, maxLen=300):
    subtokens = []
    for x in methods:
        tmpSb = []
        for t in x:
            tmpSb += t.split('_')
        x = [t for t in tmpSb if t != '']
        subtokens += x
        subtokens.append('|')
    subtokens = subtokens[:maxLen]
    lastSpliter = listRfind(subtokens, '|')
    afterSplited = subtokens[:lastSpliter]
    res, last = [], []
    for x in afterSplited:
        if x != '|':
            last.append(x)
        elif x == '|':
            res.append(last)
            last = []
    res.append(last)
    return res

def evaluateLength(methods, maxLen=300):
    subtokens = []
    for x in methods:
        subtokens += x
        subtokens.append('|')
    subtokens = subtokens[:maxLen]
    lastSpliter = listRfind(subtokens, '|')
    afterSplited = subtokens[:lastSpliter]
    if afterSplited.__len__() < maxLen:
        return True
    else:
        return False

def calCosineSim(vec_1, vec_2):
    return float(np.dot(vec_1, vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2)))

def getSimilarity(pathA, pathB, vectorDict):
    if pathA not in vectorDict or pathB not in vectorDict:
        return 0
    return calCosineSim(vectorDict[pathA], vectorDict[pathB])


def calJaccard(set_1, set_2, tokenize=False):
    if tokenize is False:
        unionSet = set_1 | set_2
        if unionSet.__len__() == 0 or set_1.__len__() == 0:
            return 0
        return (set_1 & set_2).__len__() / unionSet.__len__()
    else:
        tmpSet1 = set()
        tmpSet2 = set()
        for ele in set_1:
            for x in camel_case_split(ele):
                tmpSet1.add(x)
        for ele in set_2:
            for x in camel_case_split(ele):
                tmpSet2.add(x)
        unionSet = tmpSet1 | tmpSet2
        if unionSet.__len__() == 0 or tmpSet1.__len__() == 0:
            return 0
        return (tmpSet1 & tmpSet2).__len__() / unionSet.__len__()


def calPrecision(oracle_set, pred_set, tokenize=False):
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
        if unionSet.__len__() == 0 or tmpSet1.__len__() == 0:
            return 0
        return (tmpSet1 & tmpSet2).__len__() / tmpSet2.__len__()

def calRecall(oracle_set, pred_set, tokenize=False):
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

def f_score(precison: float, recall: float) -> float:
    return 2*precison*recall/(precison + recall)


def evaluate(res):
    precision_M_Rec, precision_T_Rec = [], []
    recall_M_Rec, recall_T_Rec = [], []
    FScore_M_Rec, FScore_T_Rec = [], []
    validPredCnt = 0

    for curCls in res:
        oracle, pred = set(res[curCls][0]), set(res[curCls][1])
        if pred.__len__() != 0:
            validPredCnt += 1
        else:
            continue
        precision_M, precision_T = calPrecision(oracle, pred), calPrecision(oracle, pred, tokenize=True)
        recall_M, recall_T = calRecall(oracle, pred), calRecall(oracle, pred, tokenize=True)
        precision_M_Rec.append(precision_M)
        precision_T_Rec.append(precision_T)
        recall_M_Rec.append(recall_M)
        recall_T_Rec.append(recall_T)
        if sum([precision_M, recall_M]) == 0:
            FScore_M_Rec.append(0)
        else:
            FScore_M_Rec.append(f_score(precision_M_Rec[-1], recall_M_Rec[-1]))
        if sum([precision_T, recall_T]) == 0:
            FScore_T_Rec.append(0)
        else:
            FScore_T_Rec.append(f_score(precision_T_Rec[-1], recall_T_Rec[-1]))

    print("Recall: ", validPredCnt/res.__len__(), end='\n')
    print("Precision#M: ", np.mean(precision_M_Rec), end='\n')
    print("Recall#M: ", np.mean(recall_M_Rec), end='\n')
    print("F-score#M: ", np.mean(FScore_M_Rec), end='\n')
    print("Precision#T: ", np.mean(precision_T_Rec), end='\n')
    print("Recall#T: ", np.mean(recall_T_Rec), end='\n')
    print("F-score#T: ", np.mean(FScore_T_Rec), end='\n')



def main(testFieldMethods, baseFieldMethods, vectors, outputPath):

    fileGraph = constrcuGraph(baseFieldMethods)
    res = {}
    competitiveNum = 8
    for i, fileHook in enumerate(tqdm(fileGraph)):
        for curIdx, targetCls in enumerate(fileGraph[fileHook]):
            if targetCls not in testFieldMethods:
                continue
            targetRepo = targetCls.split('/')[0]
            simList = []
            countMethods = defaultdict(int)
            mostCompetitiveMethods = []
            for ctxIdx, ctxCls in enumerate(fileGraph[fileHook]):
                if ctxIdx == curIdx or ctxCls not in baseFieldMethods or 'main' in os.path.basename(ctxCls).lower() or\
                        (ctxCls in testFieldMethods and targetRepo != ctxCls.split('/')[0]):  # For the "context" classes in test dir, only it's from the same repo with the target class.
                    continue
                curSimilarity = getSimilarity(targetCls, ctxCls, vectors)
                if curSimilarity > threshold:
                    simList.append((curSimilarity, baseFieldMethods[ctxCls]))
            simList.sort(key=lambda x: x[0], reverse=True)
            ctxMethods = []
            for sim, curMethods in simList:
                ctxMethods += curMethods
            for sim, curMethods in simList[:competitiveNum]:
                mostCompetitiveMethods += curMethods
            for x in ctxMethods:
                countMethods[x] += 1
            mostCompetitiveMethods = list(set(mostCompetitiveMethods))
            mostCompetitiveMethods.sort(key=lambda x: countMethods[x], reverse=True)
            mostCompetitiveMethods = mostCompetitiveMethods[:8]  # Only choose top 8 methods.
            res[targetCls] = [list(set(testFieldMethods[targetCls])), mostCompetitiveMethods]

    evaluate(res)
    with open(outputPath, 'w') as f:
        json.dump(res, f, indent=2)


def combimeMethods(fieldIrMethods, fieldReMethods):
    for curCls in fieldIrMethods:
        if curCls in fieldReMethods:
            fieldIrMethods[curCls] += fieldReMethods[curCls]
    return fieldIrMethods


'''
This baseline always recommends all methods from the most similar class.
'''
if __name__ == '__main__':

    testFieldIrMethodsPath = '../Data/testFieldIrrelevantMethods.pkl'  # field-irrelevant methods from test dir in java-large repository
    baseFieldIrMethodsPath = '../Data/baseFieldIrrelevantMethods.pkl'  # field-irrelevant methods from training and validation dir in java-large repository
    testFieldReMethodsPath = '../Data/testFieldRelevantMethods.pkl'  # field-irrelevant methods from test dir in java-large repository
    baseFieldReMethodsPath = '../Data/baseFieldRelevantMethods.pkl'  # field-irrelevant methods from training and validation dir in java-large repository
    vectorsPath = '../Data/vectorsForJavaLarge.pkl'
    outputPath = '../Data/Baseline#2.json'

    if len(sys.argv) > 1:
        threshold = float(sys.argv[1])
    else:
        threshold = 0.9
        
    with open(testFieldIrMethodsPath, 'rb') as f_1, open(baseFieldIrMethodsPath, 'rb') as f_2, \
            open(testFieldReMethodsPath, 'rb') as f_3, open(baseFieldReMethodsPath, 'rb') as f_4,   open(vectorsPath, 'rb') as f_5:
        testFieldIrMethods = pickle.load(f_1)
        baseFieldIrMethods = pickle.load(f_2)
        testFieldReMethods = pickle.load(f_3)
        baseFieldReMethods = pickle.load(f_4)
        vectors = pickle.load(f_5)

    baseFieldIrMethods.update(testFieldIrMethods)
    baseFieldReMethods.update(testFieldReMethods)
    testFieldMethods = combimeMethods(testFieldIrMethods, testFieldReMethods)
    baseFieldMethods = combimeMethods(baseFieldIrMethods, baseFieldReMethods)

    main(testFieldMethods, baseFieldMethods, vectors, outputPath)
