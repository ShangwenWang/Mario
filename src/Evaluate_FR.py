'''
Evaluator for the field-relevant method which can't find same field in proximate classes.
'''
import json
import os
import pickle
from gensim.models import FastText
from tqdm import tqdm
from typing import List, Dict
import fnmatch
import wordninja
import re
import javalang
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
import numpy as np


processedIdentifier = {}
def camel_case_split(identifier):
    if identifier in processedIdentifier:
        return processedIdentifier[identifier]
    else:
        temp = re.sub(r'([A-Z][a-z])', r' \1', re.sub(r'([A-Z]+)', r' \1', identifier)).replace('_', ' ').strip().split()
        tmpRes = [x.lower() for x in temp if x!=""]
        processedIdentifier[identifier] = tmpRes
    return tmpRes

class FileNode(object):
    def __init__(self, readablePath, path: str):
        self.readablePath = readablePath
        self.path = path.replace('\\', '/')
        tmp = self.path.split('/')
        self.classPath = list(wordninja.split(" ".join(tmp[-4:-1])))
        self.filename = list(camel_case_split(tmp[-1][:-5]))  # Use [:-5] to remove '.java'
        self.similarity = []
        if self.filename.__len__() >= 1:
            self.tag = self.filename[-1]
        else:
            self.tag = ''
        self.pathVec = None
        self.nameVec = None
        self.finalVec = None
        self.valid = None
        self.fields = None
        self.AST = None
        self.methodNames = None
        self.FIRMethods = None
        self.FReMethods = None
        self.source = None
        self.methodSubtokens = None
        self.fieldMethodMapping = None
        self.FReFields = None

    def updateVec(self, bertModel, fasttextModel):
        if self.filename.__len__() == 1 or 'main' in self.filename or 'Main' in self.filename:
            return False
        self.nameVec = np.mean([fasttextModel.wv[x] for x in self.filename[:-1]], axis=0)
        self.pathVec = np.mean([fasttextModel.wv[x] for x in self.classPath], axis=0)
        self.finalVec = np.concatenate([self.pathVec, self.nameVec])

    def updateClassVec(self, vec):
        self.classVec = vec

    def updateNameVec(self, vec):
        self.nameVec = vec

    def getSubtokens(self):
        if self.methodSubtokens is not None:
            return self.methodSubtokens
        self.methodSubtokens = set()
        for mn in self.getCleanedMethods():
            for st in camel_case_split(mn):
                self.methodSubtokens.add(st)
        return self.methodSubtokens


    def getAST(self):
        if self.AST is not None:
            return self.AST
        if self.path in ASTDict:
            self.AST = ASTDict[self.path]
            return self.AST
        with open(self.readablePath, 'r', encoding='utf8') as f:
            try:
                tmpFile = f.read()
                self.source = tmpFile
                AST = javalang.parse.parse(tmpFile)
                self.AST = AST.types[0]
            except:
                self.valid = False
                self.AST = False
                return False
        return self.AST

    def isValid(self):
        if self.valid is not None:
            return self.valid
        AST = self.getAST()
        if AST is False:
            self.valid = False
        try:
            if isinstance(AST, javalang.tree.InterfaceDeclaration) or (hasattr(AST, 'implements') and AST.implements is not None and AST.implements.__len__() > 0):
                self.valid = False
            else:
                self.valid = True
        except:
            print("Encounter Error when generating AST in " + self.readablePath)
            self.valid = False
        return self.valid

    def getFields(self):
        if self.fields is not None:
            return self.fields
        self.fields = getClassFiled(self.getAST())
        return self.fields

    def getMethodNames(self):
        if self.methodNames is None:
            if self.AST is None:
                self.getAST()
            self.methodNames = [x.name for x in getAllMethod(self.AST.body)]
        return self.methodNames

    def getFIRMethods(self):
        if self.FIRMethods is not None:
            return self.FIRMethods
        fields = set([x.lower() for x in self.getFields()])
        self.FIRMethods = []
        for mn in self.getMethodNames():
            splitedMethodName = camel_case_split(mn)
            if splitedMethodName[0] in verbVocab and "".join(splitedMethodName[1:]).lower() in fields:
                self.FReMethods.append(mn)  # field related method
            else:
                self.FIRMethods.append(mn)
        return self.FIRMethods

    def getFReMethods(self):
        if self.FReMethods is not None:
            return self.FReMethods
        fields = set([x.lower() for x in self.getFields()])
        self.FReMethods = []
        self.FIRMethods = []
        self.FReFields = set()
        self.fieldMethodMapping = {}
        for mn in self.getMethodNames():
            splitedMethodName = camel_case_split(mn)
            if splitedMethodName.__len__() < 2:
                continue
            if splitedMethodName[0] in verbVocab and "".join(splitedMethodName[1:]).lower() in fields:
                self.FReMethods.append(mn)  # field related method
                rawField = splitedMethodName[1] + "".join([x.capitalize() for x in splitedMethodName[2:]])
                self.fieldMethodMapping[rawField] = mn
                self.FReFields.add(rawField)
            else:
                self.FIRMethods.append(mn)
        return self.FReMethods

    def getFieldMethodMapping(self):
        if self.fieldMethodMapping is not None:
            return self.fieldMethodMapping
        else:
            self.getFReMethods()
            return self.fieldMethodMapping

    def getFRFields(self):
        if self.FReFields is not None:
            return self.FReFields
        else:
            self.getFReMethods()
            return self.FReFields


def getClassFiled(AST_ClassDeclaration):
    fields = []
    for x in AST_ClassDeclaration.fields:
        fields.append(x.declarators[0].name)
    return fields

def getAllMethod(classNode):
    methods = []
    for x in classNode:
        if isinstance(x, consideredType):
            methods.append(x)
        elif isinstance(x, javalang.tree.ClassDeclaration):
            methods += getAllMethod(x.body)
    return methods

def travFolder(dir, files=[], suffix='') -> List[str]:
    listdirs = os.listdir(dir)
    for f in listdirs:
        pattern = '*.' + suffix if suffix != '' else '*.*'
        if os.path.isfile(os.path.join(dir, f)) and fnmatch.fnmatch(f, pattern) \
                and 'test' not in os.path.join(dir, f) and 'src' in os.path.join(dir, f):
            files.append(os.path.join(dir, f))
        elif os.path.isdir(os.path.join(dir, f)):
            travFolder(dir + '/' + f, files, suffix)
    return files

def constrcuGraph(fieldIrMethods):
    fileGraph = defaultdict(list)
    for curCls in fieldIrMethods:
        tag = camel_case_split(os.path.splitext(os.path.basename(curCls))[0])[-1]
        fileGraph[tag].append(curCls)
    return fileGraph


def getVerbVocab(verbPath):
    with open(verbPath, 'r', encoding='utf8') as f:
        tmp = f.readlines()
        return set([x.strip('\n').strip("\"") for x in tmp])

def calCosineSim(vec_1, vec_2):
    return float(np.dot(vec_1, vec_2)/(np.linalg.norm(vec_1)*np.linalg.norm(vec_2)))

def constructGraph(repoPath, fasttestModel, nodeGraph=None, testDir=False):
    if nodeGraph is None:
        nodeGraph = defaultdict(list)
    allJavaFiles = travFolder(repoPath, [], 'java')
    curRepoDir = repoPath.replace('\\', '/')
    for filePath in allJavaFiles:
        rawPath = filePath
        filePath = filePath.replace('\\', '/').replace(curRepoDir, '').strip('/')
        curNode = FileNode(rawPath, filePath)
        if curNode.filename.__len__() == 0:
            continue
        curNode.updateVec(None, fasttestModel)
        nodeGraph[curNode.tag].append(curNode)
        nodeDict[curNode.path] = curNode
        if testDir is True:
            clsInTest.add(curNode.path)
        else:
            clsInBase.add(curNode.path)
    return nodeGraph


def getFieldMethodMapping(fieldMethodMapping, curClass):
    if curClass in fieldMethodMapping:
        return fieldMethodMapping[curClass]
    else:
        return []

def diffFieldPredictor(curFields, probs, threshold=0.5):
    fieldProbs = []
    predMethods = []
    for field in curFields:
        if field.upper() == field:
            continue
        fieldTag = camel_case_split(field)[-1].lower()
        fieldTagLematized = wnl.lemmatize(fieldTag, 'n')
        verbs = []
        if fieldTag in probs:
            verbs = [x for x in probs[fieldTag] if probs[fieldTag][x] >= threshold and x != "total"]
        elif fieldTagLematized in probs:
            verbs = [x for x in probs[fieldTagLematized] if probs[fieldTagLematized][x] >= threshold and x != "total"]
        for verb in verbs:
            if field.__len__() > 1:
                predMethods.append(verb + field[0].upper() + field[1:])
            else:
                predMethods.append(verb + field.capitalize())
    return predMethods

def predictorChooser(classesWithSameTag: List[FileNode], targetCls:FileNode, fieldDict:Dict[str, List[str]], fieldMethodMapping:Dict[str, any], threshold:float):
    def getFields(fieldDict, cls):
        if cls in fieldDict:
            return fieldDict[cls]
        else:
            return []
    targetFields = set([x for x in getFields(fieldDict, targetCls.path) if x.upper() != x])
    fieldsInCtx = set()
    FRMethodsMapping = defaultdict(list)
    targetRepo = targetCls.path.split('/')[0]
    for ctxCls in classesWithSameTag:
        if ctxCls.path == targetCls.path or ctxCls.finalVec is None \
                or calCosineSim(ctxCls.finalVec, targetCls.finalVec) < threshold:
            continue
        if 'main' in os.path.basename(ctxCls.path).lower() or \
                (ctxCls.path in clsInTest and targetRepo != ctxCls.path.split('/')[0]):
            continue
        curFRMapping = getFieldMethodMapping(fieldMethodMapping, ctxCls.path)
        for field in curFRMapping:
            if field.upper() != field:
                FRMethodsMapping[field].append(curFRMapping[field])
                fieldsInCtx.add(field)
    for field in FRMethodsMapping:
        tmpMethods = FRMethodsMapping[field]
        tmpCounter = defaultdict(int)
        for method in tmpMethods:
            tmpCounter[method] += 1
        tmpRes = sorted(tmpCounter.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)
        FRMethodsMapping[field] = tmpRes[0][0]
    sameField = targetFields & fieldsInCtx
    failedField = targetFields - sameField
    predMethods = [FRMethodsMapping[x] for x in sameField]
    return predMethods, failedField

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
            
    print("Threshold: ", threshold)
    print("Recall: ", validPredCnt/res.__len__(), end='\n')
    print("Precision#M: ", np.mean(precision_M_Rec), end='\n')
    print("Recall#M: ", np.mean(recall_M_Rec), end='\n')
    print("F-score#M: ", np.mean(FScore_M_Rec), end='\n')
    print("Precision#T: ", np.mean(precision_T_Rec), end='\n')
    print("Recall#T: ", np.mean(recall_T_Rec), end='\n')
    print("F-score#T: ", np.mean(FScore_T_Rec), end='\n')


def main(FRMethods, fieldDict, fieldMethodMapping, probs, targetReposPath, baseReposPath, outputPath, threshold, fasttextModel, FieldFinderThreshold):
    nodeGraph = constructGraph(targetReposPath, fasttextModel, testDir=True)
    nodeGraph = constructGraph(baseReposPath, fasttextModel, nodeGraph)
    res = {}
    for cls in tqdm(FRMethods):
        curFRMethods = list(set(FRMethods[cls]))
        curFields = fieldDict[cls]
        targetFileNode = FileNode(readablePath=os.path.join(targetReposPath, cls), path=cls)
        targetFileNode.updateVec(
            None, fasttextModel
        )
        if targetFileNode.path not in fieldMethodMapping or targetFileNode.finalVec is None:
            continue
        if nodeGraph[targetFileNode.tag].__len__() != 0:  # test
            predMethods_1, failedField = predictorChooser(nodeGraph[targetFileNode.tag], targetFileNode, fieldDict, fieldMethodMapping, threshold)
        else:
            continue
        predMethods_2 = diffFieldPredictor(curFields, probs, threshold=FieldFinderThreshold)
        res[cls] = [curFRMethods, list(set(predMethods_1 + predMethods_2))]
        
    evaluate(res)
    with open(outputPath, 'w') as f:
        json.dump(res, f, indent=2)


if __name__ == '__main__':
    fieldDictPath = '../Data/fieldDict.pkl'
    fieldMethodMappingPath = '../Data/fieldMethodMappingDict.pkl'
    testFieldReMethodsPath = '../Data/testFieldRelevantMethods.pkl'
    taragetReposPath = '../JavaRepos_all/java-large/target'
    baseReposPath = '../JavaRepos_all/java-large/training'
    probabilityPath = '../Data/probability.json'
    verbPath = '../Data/filtered_verb_vocabulary.json'
    fasttextModelPath = '../Similarity/pathModel.bin'
    verbVocab = getVerbVocab(verbPath)
    nodeDict = {}
    ASTDict = {}
    clsInBase = set()
    clsInTest = set()
    threshold = 0.7
    FieldFinderThreshold = 0.2
    outputPath = '../Data/FR_PredRes' + str(threshold)+'.json'

    fasttextModel = FastText.load(fasttextModelPath)
    consideredType = (javalang.tree.MethodDeclaration, javalang.tree.ConstructorDeclaration)
    with open(probabilityPath, 'r') as f, open(testFieldReMethodsPath, 'rb') as f_2, open(fieldDictPath, 'rb') as f_3, \
        open(fieldMethodMappingPath, 'rb') as f_4:
        probs = json.load(f)
        FRMethods = pickle.load(f_2)
        fieldDict = pickle.load(f_3)
        fieldMethodMapping = pickle.load(f_4)

    main(FRMethods, fieldDict, fieldMethodMapping, probs, taragetReposPath, baseReposPath, outputPath, threshold, fasttextModel, FieldFinderThreshold)