import random

import torch
from Models import get_model
from Process import *
import argparse
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
import re
from eval import *
from tqdm import tqdm
import numpy as np
from collections import defaultdict

def get_synonym(word, SRC):
    syns = wordnet.synsets(word)
    for s in syns:
        for l in s.lemmas():
            if SRC.vocab.stoi[l.name()] != 0:
                return SRC.vocab.stoi[l.name()]
    return 0

def multiple_replace(dict, text):
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, dict.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: dict[mo.string[mo.start():mo.end()]], text)


def translate_sentence(sentence, model, opt, SRC, TRG, index):

    model.eval()
    indexed = []
    rawSentence = sentence
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device.type == 'cuda':
        sentence = sentence.to(opt.device)

    sentence = beam_search(sentence, model, SRC, TRG, opt)
    sentence_deduplicated = set([x for x in sentence.strip('.').split(',') if x.strip() != ''])
    src_loc = defaultdict(lambda :100)
    for i, src in enumerate(rawSentence.strip('.').split(',')):
        src_loc[src] = i
    deduplicatedRawSentence = set(rawSentence.strip('.').split(','))
    unappearenceSentence = deduplicatedRawSentence - sentence_deduplicated
    offset = opt.trg_data_test[index].count(',') + 3 - sentence_deduplicated.__len__()
    unappearenceSentence = list(unappearenceSentence)
    unappearenceSentence.sort(key=lambda x:src_loc[x])
    sentence_deduplicated = list(sentence_deduplicated)
    sentence_deduplicated.sort(key=lambda x:src_loc[x])
    if offset >= 0:
        predSentence = sentence_deduplicated + unappearenceSentence[:offset]
    else:
        predSentence = sentence_deduplicated[:offset]
    predSentence = ",".join(predSentence) + '.'
    return multiple_replace({' ?' : '?',' !':'!',' .':'.','\' ':'\'',' ,':','}, predSentence)

def translate(opt, model, SRC, TRG):
    sentences = opt.src_data_test
    translated = []

    print("Predicting...")
    for i, sentence in enumerate(tqdm(sentences)):
        translated.append(translate_sentence(sentence + '.', model, opt, SRC, TRG, i).capitalize())

    return translated

def findCases(index, oracle, pred):
    recall_M = calRecall(oracle, pred)
    precision_M = calPrecision(oracle, pred)
    if recall_M == 1.0 and precision_M == 1.0:
        print("Same ", index)
        print(oracle)
        print(pred)
    elif recall_M == 1.0 and precision_M < 1.0:
        print("more", index)
        print(oracle)
        print(pred)
    elif recall_M < 1.0 and precision_M == 1.0:
        print("less", index)
        print(oracle)
        print(pred)
    elif 0.5 < recall_M < 1.0 and 0.5 < precision_M < 1.0:
        print("some mistakes", index)
        print(oracle)
        print(pred)
    else:
        pass
    
def evaluate(opt, prediction):
    
    precision_M_Rec, precision_T_Rec = [], []
    recall_M_Rec, recall_T_Rec = [], []
    FScore_M_Rec, FScore_T_Rec = [], []

    precision_M_FIR_Rec, precision_T_FIR_Rec = [], []
    recall_M_FIR_Rec, recall_T_FIR_Rec = [], []
    FScore_M_FIR_Rec, FScore_T_FIR_Rec = [], []

    for index in range(prediction.__len__()):
        rawIndex = index
        curCls = opt.classnames[index]
        FIROracle = ["".join([x.capitalize() for x in m.split(' ')]) for m in opt.trg_data_test[index].split(',')]
        FROracle = opt.FR_data[curCls][0] if curCls in opt.FR_data else []
        curOracle = set(FIROracle + FROracle)
        FIRPred = ["".join([x.capitalize() for x in m.split(' ')]) for m in prediction[rawIndex].strip('.').split(',')]
        FRPred = opt.FR_data[curCls][1] if curCls in opt.FR_data else []
        curPred = set(FIRPred + FRPred)
        precision_M, precision_T = calPrecision(curOracle, curPred), calPrecision(curOracle, curPred, tokenize=True)
        precision_M_FIR, precision_T_FIR = calPrecision(FIROracle, FIRPred), calPrecision(FIROracle, FIRPred, tokenize=True)
        recall_M, recall_T = calRecall(curOracle, curPred), calRecall(curOracle, curPred, tokenize=True)
        recall_M_FIR, recall_T_FIR = calRecall(FIROracle, FIRPred), calRecall(FIROracle, FIRPred, tokenize=True)
        precision_M_Rec.append(precision_M)
        precision_T_Rec.append(precision_T)
        recall_M_Rec.append(recall_M)
        recall_T_Rec.append(recall_T)

        precision_M_FIR_Rec.append(precision_M_FIR)
        precision_T_FIR_Rec.append(precision_T_FIR)
        recall_M_FIR_Rec.append(recall_M_FIR)
        recall_T_FIR_Rec.append(recall_T_FIR)
        findCases(index, curOracle, curPred)
        if sum([precision_M, recall_M]) == 0:
            FScore_M_Rec.append(0)
        else:
            FScore_M_Rec.append(f_score(precision_M_Rec[-1], recall_M_Rec[-1]))

        if sum([precision_M_FIR, recall_M_FIR]) == 0:
            FScore_M_FIR_Rec.append(0)
        else:
            FScore_M_FIR_Rec.append(f_score(precision_M_FIR_Rec[-1], recall_M_FIR_Rec[-1]))

        if sum([precision_T, recall_T]) == 0:
            FScore_T_Rec.append(0)
        else:
            FScore_T_Rec.append(f_score(precision_T_Rec[-1], recall_T_Rec[-1]))

        if sum([precision_T_FIR, recall_T_FIR]) == 0:
            FScore_T_FIR_Rec.append(0)
        else:
            FScore_T_FIR_Rec.append(f_score(precision_T_FIR_Rec[-1], recall_T_FIR_Rec[-1]))

    print("Test_data:", opt.test_data)
    print("Mario: ")
    print("Recall#C: ", FScore_T_FIR_Rec.__len__()/prediction.__len__())
    print("Precision#M: ", np.mean(precision_M_Rec), end='\n')
    print("Recall#M: ", np.mean(recall_M_Rec), end='\n')
    print("F-score#M: ", np.mean(FScore_M_Rec), end='\n')
    print("Precision#T: ", np.mean(precision_T_Rec), end='\n')
    print("Recall#T: ", np.mean(recall_T_Rec), end='\n')
    print("F-score#T: ", np.mean(FScore_T_Rec), end='\n\n')

    print("FIR: ")
    print("Precision#M: ", np.mean(precision_M_FIR_Rec), end='\n')
    print("Recall#M: ", np.mean(recall_M_FIR_Rec), end='\n')
    print("F-score#M: ", np.mean(FScore_M_FIR_Rec), end='\n')
    print("Precision#T: ", np.mean(precision_T_FIR_Rec), end='\n')
    print("Recall#T: ", np.mean(recall_T_FIR_Rec), end='\n')
    print("F-score#T: ", np.mean(FScore_T_FIR_Rec), end='\n\n')
    with open("./result/Fscore_T_" + opt.test_data.split('_')[1] + '.json', "w", encoding="utf8") as f:
        json.dump(FScore_T_Rec, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_data', type=str, default='./data/Evaluate.json')
    parser.add_argument('-FR_data', type=str, default='./data/FR_PredRes0.1.json')
    parser.add_argument('-field_dict', type=str, default='./data/fieldDict.pkl')
    parser.add_argument('-load_weights', type=str, default='weights')
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=80)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-src_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-trg_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument("--gpu", type=str, default="cuda:0", help="gpu")
    opt = parser.parse_args()

    opt.device = torch.device(opt.gpu if not opt.no_cuda and torch.cuda.is_available() else "cpu")
    readEvaluateData(opt)
    assert opt.k > 0
    assert opt.max_len > 10
    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    predictedMethodNames = translate(opt, model, SRC, TRG)
    evaluate(opt, predictedMethodNames)

if __name__ == '__main__':
    main()
