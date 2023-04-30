import pandas as pd
import torchtext
from torchtext import data
from Tokenize import tokenize
from Batch import MyIterator, batch_size_fn
import os
import dill as pickle
import json


def readMethodData(opt):
    if opt.method_data is not None:
        try:
            with open(opt.method_data, encoding='utf8') as f:
                tmpData = json.load(f)
                src = [",".join([" ".join(t) for t in x[0]]) for x in tmpData]
                dst = [",".join([" ".join(t) for t in x[1]]) for x in tmpData]
                opt.src_data = src
                opt.trg_data = dst
        except:
            print("Fail to load data")

    if opt.test_data is not None:
        try:
            with open(opt.test_data, encoding='utf8') as f:
                tmpData = json.load(f)
                src = [",".join([" ".join(t) for t in x[0]]) for x in tmpData]
                dst = [",".join([" ".join(t) for t in x[1]]) for x in tmpData]
                opt.src_data_test = src
                opt.trg_data_test = dst
        except:
            print("Fail to load data")

def readTestData(opt):
    if opt.test_data is not None:
        try:
            with open(opt.test_data, encoding='utf8') as f:
                tmpData = json.load(f)
                src = [",".join([" ".join(t) for t in x[0]]) for x in tmpData]
                dst = [",".join([" ".join(t) for t in x[1]]) for x in tmpData]
                opt.src_data_test = src
                opt.trg_data_test = dst
        except:
            print("Fail to load data")

def readEvaluateData(opt):
    if opt.test_data is not None:
        try:
            with open(opt.test_data, encoding='utf8') as f:
                tmpData = json.load(f)
                rawTestData = tmpData
                tags = [x for x in tmpData]
                src = [",".join([" ".join(t) for t in tmpData[x][0]]) for x in tmpData]
                dst = [",".join([" ".join(t) for t in tmpData[x][1]]) for x in tmpData]
                opt.src_data_test = src
                opt.trg_data_test = dst
                opt.classnames = tags
        except:
            print("Fail to load data")
    if opt.FR_data is not None:
        try:
            with open(opt.FR_data, encoding='utf8') as f:
                tmpData = json.load(f)
                opt.FR_data = tmpData
        except:
            print("Fail to load data")
    if opt.field_dict is not None:
        try:
            with open(opt.field_dict, 'rb') as f:
                tmpData = pickle.load(f)
                opt.field_dict = tmpData
            opt.field_cnt = []
            for curCls in rawTestData:
                if curCls in opt.field_dict:
                    opt.field_cnt.append(opt.field_dict[curCls].__len__())
                else:
                    opt.field_cnt.append(0)

        except:
            print("Fail to load data")
def read_data(opt):
    if opt.src_data is not None:
        try:
            opt.src_data = open(opt.src_data, encoding='utf8').read().strip().split('\n')
        except:
            print("error: '" + opt.src_data + "' file not found")
            quit()

    if opt.trg_data is not None:
        try:
            opt.trg_data = open(opt.trg_data, encoding='utf8').read().strip().split('\n')
        except:
            print("error: '" + opt.trg_data + "' file not found")
            quit()

def create_fields(opt):
    
    spacy_langs = ['en', 'fr', 'de', 'es', 'pt', 'it', 'nl', 'en_core_web_sm']
    if opt.src_lang not in spacy_langs:
        print('invalid src language: ' + opt.src_lang + 'supported languages : ' + spacy_langs)  
    if opt.trg_lang not in spacy_langs:
        print('invalid trg language: ' + opt.trg_lang + 'supported languages : ' + spacy_langs)
    
    print("loading spacy tokenizers...")
    
    t_src = tokenize(opt.src_lang)
    t_trg = tokenize(opt.trg_lang)

    TRG = data.Field(lower=True, tokenize=t_trg.tokenizer, init_token='<sos>', eos_token='<eos>')
    SRC = data.Field(lower=True, tokenize=t_src.tokenizer)

    if opt.load_weights is not None:
        try:
            print("loading presaved fields...")
            SRC = pickle.load(open(f'{opt.load_weights}/SRC.pkl', 'rb'))
            TRG = pickle.load(open(f'{opt.load_weights}/TRG.pkl', 'rb'))
        except:
            print("error opening SRC.pkl and TXT.pkl field files, please ensure they are in " + opt.load_weights + "/")
            quit()
        
    return(SRC, TRG)

def create_dataset(opt, SRC, TRG):

    print("creating dataset and iterator... ")

    # ------------ Build training data ------------
    raw_data = {'src' : [line for line in opt.src_data], 'trg': [line for line in opt.trg_data]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])
    
    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)
    
    data_fields = [('src', SRC), ('trg', TRG)]
    train = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    train_iter = MyIterator(train, batch_size=opt.batchsize, device=opt.device,
                        repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                        batch_size_fn=batch_size_fn, train=True, shuffle=True)

    os.remove('translate_transformer_temp.csv')
    # ------------ ------END------ ------------

    # ------------ Build test data ------------
    raw_data = {'src' : [line for line in opt.src_data_test], 'trg': [line for line in opt.trg_data_test]}
    df = pd.DataFrame(raw_data, columns=["src", "trg"])

    mask = (df['src'].str.count(' ') < opt.max_strlen) & (df['trg'].str.count(' ') < opt.max_strlen)
    df = df.loc[mask]

    df.to_csv("translate_transformer_temp.csv", index=False)

    data_fields = [('src', SRC), ('trg', TRG)]
    test = data.TabularDataset('./translate_transformer_temp.csv', format='csv', fields=data_fields)

    test_iter = MyIterator(test, batch_size=opt.batchsize, device=opt.device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True, shuffle=True)

    os.remove('translate_transformer_temp.csv')
    # ------------ ------END------ ------------

    if opt.load_weights is None:
        SRC.build_vocab(train)
        TRG.build_vocab(train)
        if opt.checkpoint > 0:
            try:
                os.mkdir(opt.output_path)
            except:
                print("weights folder already exists, run program with -load_weights weights to load them")
                quit()
            pickle.dump(SRC, open(opt.output_path + '/SRC.pkl', 'wb'))
            pickle.dump(TRG, open(opt.output_path + '/TRG.pkl', 'wb'))

    opt.src_pad = SRC.vocab.stoi['<pad>']
    opt.trg_pad = TRG.vocab.stoi['<pad>']

    opt.train_len = get_len(train_iter)
    opt.test_len = get_len(test_iter)
    return train_iter, test_iter

def get_len(train):

    for i, b in enumerate(train):
        pass
    
    return i
