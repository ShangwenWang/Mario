import argparse
import time

import numpy as np
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import dill as pickle
from eval import evaluate_beam
import utils
import random
from tqdm import tqdm

def train_model(model, opt):
    
    print("training model...")
    model.train()
    start = time.time()
    best_test_loss = 1e5
    for epoch in range(opt.epochs):

        if opt.floyd is False:
            print("   %dm: epoch %d [%s]  %d%%  loss = %s" %\
            ((time.time() - start)//60, epoch + 1, "".join(' '*20), 0, '...'), end='\r')
        if opt.shuffle_target:
            shuffle_target_methods(opt)
        forward_model(opt, model, start, epoch, test=False)
        cur_test_loss = forward_model(opt, model, start, epoch, test=True)
        if cur_test_loss < best_test_loss:
            best_test_loss = cur_test_loss
            output_path = opt.output_path + '/' + "{:.3f}".format(cur_test_loss)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            torch.save(model.state_dict(), output_path + '/model_weights')
        if epoch and epoch % 10 == 0:
            evaluate_beam(opt, model)

def shuffle_target_methods(opt):
    '''
    Shuffle the target methods in training set.
    :param opt: Parameters
    :return: None
    '''
    testDataset = opt.train.dataset
    for example_i in tqdm(range(testDataset.examples.__len__())):
        if example_i == 0:
            continue
        example = testDataset.examples[example_i]
        target = example.trg
        indexes = utils.getMethodsInterval(target, ',')
        random.shuffle(indexes)
        shuffledTarget = []
        for i, index in enumerate(indexes):
            shuffledTarget += target[index[0]:index[1]]
            if i != indexes.__len__() - 1:
                shuffledTarget.append(',')
        example.trg = shuffledTarget

def forward_model(opt, model, startTime, epoch, test=False):
    total_loss = []
    cur_print_loop_loss = 0
    datset = opt.test if test else opt.train
    length = opt.test_len if test else opt.train_len
    model.train() if not test else model.eval()
    separatorIndex = datset.dataset.fields['trg'].vocab.stoi[',']
    if opt.checkpoint > 0:
        cptime = time.time()
    for i, batch in enumerate(datset):
        src = batch.src.transpose(0, 1)
        trg = batch.trg.transpose(0, 1)
        trg_input = trg[:, :-1]   # BatchSize * predLen
        src_mask, trg_mask = create_masks(src, trg_input, opt)
        preds = model(src, trg_input, src_mask, trg_mask)  # BatchSize * predLen * vocabSize
        ys = trg[:, 1:].contiguous().view(-1)  # remove the start symbol
        opt.optimizer.zero_grad() if test is False else None
        loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=opt.trg_pad)
        loss.backward() if test is False else None
        opt.optimizer.step() if test is False else None
        if opt.SGDR is True and test is False:
            opt.sched.step()
        cur_print_loop_loss += loss.item()
        total_loss.append(loss.item())

        # -----------  Release memory  -----------
        batch.src = None
        batch.trg = None
        batch = None
        trg_input = None
        # -----------  END  -----------
        if (i + 1) % opt.printevery == 0:
            p = int(100 * (i + 1) / length)
            cur_loop_avg_loss = cur_print_loop_loss / opt.printevery
            if opt.floyd is False and test is False:
                print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time.time() - startTime) // 60, epoch + 1, "".join('#' * (p // 5)),
                       "".join(' ' * (20 - (p // 5))), p, cur_loop_avg_loss), end='\r')
            elif opt.floyd is True and test is False:
                print("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time.time() - startTime) // 60, epoch + 1, "".join('#' * (p // 5)),
                       "".join(' ' * (20 - (p // 5))), p, cur_loop_avg_loss))
            elif opt.floyd is False and test is True:
                print("   %dm: [TEST] epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time.time() - startTime) // 60, epoch + 1, "".join('#' * (p // 5)),
                       "".join(' ' * (20 - (p // 5))), p, cur_loop_avg_loss), end='\r')
            elif opt.floyd is True and test is True:
                print("   %dm: [TEST] epoch %d [%s%s]  %d%%  loss = %.3f" % \
                      ((time.time() - startTime) // 60, epoch + 1, "".join('#' * (p // 5)),
                       "".join(' ' * (20 - (p // 5))), p, cur_loop_avg_loss))
            cur_print_loop_loss = 0

        if opt.checkpoint > 0 and ((time.time() - cptime) // 60) // opt.checkpoint >= 1 and test is False:
            torch.save(model.state_dict(), opt.output_path + '/model_weights')
            cptime = time.time()

    avg_loss = float(np.mean(total_loss))
    if test is False:
        print("%dm: epoch %d [%s%s]  %d%%  loss = %.3f\nepoch %d complete, loss = %.03f" % \
              ((time.time() - startTime) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100, avg_loss, epoch + 1, avg_loss))
    else:
        print("%dm: [TEST] epoch %d [%s%s]  %d%%  loss = %.3f\n[TEST] epoch %d complete, loss = %.03f" % \
              ((time.time() - startTime) // 60, epoch + 1, "".join('#' * (100 // 5)), "".join(' ' * (20 - (100 // 5))), 100, avg_loss, epoch + 1, avg_loss))
    return float(np.mean(total_loss))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-method_data', type=str, default="./data/Training.json")
    parser.add_argument('-test_data', type=str, default="./data/Test.json")
    parser.add_argument('-src_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-trg_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-shuffle_target', type=bool, default=False)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1500)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-load_weights', type=str, default=None)
    parser.add_argument('-output_path', type=str, default='weights')
    parser.add_argument('-create_valset', action='store_true')
    parser.add_argument('-max_strlen', type=int, default=80)
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-checkpoint', type=int, default=10)
    parser.add_argument("--gpu", type=str, default="cuda:1", help="gpu")
    opt = parser.parse_args()
    
    opt.device = torch.device(opt.gpu if not opt.no_cuda and torch.cuda.is_available() else "cpu")
    if opt.device.type == 'cuda':
        assert torch.cuda.is_available()
    
    # read_data(opt)
    readMethodData(opt)
    SRC, TRG = create_fields(opt)
    opt.train, opt.test = create_dataset(opt, SRC, TRG)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    if opt.checkpoint > 0:
        print("model weights will be saved every %d minutes and at end of epoch to directory weights/"%(opt.checkpoint))
    
    if opt.load_weights is not None and opt.floyd is not None:
        os.mkdir('weights')
        pickle.dump(SRC, open('weights/SRC.pkl', 'wb'))
        pickle.dump(TRG, open('weights/TRG.pkl', 'wb'))
    
    train_model(model, opt)

    if opt.floyd is False:
        promptNextAction(model, opt, SRC, TRG)

def yesno(response):
    while True:
        if response != 'y' and response != 'n':
            response = input('command not recognised, enter y or n : ')
        else:
            return response

def promptNextAction(model, opt, SRC, TRG):

    saved_once = 1 if opt.load_weights is not None or opt.checkpoint > 0 else 0
    
    if opt.load_weights is not None:
        dst = opt.load_weights
    if opt.checkpoint > 0:
        dst = opt.output_path

    while True:
        save = yesno(input('training complete, save results? [y/n] : '))
        if save == 'y':
            while True:
                if saved_once != 0:
                    res = yesno("save to same folder? [y/n] : ")
                    if res == 'y':
                        break
                dst = input('enter folder name to create for weights (no spaces) : ')
                if ' ' in dst or len(dst) < 1 or len(dst) > 30:
                    dst = input("name must not contain spaces and be between 1 and 30 characters length, enter again : ")
                else:
                    try:
                        os.mkdir(dst)
                    except:
                        res= yesno(input(dst + " already exists, use anyway? [y/n] : "))
                        if res == 'n':
                            continue
                    break
            
            print("saving weights to " + dst + "/...")
            torch.save(model.state_dict(), f'{dst}/model_weights')
            if saved_once == 0:
                pickle.dump(SRC, open(f'{dst}/SRC.pkl', 'wb'))
                pickle.dump(TRG, open(f'{dst}/TRG.pkl', 'wb'))
                saved_once = 1
            
            print("weights and field pickles saved to " + dst)

        res = yesno(input("train for more epochs? [y/n] : "))
        if res == 'y':
            while True:
                epochs = input("type number of epochs to train for : ")
                try:
                    epochs = int(epochs)
                except:
                    print("input not a number")
                    continue
                if epochs < 1:
                    print("epochs must be at least 1")
                    continue
                else:
                    break
            opt.epochs = epochs
            train_model(model, opt)
        else:
            print("exiting program...")
            break

if __name__ == "__main__":
    main()
