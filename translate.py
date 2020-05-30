import argparse
import time
import torch
from Models import get_model
from Process import *
import torch.nn.functional as F
from Optim import CosineWithRestarts
from Batch import create_masks
import pdb
import dill as pickle
import argparse
from Models import get_model
from Beam import beam_search
from nltk.corpus import wordnet
from torch.autograd import Variable
import re

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

def translate_sentence(sentence, model, opt, SRC, TRG):
    
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    print(sentence[:-1])
    for tok in sentence:
        if SRC.vocab.stoi[tok] != 0 or opt.floyd == True:
            indexed.append(SRC.vocab.stoi[tok])
        else:
            indexed.append(get_synonym(tok, SRC))
    sentence = Variable(torch.LongTensor([indexed]))
    if opt.device == 0:
        sentence = sentence.cuda()
    
    sentence = beam_search(sentence, model, SRC, TRG, opt)

    return  sentence

def translate(opt, model, SRC, TRG):
    sentence = opt.text.lower()
    translated = translate_sentence(sentence + '.', model, opt, SRC, TRG).capitalize()

    return (translated)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-fold', default=0)
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=1)
    parser.add_argument('-max_len', type=int, default=3)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-src_lang', default='en')
    parser.add_argument('-trg_lang', default='en')
    parser.add_argument('-heads', type=int, default=1)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-no_cuda', action='store_true')
    parser.add_argument('-floyd', action='store_true')
    parser.add_argument('-savetokens', type=int, default=0)
    opt = parser.parse_args()

    opt.device = 0 if opt.no_cuda is False else -1
 
    assert opt.k > 0

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab), model_type='inference')

    data = pd.read_csv('data/train_folds.csv')
    for ind_ in [13, 11, 15, ]:
        fold_data = data.loc[data['kfold'] == opt.fold].iloc[ind_]
        opt.text = fold_data['text']
        print(f'original text > {opt.text}')
        print(f'original selected text > {fold_data["selected_text"]} \nsentiment: {fold_data["sentiment"]}')
        phrase = translate(opt, model, SRC, TRG)
        print('> prediction: '+ phrase + '\n')
        print('')

if __name__ == '__main__':
    main()
