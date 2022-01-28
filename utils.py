import torch
import torch.nn as nn
from torch.autograd import Variable
import collections


class labelConverter(object):

    def __init__(self, alphabet):
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

        self.dict[''] = 0

    def encode(self, text):
        '''Convert list of strings to label seq'''
        length=[]
        seq=[]

        for item in text:
            length.append(len(item))

            for char in item:
                if char in self.dict:
                    label = self.dict[char]
                else:
                    label = 0
                seq.append(label)
        
        return (torch.IntTensor(seq),torch.IntTensor(length))

    def decode(self, seq, length):
        '''Reverse the above conversion'''
        seq_len = seq.numel() # return total no. of elements without multiplying dims
        word_no = length.numel()

        # word = []
        text = []
        
        assert sum(length)==seq_len
        start = 0
        for i in range(word_no):
            chars = []
            for label in list(seq[start : start+length[i]]):
                char = list(self.dict.keys())[list(self.dict.values()).index(label)]
                chars.append(char)
            word = ''.join(chars)
            start += length[i]
            text.append(word)
        
        return text
    
