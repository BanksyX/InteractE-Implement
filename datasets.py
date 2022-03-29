from cProfile import label
from select import select

from matplotlib.pyplot import tripcolor
from utils import *
from torch.utils.data import Dataset 

class TrainDataset(Dataset):
    '''
    Traing dataset class.
    ---------------------
    Parameters:
    triples:    The triples used for training the model
    params:     Parameters to be used
    ---------------------
    Returns:
    A training Dataset class instance used by DataLoader
    '''
    
    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
        self.strategy = self.p.train_strategy
        self.entities = np.arange(self.p.num_ent, dtype=np.int32)

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, index):
        ele = self.triples[index]
        triple, label, sub_samp = torch.LongTensor(ele['triple']), np.int32(ele['label']), np.float32(ele['sub_samp'])
        trp_label = self.get_label(label)
        
        if self.p.lbl_smooth != 0.0:
            trp_label = (1.0 - self.p.lbl_smooth)*trp_label + (1.0 / self.p.num_ent)
        
        if self.strategy == 'one2n':
            return triple, trp_label, None, None

        elif self.strategy == 'one2x':
            sub_samp = torch.FloatTensor([sub_samp])
            neg_ent = torch.LongTensor(self.get_neg_ent(triple, label))
            return triple, trp_label, neg_ent, sub_samp
        else:
            return NotImplementedError

    @staticmethod
    def collate_fn(data):
        pass
    
    def get_neg_ent(self, triple, label):
        pass

    def get_label(self, label):
        if self.strategy == 'one2n':
            y = np.zeros([self.p.num_ent], dtype = np.float32)
            for e2 in label: y[e2] = 1.0
        elif self.strategy == 'one2x':
            y = [1] + [0] * self.p.neg_num
        else:
            raise NotImplementedError
        return torch.FloatTensor(y)
    

class TestDataset(Dataset):
    '''
    Evaluation Dataset class
    ------------------------
    Parameters:
    triples:    
    params:
    -----------------------
    Returns:
    An evaluation Dataset class instance used by DataLoader for model evaluation
    '''
    def __init__(self, triples, params):
        self.triples = triples
        self.p = params
    
    def __len__(self):
        return len(self.triples)
    
    def __getitem__(self, index):
        ele = self.triples[index]
        triple, label = torch.LongTensor(ele['triple'], np.int32(ele['label']))
        label = self.get_label(label)

        return triple, label
    
    @staticmethod
    def collate_fn(data):
        triple = torch.stack([_[0] for _ in data], dim=0)
        label = torch.stack([_[1] for _ in data], dim=0)

        return triple, label
    
    def get_label(self, label):
        y = np.zeros([self.p.num_ent], dtype=np.float32)
        for e2 in label: y[e2] = 1.0
        return torch.FloatTensor(y)




