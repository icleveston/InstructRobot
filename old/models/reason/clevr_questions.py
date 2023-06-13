import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import utils as utils
from torch.utils.data import DataLoader


def get_dataset(opt, split):
    """Get function for dataset class"""
    assert split in ['train', 'val']

    if opt.dataset == 'clevr':
        if split == 'train':
            question_h5_path = opt.clevr_train_question_path
            max_sample = opt.max_train_samples
        else:
            question_h5_path = opt.clevr_val_question_path
            max_sample = opt.max_val_samples
        dataset = ClevrQuestionDataset(question_h5_path, max_sample, opt.clevr_vocab_path)
    else:
        raise ValueError('Invalid dataset')

    return dataset


def get_dataloader(opt, split):
    """Get function for dataloader class"""
    dataset = get_dataset(opt, split)
    shuffle = opt.shuffle if split == 'train' else 0
    loader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=shuffle, num_workers=opt.num_workers)
    print('| %s %s loader has %d samples' % (opt.dataset, split, len(loader.dataset)))
    return loader


class ClevrQuestionDataset(Dataset):

    def __init__(self, question_h5_path, max_samples, vocab_json):
        self.max_samples = max_samples
        question_h5 = h5py.File(question_h5_path, 'r')
        self.questions = torch.LongTensor(np.asarray(question_h5['questions'], dtype=np.int64))
        self.image_idxs = np.asarray(question_h5['image_idxs'], dtype=np.int64)
        self.programs, self.answers = None, None
        if 'programs' in question_h5:
            self.programs = torch.LongTensor(np.asarray(question_h5['programs'], dtype=np.int64))
        if 'answers' in question_h5:
            self.answers = np.asarray(question_h5['answers'], dtype=np.int64)
        self.vocab = utils.load_vocab(vocab_json)

    def __len__(self):
        if self.max_samples:
            return min(self.max_samples, len(self.questions))
        else:
            return len(self.questions)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('index %d out of range (%d)' % (idx, len(self)))
        question = self.questions[idx]
        image_idx = self.image_idxs[idx]
        program = -1
        answer = -1
        if self.programs is not None:
            program = self.programs[idx] 
        if self.answers is not None:
            answer = self.answers[idx]
        return question, program, answer, image_idx

