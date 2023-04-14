import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re
import sys
import pickle
import struct
import io
import math
from PIL import Image
from random import randrange

class HT100M_DataLoader_clip(Dataset):
    """HowTo100M Video-Text loader."""

    def __init__(
            self,
            csv,
            video_root='',
            caption_root='',
            min_time=4.0,
            fps=16,
            num_frames=16,
            size=224,
            crop_only=False,
            center_crop=True,
            benchmark=False,
            token_to_word_path='data/dict.npy',
            max_words=20,
            num_candidates=1,
            random_left_right_flip=False,
            num_sec_control=0,
            fix_start=False,
            early_start=False,
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        self.csv = pd.read_csv(os.path.join(os.path.dirname(__file__), csv))
        self.video_root = video_root
        self.caption_root = caption_root
        self.min_time = min_time
        self.size = size
        self.num_frames = num_frames 
        self.fps = fps
        self.num_sec_control = num_sec_control
        self.num_sec = self.num_frames / float(self.fps) + self.num_sec_control
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.benchmark = benchmark
        self.max_words = max_words
        self.fix_start = fix_start
        self.early_start = early_start
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.num_candidates = 1
        self.random_flip = random_left_right_flip
        
        #self.all_vids = list(pickle.load(open('./data/training_clips.pkl', 'rb')))

    def __len__(self):
        #return len(self.all_vids)
        return len(self.csv)

    def _get_text(self, caption):
        #print('caption',caption)
        def isNaN(num):
            return num!= num
        cap = pd.read_csv(caption)     
        num_words= 0
        count=0
        while num_words==0:   
            count+=1
            if count>100:
                raw_text = 'None'
                words, num_words = self.words_to_ids(raw_text)
            
                break
            narr_ind = randrange(len(cap['text']))#0#int(cap[cap['start'] == start].index.values[0])
            raw_text = cap['text'].values[narr_ind]
            words, num_words = self.words_to_ids(cap['text'].values[narr_ind])
            
            if isNaN(raw_text):
                num_words=0
        #print('cap',len(cap['text']))
        
        # if self.num_candidates == 1:
            
        #     try:
        #         words, num_words = self.words_to_ids(cap['text'].values[narr_ind])
        #     except:
        #         print('self.words_to_ids',self.words_to_ids(cap['text'].values[narr_ind]))
        # else:
        #     words = th.zeros(self.num_candidates, self.max_words, dtype=th.long)
        #     cap_start = self._find_nearest_candidates(cap, ind)
        #     for i in range(self.num_candidates):
        #         words[i] = self.words_to_ids(cap['text'].values[max(0, min(len(cap['text']) - 1, cap_start + i))])
        start, end = cap['start'].values[narr_ind], cap['end'].values[narr_ind]

        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time 
            
        return words, int(start), int(end), num_words, raw_text

    def compute_offset(self, size, sizes):
        offsets = [0]
        for i in range(size):
            offsets.append(offsets[i]+struct.unpack_from('<Q', sizes, 8*i)[0])
        return offsets
        
    def _get_video(self, video_path, start, end):
        #print('start end',start,end,end-start)
        if self.fix_start:
            #print('fix start')
            start_seek = start
        elif self.early_start:
            mid = int((start+end)/2)
            start_seek = max (0,mid-8)
            #print('mid',mid)
        else:
            
            start_seek = random.randint(start, int(max(start, end - self.num_sec)))
        #print('start',start_seek,start,start_seek-start,end-start,self.num_sec)
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec + 0.1 )
            .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        if self.random_flip and random.uniform(0, 1) > 0.5:
            cmd = cmd.hflip()
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        #video = th.from_numpy(video)
        video = th.tensor(video)
        video = video.permute(3, 0, 1, 2)
        #print('video length',video.shape)
        if video.shape[1] < self.num_frames+self.num_sec_control:
            zeros = th.zeros((3, self.num_frames+self.num_sec_control - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        return video[:, :self.num_frames+self.num_sec_control]

    # def parse_bcf(self, video, start, end, fps):
    #     f = open(video, 'rb')
    #     size = struct.unpack('<Q', f.read(8))[0]
    #     sizes = f.read(size*8)
    #     offsets = self.compute_offset(size, sizes)
    #     end_idx = min(math.ceil(end * fps), len(offsets)-1)
    #     start_idx = max(0, min(math.floor(start * fps), end_idx - math.ceil(self.num_frames/fps)))
    #     frames = []
    #     for i in range(start_idx, end_idx):
    #         f.seek(len(offsets)*8 + offsets[i])
    #         data_i = f.read(offsets[i+1] - offsets[i])
    #         img = Image.open(io.BytesIO(data_i))
    #         img = np.asarray(img)
    #         img = th.from_numpy(img)
    #         frames.append(img.unsqueeze(0))
    #     frames = th.cat(frames, dim=0)
    #     frames = frames.permute(3, 0, 1, 2)
    #     if frames.shape[1] < self.num_frames:
    #         zeros = th.zeros((3, self.num_frames - frames.shape[1], self.size, self.size), dtype=th.uint8)
    #         frames = th.cat((frames, zeros), axis=1)
    #     f.close()
        
    #     return frames[:, :self.num_frames]

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    # def _words_to_token(self, words):
    #     words = [self.word_to_token[word] for word in words if word in self.word_to_token]
    #     if words:
    #         we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
    #         return we
    #     else:
    #         return th.zeros(self.max_words, dtype=th.long)

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we, len(words)
        else:
            return th.zeros(self.max_words, dtype=th.long), 0
            
    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def _find_nearest_candidates(self, caption, ind):
        start, end = ind, ind
        diff = caption['end'][end] - caption['start'][start]
        n_candidate = 1
        while n_candidate < self.num_candidates:
           if start == 0:
               return 0
           elif end == len(caption) - 1:
               return start - (self.num_candidates - n_candidate)
           elif caption['end'][end] - caption['start'][start - 1] < caption['end'][end + 1] - caption['start'][start]:
               start -= 1
           else:
               end += 1
           n_candidate += 1
        return start

    def __getitem__(self, idx):
        # clip = self.all_vids[idx]
        
        # video_file = clip[-1].strip()     
        # text, start, end, num_words = self._get_text(video_file, os.path.join(self.caption_root, video_file + '.csv'), clip[0])
        
        # mask = th.zeros((self.max_words), dtype=th.bool)
        # mask[:num_words] = True
        
        # if self.video_format == 'bcf':
        #     video_path = os.path.join(self.video_root, video_file + '.bcf')
        #     video = self.parse_bcf(video_path, start, end, self.fps)
        # else:
        #     video_path = os.path.join(self.video_root, video_file + '.mp4')
        #     video = self._get_video(video_path, start, end)

        video_file = self.csv['video_path'][idx]
        video_id = video_file.split('/')[-1].split('.')[0]
        video_path = os.path.join(self.video_root, video_file)

        #audio_path = os.path.join(self.features_path_audio, video_path, video_id + "_spec.npz")

        text, start, end, num_words, raw_text = self._get_text(os.path.join(self.caption_root, video_id + '.csv'))
        mask = th.zeros((self.max_words), dtype=th.bool)
        mask[:num_words] = True
        try:
            video = self._get_video(video_path, start, end)
        except:
            print('error vid',video_id)
            new_idx = np.random.randint(0,len(self.csv))
            return self.__getitem__(new_idx)
        #return {'video': video, 'text': text}
        #pritn('video_id')
        #print('mask',mask,num_words)
        return {'video': video, \
        'text': text, 'mask': mask, 'idx': idx, \
        'vid': video_id, 'raw_text':raw_text}



class HT100M_DataLoader_clip_prompt(Dataset):
    """HowTo100M Video-Text loader."""

    def __init__(
            self,
            csv,
            video_root='',
            caption_root='',
            min_time=4.0,
            fps=16,
            num_frames=16,
            size=224,
            crop_only=False,
            center_crop=True,
            benchmark=False,
            token_to_word_path='data/dict.npy',
            max_words=20,
            num_candidates=1,
            random_left_right_flip=False,
            num_sec_control=0,
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        self.csv = pd.read_csv(os.path.join(os.path.dirname(__file__), csv))
        self.video_root = video_root
        self.caption_root = caption_root
        self.min_time = min_time
        self.size = size
        self.num_frames = num_frames 
        self.fps = fps
        self.num_sec_control = num_sec_control
        self.num_sec = self.num_frames / float(self.fps) + self.num_sec_control
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.benchmark = benchmark
        self.max_words = max_words
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.num_candidates = 1
        self.random_flip = random_left_right_flip
        
        #self.all_vids = list(pickle.load(open('./data/training_clips.pkl', 'rb')))

    def __len__(self):
        #return len(self.all_vids)
        return len(self.csv)

    def _get_text(self, caption):
        #print('caption',caption)
        def isNaN(num):
            return num!= num
        cap = pd.read_csv(caption)     
        num_words= 0
        while num_words==0:   
            narr_ind = randrange(len(cap['text']))#0#int(cap[cap['start'] == start].index.values[0])
            raw_text = cap['text'].values[narr_ind]
            words, num_words = self.words_to_ids(cap['text'].values[narr_ind])
            if isNaN(raw_text):
                num_words=0
        #print('cap',len(cap['text']))
        
        # if self.num_candidates == 1:
            
        #     try:
        #         words, num_words = self.words_to_ids(cap['text'].values[narr_ind])
        #     except:
        #         print('self.words_to_ids',self.words_to_ids(cap['text'].values[narr_ind]))
        # else:
        #     words = th.zeros(self.num_candidates, self.max_words, dtype=th.long)
        #     cap_start = self._find_nearest_candidates(cap, ind)
        #     for i in range(self.num_candidates):
        #         words[i] = self.words_to_ids(cap['text'].values[max(0, min(len(cap['text']) - 1, cap_start + i))])
        start, end = cap['start'].values[narr_ind], cap['end'].values[narr_ind]

        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time 
            
        return words, int(start), int(end), num_words, raw_text

    def compute_offset(self, size, sizes):
        offsets = [0]
        for i in range(size):
            offsets.append(offsets[i]+struct.unpack_from('<Q', sizes, 8*i)[0])
        return offsets
        
    def _get_video(self, video_path, start, end):
        
        start_seek = random.randint(start, int(max(start, end - self.num_sec)))
        cmd = (
            ffmpeg
            .input(video_path, ss=start_seek, t=self.num_sec + 0.1)
            .filter('fps', fps=self.fps)
        )
        if self.center_crop:
            aw, ah = 0.5, 0.5
        else:
            aw, ah = random.uniform(0, 1), random.uniform(0, 1)
        if self.crop_only:
            cmd = (
                cmd.crop('(iw - {})*{}'.format(self.size, aw),
                         '(ih - {})*{}'.format(self.size, ah),
                         str(self.size), str(self.size))
            )
        else:
            cmd = (
                cmd.crop('(iw - min(iw,ih))*{}'.format(aw),
                         '(ih - min(iw,ih))*{}'.format(ah),
                         'min(iw,ih)',
                         'min(iw,ih)')
                .filter('scale', self.size, self.size)
            )
        if self.random_flip and random.uniform(0, 1) > 0.5:
            cmd = cmd.hflip()
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        #video = th.from_numpy(video)
        video = th.tensor(video)
        video = video.permute(3, 0, 1, 2)
        #print('video length',video.shape[1])
        if video.shape[1] < self.num_frames+self.num_sec_control:
            zeros = th.zeros((3, self.num_frames+self.num_sec_control - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        return video[:, :self.num_frames+self.num_sec_control]

    # def parse_bcf(self, video, start, end, fps):
    #     f = open(video, 'rb')
    #     size = struct.unpack('<Q', f.read(8))[0]
    #     sizes = f.read(size*8)
    #     offsets = self.compute_offset(size, sizes)
    #     end_idx = min(math.ceil(end * fps), len(offsets)-1)
    #     start_idx = max(0, min(math.floor(start * fps), end_idx - math.ceil(self.num_frames/fps)))
    #     frames = []
    #     for i in range(start_idx, end_idx):
    #         f.seek(len(offsets)*8 + offsets[i])
    #         data_i = f.read(offsets[i+1] - offsets[i])
    #         img = Image.open(io.BytesIO(data_i))
    #         img = np.asarray(img)
    #         img = th.from_numpy(img)
    #         frames.append(img.unsqueeze(0))
    #     frames = th.cat(frames, dim=0)
    #     frames = frames.permute(3, 0, 1, 2)
    #     if frames.shape[1] < self.num_frames:
    #         zeros = th.zeros((3, self.num_frames - frames.shape[1], self.size, self.size), dtype=th.uint8)
    #         frames = th.cat((frames, zeros), axis=1)
    #     f.close()
        
    #     return frames[:, :self.num_frames]

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    # def _words_to_token(self, words):
    #     words = [self.word_to_token[word] for word in words if word in self.word_to_token]
    #     if words:
    #         we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
    #         return we
    #     else:
    #         return th.zeros(self.max_words, dtype=th.long)

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we, len(words)
        else:
            return th.zeros(self.max_words, dtype=th.long), 0
            
    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def _find_nearest_candidates(self, caption, ind):
        start, end = ind, ind
        diff = caption['end'][end] - caption['start'][start]
        n_candidate = 1
        while n_candidate < self.num_candidates:
           if start == 0:
               return 0
           elif end == len(caption) - 1:
               return start - (self.num_candidates - n_candidate)
           elif caption['end'][end] - caption['start'][start - 1] < caption['end'][end + 1] - caption['start'][start]:
               start -= 1
           else:
               end += 1
           n_candidate += 1
        return start

    def __getitem__(self, idx):
        # clip = self.all_vids[idx]
        
        # video_file = clip[-1].strip()     
        # text, start, end, num_words = self._get_text(video_file, os.path.join(self.caption_root, video_file + '.csv'), clip[0])
        
        # mask = th.zeros((self.max_words), dtype=th.bool)
        # mask[:num_words] = True
        
        # if self.video_format == 'bcf':
        #     video_path = os.path.join(self.video_root, video_file + '.bcf')
        #     video = self.parse_bcf(video_path, start, end, self.fps)
        # else:
        #     video_path = os.path.join(self.video_root, video_file + '.mp4')
        #     video = self._get_video(video_path, start, end)

        video_file = self.csv['video_path'][idx]
        video_id = video_file.split('/')[-1].split('.')[0]
        video_path = os.path.join(self.video_root, video_file)

        #audio_path = os.path.join(self.features_path_audio, video_path, video_id + "_spec.npz")

        text, start, end, num_words, raw_text = self._get_text(os.path.join(self.caption_root, video_id + '.csv'))
        mask = th.zeros((self.max_words), dtype=th.bool)
        mask[:num_words] = True
        try:
            video = self._get_video(video_path, start, end)
        except:
            print('error vid',video_id)
            new_idx = np.random.randint(0,len(self.csv))
            return self.__getitem__(new_idx)

        prompt_text = []
        raw_text_split = raw_text.split()
        #for t in raw_text_split:
        for i in range(num_words):
            try:
                prompt_text.append('This is a photo of '+raw_text_split[i])
            except:
                prompt_text.append('This is a photo of')
        #print('prompt_text',prompt_text)
        #print('raw_text',raw_text)
        #print('text',text)
        #print('mask',mask)
        #return {'video': video, 'text': text}
        #pritn('video_id')
        #print('mask',mask,num_words)
        return {'video': video, \
        'text': text, 'mask': mask, 'idx': idx, \
        'vid': video_id, 'raw_text':raw_text,'prompt_text':prompt_text}