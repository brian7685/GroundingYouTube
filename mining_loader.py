import json
import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import random
import ffmpeg
import time
import re
import pickle
import sys
import csv
import math
import glob

class Mining_DataLoader(Dataset):
    """Youcook Video-Text loader."""

    def __init__(
            self,
            args,
            data,
            video_root='',
            num_clip=4,
            fps=16,
            num_frames=32,
            size=224,
            crop_only=False,
            center_crop=True,
            token_to_word_path='data/dict.npy',
            max_words=20,
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        self.data = pd.read_csv(data)
        self.video_root = video_root
        self.size = 224
        self.num_frames = 16
        self.fps = 1
        self.num_clip = num_clip
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.max_words = max_words
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
            
        self.split = 'validation'
        
        self.annotations = json.load(open(args.youcook2_annotations_path))
        final_dataset = json.load(open(args.interactions_annotations_path))
        #final_segments = pickle.load(open(args.interactions_segments_path, 'rb'))
        
        self.vis_clips = []
        self.selected_clips = {}
        self.clip_list = []
        
        # Full dataset evaluation
        for x,ys in self.annotations.items():
            #print(x,y)
            vid = x#curr_seg[0]
            #['Gf4eEhDpxgk','0uG0TwmuLHg','1CfeeRIdvgc','zUQGW3bpWOY']
            # process_l = ['-89_obTUlSU','-_9E7vmxi4o','01FLJjr-vJM','0GGOV4f7mKQ']
            # if vid not in process_l:
            #    continue
            video_id = vid
            if os.path.isfile(os.path.join(self.video_root, video_id + '.mp4')):
                video_path = os.path.join(self.video_root, video_id + '.mp4')
            elif os.path.isfile(os.path.join(self.video_root, video_id + '.mkv')):
                video_path = os.path.join(self.video_root, video_id + '.mkv')
            elif os.path.isfile(os.path.join(self.video_root, video_id + '.webm')):
                video_path = os.path.join(self.video_root, video_id + '.webm')
            else:
                print('no vid',vid)
                continue
            for y in ys:
                #print('seg',y['segment'])
                start = int(y['segment'][0])#curr_seg[1]
                end = int(y['segment'][1])#curr_seg[2]
                if end<=start:
                    continue
                final_anns = final_dataset[vid]
                #print('final_anns',final_anns)
                #orig = self.annotations[vid]['annotations']
                # idx = -1
                # for seg in orig:
                #     times = seg['segment']
                #     if start == times[0]:
                #         idx = seg['id']
                #         break
                idx = y['id']
                if idx > -1:
                    self.clip_list.append((vid, idx))
                if vid not in self.selected_clips:
                    self.selected_clips[vid] = {}
                if idx > -1 and idx not in self.selected_clips[vid]:
                    self.selected_clips[vid][idx] = {}
                #selected_seg = orig[idx]['segment']
                #actual_start = selected_seg[0]
                #actual_end = selected_seg[1]
                for k in range(start, end+1):
                    #print(k)
                    if k not in self.selected_clips[vid][idx]:
                        self.selected_clips[vid][idx][k] = []
                    if str(k) in final_anns:
                        #print('in')
                        bbox = final_anns[str(k)]
                        bbox += (0, 0, ) 
                    else:
                        bbox = (0, 0, 0, 0, 1, 1)
                    self.selected_clips[vid][idx][k] = bbox
        
    def _get_video_resolution(self, video_path):
        cmd = (ffmpeg.probe(video_path))
        streams = cmd['streams'][0]
        width = streams['width']
        height = streams['height']
        return height, width

    def __len__(self):
        return len(self.clip_list)

    def _get_video_start(self, video_path, start, end):
        start_seek = start
        dur = end - start
        
        cmd = (ffmpeg.probe(video_path))
        streams = cmd['streams'][0]
        width = streams['width']
        height = streams['height']
        
        cmd = (
            ffmpeg
            .input(video_path)
            .filter('fps', fps=self.fps)
        )
        cmd = (cmd.filter('scale', self.size, self.size))
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video)
        video = video.permute(3, 0, 1, 2)
        
        #video = video[:, int(start)-1:int(end)-1, :, :]
        video = video[:, int(start):int(end), :, :]
        output = []
        num_clips = math.ceil(video.shape[1] / self.num_frames)
        start_idx = 0
        end_idx = start_idx + self.num_frames
        for i in range(num_clips):
            tmp_clip = video[:, start_idx:end_idx, :, :]
            start_idx = end_idx
            end_idx = start_idx + self.num_frames
            zeros = th.zeros((3, self.num_frames - tmp_clip.shape[1], self.size, self.size), dtype=th.uint8) 
            tmp_clip = th.cat((tmp_clip, zeros), axis=1)
            output.append(tmp_clip.unsqueeze(0))
        output = th.cat(output, dim=0)

        return output, width, height
    
    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        idx = 0
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we, idx, len(words)
        else:
            return th.zeros(self.max_words).long(), 1, 1

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def __getitem__(self, idx):
        clip = self.clip_list[idx]
        video_id = clip[0]
        boxes = self.selected_clips[clip[0]][clip[1]]
        ann = self.annotations[video_id][int(clip[1])]['segment']
        cap = self.annotations[video_id][int(clip[1])]['sentence']
        cap = cap.replace('_',' ')
        start = ann[0]
        end = ann[1]
        
        if os.path.isfile(os.path.join(self.video_root, video_id + '.mp4')):
            video_path = os.path.join(self.video_root, video_id + '.mp4')
        elif os.path.isfile(os.path.join(self.video_root, video_id + '.mkv')):
            video_path = os.path.join(self.video_root, video_id + '.mkv')
        elif os.path.isfile(os.path.join(self.video_root, video_id + '.webm')):
            video_path = os.path.join(self.video_root, video_id + '.webm')
        else:
            video_path = os.path.join(self.video_root, video_id + '.mp4')
            print('error ID',video_path)
            raise ValueError
        # print('idx',idx)
        # print('anno',ann) #anno [95, 150] [4, 3, 16, 224, 224]
        # print('cap',cap)
        # print('video_path',video_path)
        # print('boxes',boxes)
        video, width, height = self._get_video_start(video_path, start, end)
        #print('width, height',width, height)
        # if width > 640:
        #     width = 640
        # if height > 360:
        #     height = 360
        raw_text = cap
        text, word_idx, num_words = self.words_to_ids(cap)
        mask = th.zeros((self.max_words), dtype=th.bool)
        mask[:num_words] = True        
        
        frame_indices = list(range(start, end+1))
        # print('video1',video.shape) #[1, 3, 16, 224, 224]
        # print('text1',text.shape)
        # print('mask1',mask.shape)
        
        return {'video': video, 'text': text, 'gt': boxes, 'video_id':video_id,\
            'idx': word_idx, 'mask': mask, 'width': width, 'height': height,\
                 'name': video_id, 'segment': int(clip[1]), 'frame_indices': frame_indices, 'raw_text':raw_text}