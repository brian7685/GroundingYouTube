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

class Youcook_DataLoader_clip(Dataset):
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
        
        self.annotations = json.load(open(args.youcook2_annotations_path))['database']
        final_dataset = pickle.load(open(args.interactions_annotations_path, 'rb'))
        final_segments = pickle.load(open(args.interactions_segments_path, 'rb'))
        
        self.vis_clips = []
        self.selected_clips = {}
        self.clip_list = []
        
        # Full dataset evaluation
        for curr_seg in final_segments:
            
            vid = curr_seg[0]
            
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
            start = curr_seg[1]
            end = curr_seg[2]
            final_anns = final_dataset[vid]
            orig = self.annotations[vid]['annotations']
            idx = -1
            for seg in orig:
                times = seg['segment']
                if start == times[0]:
                    idx = seg['id']
                    break
            if idx > -1:
                self.clip_list.append((vid, idx))
            if vid not in self.selected_clips:
                self.selected_clips[vid] = {}
            if idx > -1 and idx not in self.selected_clips[vid]:
                self.selected_clips[vid][idx] = {}
            selected_seg = orig[idx]['segment']
            actual_start = selected_seg[0]
            actual_end = selected_seg[1]
            for k in range(actual_start, actual_end+1):
                if k not in self.selected_clips[vid][idx]:
                    self.selected_clips[vid][idx][k] = []
                if k in final_anns:
                    bbox = final_anns[k]
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
        
        video = video[:, int(start)-1:int(end)-1, :, :]
        
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
        ann = self.annotations[clip[0]]['annotations'][int(clip[1])]['segment']
        cap = self.annotations[clip[0]]['annotations'][int(clip[1])]['sentence']
        
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
            
        video, width, height = self._get_video_start(video_path, start, end)
        
        if width > 640:
            width = 640
        if height > 360:
            height = 360
        
        text, word_idx, num_words = self.words_to_ids(cap)
        mask = th.zeros((self.max_words), dtype=th.bool)
        mask[:num_words] = True        
        
        frame_indices = list(range(start, end+1))
        # print('video1',video.shape) #[1, 3, 16, 224, 224]
        # print('text1',text.shape)
        # print('mask1',mask.shape)
        # print('anno',ann) #anno [95, 150] [4, 3, 16, 224, 224]
        # print('cap',cap)
        return {'video': video, 'text': text,'raw_text':cap,  'gt': boxes, 'idx': word_idx, 'mask': mask, 'width': width, 'height': height, 'name': video_id, 'segment': int(clip[1]), 'frame_indices': frame_indices}
        #return {'video': video, 'text': text, 'raw_text':cap, 'audio': audio, 'nframes': nframes, 'video_id':video_id}


class Youcook_DataLoader_clip2(Dataset):
    """Youcook Video-Text loader."""

    def __init__(
            self,
            data,
            video_root='',
            num_clip=4,
            fps=16,
            num_frames=32,
            size=224,
            crop_only=False,
            center_crop=True,
            token_to_word_path='data/dict.npy',
            max_words=30,
            audio=False,
            audio_data_path='',
            num_frames_multiplier=1
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        self.data = pd.read_csv(data)
        self.video_root = video_root
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.num_clip = num_clip
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.max_words = max_words
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.audio = audio
        if self.audio:
            self.audio_data = pickle.load(open(audio_data_path, 'rb'))
            self.num_frames_multiplier = num_frames_multiplier

    def __len__(self):
        if not self.audio:
            return len(self.data) 
        return len(self.audio_data)

    def _get_video(self, video_path, start, end, num_clip):
        video = th.zeros(num_clip, 3, self.num_frames, self.size, self.size)
        start_ind = np.linspace(start, max(start, end-self.num_sec - 0.4), num_clip) 
        for i, s in enumerate(start_ind):
            video[i] = self._get_video_start(video_path, s) 
        return video

    def _get_video_start(self, video_path, start):
        start_seek = start
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
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video)
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.num_frames:
            zeros = th.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        return video[:, :self.num_frames]

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def __getitem__(self, idx):
        if not self.audio: 
            video_id = self.data['video_id'].values[idx]
            task = self.data['task'].values[idx]
            start = self.data['start'].values[idx]
            end = self.data['end'].values[idx]
            cap = self.data['text'].values[idx]
            text = self.words_to_ids(cap)
            audio, nframes = th.zeros(1), th.zeros(1)
        else: # use the spectrogram file since indices are different
            #print('has audio')
            video_id = self.audio_data[idx]['video_id']
            task = self.audio_data[idx]['task']
            start = self.audio_data[idx]['start']
            end = self.audio_data[idx]['end']
            # load audio and zero pad/truncate if necessary
            audio = self.audio_data[idx]['audio']
            target_length = 1024 * self.num_frames_multiplier
            nframes = audio.numpy().shape[1]
            p = target_length - nframes
            if p > 0:
                audio = np.pad(audio, ((0,0),(0,p)), 'constant', constant_values=(0,0))
            elif p < 0:
                audio = audio[:,0:p]
            audio = th.FloatTensor(audio)
            #text = th.zeros(1)
            cap = self.audio_data[idx]['caption']
            
            text = self.words_to_ids(cap)
        # load the video
        #print('path',os.path.join(self.video_root, 'validation', str(task), video_id + '.mp4'))
        if os.path.isfile(os.path.join(self.video_root, 'validation', str(task), video_id + '.mp4')):
            video_path = os.path.join(self.video_root, 'validation', str(task), video_id + '.mp4')
        elif os.path.isfile(os.path.join(self.video_root, 'validation', str(task), video_id + '.mkv')):
            video_path = os.path.join(self.video_root, 'validation', str(task), video_id + '.mkv')
        elif os.path.isfile(os.path.join(self.video_root, 'validation', str(task), video_id + '.webm')):
            video_path = os.path.join(self.video_root, 'validation', str(task), video_id + '.webm')
        else:
            print('error',os.path.join(self.video_root, 'validation', str(task), video_id + '.mp4'))
            raise ValueError 
        
        video = self._get_video(video_path, start, end, self.num_clip)
        print('clip raw text')
        return {'video': video, 'text': text, 'raw_text':cap, 'audio': audio, 'nframes': nframes, 'video_id':video_id}


class Youcook_DataLoader_clip_temp(Dataset):
    """Youcook Video-Text loader."""

    def __init__(
            self,
            data,
            video_root='',
            num_clip=4,
            fps=16,
            num_frames=32,
            size=224,
            crop_only=False,
            center_crop=True,
            token_to_word_path='data/dict.npy',
            max_words=30,
            audio=False,
            audio_data_path='',
            num_frames_multiplier=1
    ):
        """
        Args:
        """
        assert isinstance(size, int)
        self.data = pd.read_csv(data)
        self.video_root = video_root
        self.size = size
        self.num_frames = num_frames
        self.fps = fps
        self.num_clip = num_clip
        self.num_sec = self.num_frames / float(self.fps)
        self.crop_only = crop_only
        self.center_crop = center_crop
        self.max_words = max_words
        token_to_word = np.load(os.path.join(os.path.dirname(__file__), token_to_word_path))
        self.word_to_token = {}
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1
        self.audio = audio
        if self.audio:
            self.audio_data = pickle.load(open(audio_data_path, 'rb'))
            self.num_frames_multiplier = num_frames_multiplier

    def __len__(self):
        if not self.audio:
            return len(self.data) 
        return len(self.audio_data)

    def _get_video(self, video_path, start, end, num_clip):
        video = th.zeros(num_clip, 3, self.num_frames, self.size, self.size)
        start_ind = np.linspace(start, max(start, end-self.num_sec - 0.4), num_clip) 
        for i, s in enumerate(start_ind):
            video[i] = self._get_video_start(video_path, s) 
        return video

    def _get_video_start(self, video_path, start):
        start_seek = start
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
        out, _ = (
            cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
            .run(capture_stdout=True, quiet=True)
        )
        video = np.frombuffer(out, np.uint8).reshape([-1, self.size, self.size, 3])
        video = th.from_numpy(video)
        video = video.permute(3, 0, 1, 2)
        if video.shape[1] < self.num_frames:
            zeros = th.zeros((3, self.num_frames - video.shape[1], self.size, self.size), dtype=th.uint8)
            video = th.cat((video, zeros), axis=1)
        return video[:, :self.num_frames]

    def _split_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_token(self, words):
        words = [self.word_to_token[word] for word in words if word in self.word_to_token]
        if words:
            we = self._zero_pad_tensor_token(th.LongTensor(words), self.max_words)
            return we
        else:
            return th.zeros(self.max_words).long()

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def words_to_ids(self, x):
        return self._words_to_token(self._split_text(x))

    def __getitem__(self, idx):
        if not self.audio: 
            video_id = self.data['video_id'].values[idx]
            task = self.data['task'].values[idx]
            start = self.data['start'].values[idx]
            end = self.data['end'].values[idx]
            cap = self.data['text'].values[idx]
            text = self.words_to_ids(cap)
        #     audio, nframes = th.zeros(1), th.zeros(1)
        # else: # use the spectrogram file since indices are different
        #     #print('has audio')
        #     video_id = self.audio_data[idx]['video_id']
        #     task = self.audio_data[idx]['task']
        #     start = self.audio_data[idx]['start']
        #     end = self.audio_data[idx]['end']
        #     # load audio and zero pad/truncate if necessary
        #     audio = self.audio_data[idx]['audio']
        #     target_length = 1024 * self.num_frames_multiplier
        #     nframes = audio.numpy().shape[1]
        #     p = target_length - nframes
        #     if p > 0:
        #         audio = np.pad(audio, ((0,0),(0,p)), 'constant', constant_values=(0,0))
        #     elif p < 0:
        #         audio = audio[:,0:p]
        #     audio = th.FloatTensor(audio)
        #     #text = th.zeros(1)
        #     cap = self.audio_data[idx]['caption']
            
        #     text = self.words_to_ids(cap)
        # load the video
        #print('path',os.path.join(self.video_root, 'validation', str(task), video_id + '.mp4'))
        if os.path.isfile(os.path.join(self.video_root, 'validation', str(task), video_id + '.mp4')):
            video_path = os.path.join(self.video_root, 'validation', str(task), video_id + '.mp4')
        elif os.path.isfile(os.path.join(self.video_root, 'validation', str(task), video_id + '.mkv')):
            video_path = os.path.join(self.video_root, 'validation', str(task), video_id + '.mkv')
        elif os.path.isfile(os.path.join(self.video_root, 'validation', str(task), video_id + '.webm')):
            video_path = os.path.join(self.video_root, 'validation', str(task), video_id + '.webm')
        else:
            print('error',os.path.join(self.video_root, 'validation', str(task), video_id + '.mp4'))
            raise ValueError 
        
        video = self._get_video(video_path, start, end, self.num_clip)

        return {'video': video, 'text': text, 'raw_text':cap, \
            #'audio': audio, 
            'nframes': nframes, 'video_id':video_id}
