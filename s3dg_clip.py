"""Contains the definition for Gated Separable 3D network (S3D-G).
"""

import torch as th
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import re
import sys
import math

class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        return x

class Context_Gating(nn.Module):
    def __init__(self, dimension):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)

class InceptionBlock(nn.Module):

    def __init__(self, input_dim,
                num_outputs_0_0a,
                num_outputs_1_0a,
                num_outputs_1_0b,
                num_outputs_2_0a,
                num_outputs_2_0b,
                num_outputs_3_0b,
                gating=True):
        super(InceptionBlock, self).__init__()
        self.conv_b0 = STConv3D(input_dim, num_outputs_0_0a, [1, 1, 1])
        self.conv_b1_a = STConv3D(input_dim, num_outputs_1_0a, [1, 1, 1])
        self.conv_b1_b = STConv3D(num_outputs_1_0a, num_outputs_1_0b, [3, 3, 3],
                                    padding=1, separable=True)
        self.conv_b2_a = STConv3D(input_dim, num_outputs_2_0a, [1, 1, 1])
        self.conv_b2_b = STConv3D(num_outputs_2_0a, num_outputs_2_0b, [3, 3, 3],
                                    padding=1, separable=True)
        self.maxpool_b3 = th.nn.MaxPool3d((3, 3, 3), stride=1, padding=1)
        self.conv_b3_b = STConv3D(input_dim, num_outputs_3_0b, [1, 1, 1])
        self.gating = gating
        self.output_dim = num_outputs_0_0a + num_outputs_1_0b +\
                    num_outputs_2_0b + num_outputs_3_0b
        if gating:
            self.gating_b0 = SelfGating(num_outputs_0_0a)
            self.gating_b1 = SelfGating(num_outputs_1_0b)
            self.gating_b2 = SelfGating(num_outputs_2_0b)
            self.gating_b3 = SelfGating(num_outputs_3_0b)

    def forward(self, input):
      """Inception block
      """
      b0 = self.conv_b0(input)
      b1 = self.conv_b1_a(input)
      b1 = self.conv_b1_b(b1)
      b2 = self.conv_b2_a(input)
      b2 = self.conv_b2_b(b2)
      b3 = self.maxpool_b3(input)
      b3 = self.conv_b3_b(b3)
      if self.gating:
          b0 = self.gating_b0(b0)
          b1 = self.gating_b1(b1)
          b2 = self.gating_b2(b2)
          b3 = self.gating_b3(b3)
      return th.cat((b0, b1, b2, b3), dim=1)

class SelfGating(nn.Module):

    def __init__(self, input_dim):
        super(SelfGating, self).__init__()
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, input_tensor):
      """Feature gating as used in S3D-G.
      """
      spatiotemporal_average = th.mean(input_tensor, dim=[2, 3, 4])
      weights = self.fc(spatiotemporal_average)
      weights = th.sigmoid(weights)
      return weights[:, :, None, None, None] * input_tensor

class STConv3D(nn.Module):

    def __init__(self,
                input_dim,
                output_dim,
                kernel_size,
                stride=1,
                padding=0,
                separable=False):
        super(STConv3D, self).__init__()
        self.separable = separable
        self.relu = nn.ReLU(inplace=True)
        assert len(kernel_size) == 3
        if separable and kernel_size[0] != 1:
            spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]
            temporal_kernel_size = [kernel_size[0], 1, 1]
            if isinstance(stride, list) and len(stride) == 3:
              spatial_stride = [1, stride[1], stride[2]]
              temporal_stride = [stride[0], 1, 1]
            else:
              spatial_stride = [1, stride, stride]
              temporal_stride = [stride, 1, 1]
            if isinstance(padding, list) and len(padding) == 3:
              spatial_padding = [0, padding[1], padding[2]]
              temporal_padding = [padding[0], 0, 0]
            else:
              spatial_padding = [0, padding, padding]
              temporal_padding = [padding, 0, 0]
        if separable:
            self.conv1 = nn.Conv3d(input_dim, output_dim,
                                   kernel_size=spatial_kernel_size,
                                   stride=spatial_stride,
                                   padding=spatial_padding, bias=False)
            self.bn1 = nn.BatchNorm3d(output_dim)
            self.conv2 = nn.Conv3d(output_dim, output_dim,
                                   kernel_size=temporal_kernel_size,
                                   stride=temporal_stride,
                                   padding=temporal_padding, bias=False)
            self.bn2 = nn.BatchNorm3d(output_dim)
        else:
            self.conv1 = nn.Conv3d(input_dim, output_dim,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, bias=False)
            self.bn1 = nn.BatchNorm3d(output_dim)


    def forward(self, input):
        out = self.relu(self.bn1(self.conv1(input)))
        if self.separable:
            out = self.relu(self.bn2(self.conv2(out)))
        return out


def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


class MaxPool3dTFPadding(th.nn.Module):
    def __init__(self, kernel_size, stride=None, padding='SAME'):
        super(MaxPool3dTFPadding, self).__init__()
        if padding == 'SAME':
            padding_shape = get_padding_shape(kernel_size, stride)
            self.padding_shape = padding_shape
            self.pad = th.nn.ConstantPad3d(padding_shape, 0)
        self.pool = th.nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        inp = self.pad(inp)
        out = self.pool(inp)
        return out

class Sentence_Embedding(nn.Module):
    def __init__(self,
                 embd_dim,
                 token_to_word_path,
                 num_embeddings=66250,
                 word_embedding_dim=300,
                 word2vec_path='',
                 max_words=16,
                 output_dim=2048):
        super(Sentence_Embedding, self).__init__()
        if word2vec_path:
            self.word_embd = nn.Embedding.from_pretrained(th.load(word2vec_path)) 
        else:
            self.word_embd = nn.Embedding(num_embeddings, word_embedding_dim)
        self.fc1 = nn.Linear(word_embedding_dim, output_dim)
        self.word_to_token = {}
        self.max_words = max_words
        token_to_word = np.load(token_to_word_path)
        for i, t in enumerate(token_to_word):
            self.word_to_token[t] = i + 1

    def _zero_pad_tensor_token(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = th.zeros(size - len(tensor)).long()
            return th.cat((tensor, zero), dim=0)

    def is_cuda(self):
        return self.fc1.bias.is_cuda

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

    def words_to_ids(self, x):
        split_x = [self._words_to_token(self._split_text(sent)) for sent in x]
        return th.stack(split_x, dim=0)

    def forward(self, x, mask, raw_text=False):
        if raw_text:
            x = self.words_to_ids(x)
        with th.no_grad():
            x = self.word_embd(x)
            
        x = F.relu(self.fc1(x), inplace=True)
        
        return x
    
class CAEncoder(nn.Module):
    def __init__(self, dim=512, hidden_dim=512):
        super(CAEncoder, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, video_embd, text_embd, mask):
        video_embd_att, text_embd_att, vis_scores, text_scores = cross_attention(video_embd, text_embd, mask, self.training)
        return video_embd_att, text_embd_att, vis_scores, text_scores
    
class SAEncoder(nn.Module):
    def __init__(self, dim=512, hidden_dim=512, num_heads=1):
        super(SAEncoder, self).__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.vis_attn_mat = nn.MultiheadAttention(self.dim, self.num_heads)
        self.text_attn_mat = nn.MultiheadAttention(self.dim, self.num_heads)
        
    def forward(self, video_embd, text_embd, mask):
        if self.training:
            video_size = video_embd.size()
            text_size = text_embd.size()
        
            video_embd = video_embd.view(-1, video_embd.size(-2), video_embd.size(-1)) #[256, 98, 512]
            #print('video_embd',video_embd.shape)
            text_embd = text_embd.view(-1, text_embd.size(-2), text_embd.size(-1)) #[256, 20, 512]
            #print('text_embd',text_embd.shape)
        
            num_samples = mask.size(0)
            mask = mask.unsqueeze(0).repeat(num_samples, 1, 1) #[16, 16, 20]
            #print('mask',mask.shape) 
            mask = mask.view(-1, mask.size(-1)) #[256, 20]
            #print('mask',mask.shape)
        
            tmp_video_embd = video_embd.permute(1, 0, 2) #[98, 256, 512]
            #print('tmp_video_embd',tmp_video_embd.shape)
            video_embd_att, vis_attn_scores = self.vis_attn_mat(tmp_video_embd, tmp_video_embd, tmp_video_embd)
            #print('video_embd_att',video_embd_att.shape) #[98, 256, 512]
            video_embd_att = video_embd_att.permute(1, 0, 2) #[256, 98, 512]
            #print('video_embd_att',video_embd_att.shape)
        
            tmp_text_embd = text_embd.permute(1, 0, 2) #[20, 256, 512]
            #print('tmp_text_embd',tmp_text_embd.shape)
            text_embd_att, text_attn_scores = self.text_attn_mat(tmp_text_embd, tmp_text_embd, tmp_text_embd, key_padding_mask=~mask)
            #print('text_embd_att',text_embd_att.shape) #[20, 256, 512]
            text_embd_att  = text_embd_att.permute(1, 0, 2) #[256, 20, 512]
            #print('text_embd_att',text_embd_att.shape)
        
            # Residual Connection
            video_embd_att = video_embd_att + video_embd
            text_embd_att = text_embd_att + text_embd
        
            video_embd_att = video_embd_att.view(video_size) #[16, 16, 98, 512]
            #print('video_embd_att',video_embd_att.shape) 
            text_embd_att = text_embd_att.view(text_size)  #[16, 16, 20, 512]
            #print('text_embd_att',text_embd_att.shape)
        
        else:
            num_samples = video_embd.size(0)
            mask = mask.unsqueeze(0).repeat(num_samples, 1, 1)
            mask = mask.view(-1, mask.size(-1))
        
            tmp_video_embd = video_embd.permute(1, 0, 2)
            video_embd_att, vis_attn_scores = self.vis_attn_mat(tmp_video_embd, tmp_video_embd, tmp_video_embd)
            video_embd_att = video_embd_att.permute(1, 0, 2)
        
            tmp_text_embd = text_embd.permute(1, 0, 2)     
            text_embd_att, text_attn_scores = self.text_attn_mat(tmp_text_embd, tmp_text_embd, tmp_text_embd, key_padding_mask=~mask)
            text_embd_att  = text_embd_att.permute(1, 0, 2)
        
            # Residual Connection
            video_embd_att = video_embd_att + video_embd
            text_embd_att = text_embd_att + text_embd
        
        return video_embd_att, text_embd_att, vis_attn_scores, text_attn_scores
    
def cross_attention(video_embd, text_embd, mask, train=True):
    softmax = nn.Softmax(dim=-1)
    num_samples = len(video_embd)
    if not train:
        scores = th.bmm(video_embd, text_embd.permute(0, 2, 1))
        #print('video_embd.size(-2)',video_embd.size(-2),video_embd.shape,mask.shape)
        lang_mask = mask.unsqueeze(1).repeat(1, video_embd.size(-2), 1)
        vis_att_scores = scores.masked_fill(lang_mask==False, -float('inf'))
        vis_att_scores = softmax(vis_att_scores)
        video_embd_att = text_embd.unsqueeze(1).repeat(1, video_embd.size(-2), 1, 1)
        video_embd_att = video_embd_att * vis_att_scores.unsqueeze(-1)
        video_embd_att = video_embd_att.sum(-2)
        
        text_att_scores = scores.permute(0, 2, 1)
        text_att_scores = softmax(text_att_scores)
        text_embd_att = video_embd.unsqueeze(1).repeat(1, text_embd.size(-2), 1, 1)
        text_embd_att = text_embd_att * text_att_scores.unsqueeze(-1)
        text_embd_att = text_embd_att.sum(-2)
        
        return video_embd_att, text_embd_att, vis_att_scores, text_att_scores
    if len(video_embd.size()) == 3:
        ##print('here') text #[16,20,512]
        scores = text_embd.view(-1, text_embd.size(-1)) #[320,512]
        #print('scores',scores.shape)
        scores = th.matmul(video_embd, scores.t()) #[16,98,320] #given region, attend to one of the word across batch
        #print('scores2',scores.shape)
        scores = scores / th.sqrt(th.tensor(video_embd.size(-1)).float()) #normalize /512
        #print('scores3',scores.shape)
        scores = scores.permute(0, 2, 1) #[16, 320, 98]
        #print('scores4',scores.shape)
        scores = scores.view(scores.size(0), len(text_embd), -1, scores.size(-1)) #[16, 16, 20, 98]
        #print('scores5',scores.shape) #given a text, find corresponding region

        text_att_scores = softmax(scores) #[16v, 16t, 20, 98] #given, v, t, word, which region is more important
        #every text word in the batch calculate similarity with one video 
        #print('text_att_scores',text_att_scores.shape)
        text_embd_att = video_embd.unsqueeze(1).unsqueeze(1) #[16, 1, 1, 98, 512]
        #print('text_embd_att',text_embd_att.shape)
        text_embd_att = text_embd_att.repeat(1, text_embd.size(0), text_embd.size(1), 1, 1) #[16v, 16t, 20, 98, 512]
        #print('text_embd_att',text_embd_att.shape)
        text_embd_att = text_embd_att * text_att_scores.unsqueeze(-1) #[16v, 16t, 20, 98, 512] visual context feature for text
        #print('text_embd_att',text_embd_att.shape)
        text_embd_att = text_embd_att.sum(-2) #[16, 16, 20, 98, 512]
        #print('text_embd_att',text_embd_att.shape) #[16, 16, 20, 512]

        # Computes region-attended word representations
        vis_att_scores = scores.permute(0, 1, 3, 2) #[16v, 16t, 98, 20] #given region, word importance
        #print('vis_att_scores',vis_att_scores.shape)
        lang_mask = mask.unsqueeze(0).unsqueeze(2) #[1, 16, 1, 20]
        #print('lang_mask',lang_mask.shape)
        lang_mask = lang_mask.repeat(num_samples, 1, video_embd.size(1), 1) #[16, 16, 98, 20]
        #print('lang_mask',lang_mask.shape)
        vis_att_scores = vis_att_scores.masked_fill(lang_mask == False, -float('inf')) #[16, 16, 98, 20]
        #print('vis_att_scores',vis_att_scores.shape)
        vis_att_scores = softmax(vis_att_scores)  #[16v, 16t, 98, 20] #given, v, t, region, which word is more important
        #print('vis_att_scores',vis_att_scores) 
        video_embd_att = text_embd.unsqueeze(0).unsqueeze(2)  #[1, 16, 1, 20, 512]
        #print('video_embd_att',video_embd_att)
        video_embd_att = video_embd_att.repeat(num_samples, 1, video_embd.size(1), 1, 1) #[16, 16, 98, 20, 512]
        #print('video_embd_att',video_embd_att)
        video_embd_att = video_embd_att * vis_att_scores.unsqueeze(-1) #[16, 16, 98, 20, 512]
        #print('video_embd_att',video_embd_att)
        video_embd_att = video_embd_att.sum(-2) #[16, 16, 98, 512] #contain misaligned pairs
        #print('video_embd_att',video_embd_att)
        
        video_embd = video_embd.unsqueeze(1).repeat(1, num_samples, 1, 1)
        #print('video_embd',video_embd.shape)
        text_embd = text_embd.unsqueeze(0).repeat(num_samples, 1, 1, 1)
        #print('text_embd',text_embd.shape)
        
        return video_embd_att, text_embd_att, vis_att_scores, text_att_scores
    else:
        ##print('there')
        scores = th.matmul(video_embd, text_embd.permute(0, 1, 3, 2)) #16,16,512,20 => #[16,16,98,20]
        #print('scores',scores.shape)
        scores = scores / th.sqrt(th.tensor(video_embd.size(-1)).float()) #[16,16,98,20]
        #print('scores',scores.shape)
        
        lang_mask = mask.unsqueeze(0).unsqueeze(2) #[1, 16, 1, 20]
        #print('lang_mask',lang_mask.shape)
        lang_mask = lang_mask.repeat(num_samples, 1, video_embd.size(-2), 1) #[16, 16, 98, 20]
        #print('lang_mask',lang_mask.shape)
        video_embd_att = text_embd.unsqueeze(2).repeat(1, 1, video_embd.size(-2), 1, 1) #[16, 16, 98, 20, 512]
        #print('video_embd_att',video_embd_att.shape)
        vis_att_scores = scores.masked_fill(lang_mask == False, -float('inf')) #[16,16,98,20]
        #print('vis_att_scores',vis_att_scores.shape)
        vis_att_scores = softmax(vis_att_scores) #[16,16,98,20]
        #print('vis_att_scores',vis_att_scores.shape)
        video_embd_att = video_embd_att * vis_att_scores.unsqueeze(-1) #[16, 16, 98, 20, 512]
        #print('video_embd_att',video_embd_att.shape)
        video_embd_att = video_embd_att.sum(-2) #[16, 16, 98, 512]
        #print('video_embd_att',video_embd_att.shape)
        
        text_att_scores = scores.permute(0, 1, 3, 2) #[16,16,20,98]
        #print('text_att_scores',text_att_scores.shape)
        text_att_scores = softmax(text_att_scores) #[16,16,20,98]
        #print('text_att_scores',text_att_scores.shape)
        text_embd_att = video_embd.unsqueeze(2).repeat(1, 1, text_embd.size(-2), 1, 1) #[16, 16, 20, 98, 512]
        #print('text_embd_att',text_embd_att.shape)
        text_embd_att = text_embd_att * text_att_scores.unsqueeze(-1) #[16, 16, 20, 98, 512]
        #print('text_embd_att',text_embd_att.shape)
        text_embd_att = text_embd_att.sum(-2) #[16, 16, 20, 512]
        #print('text_embd_att',text_embd_att.shape)
        
        video_embd_att = video_embd_att + video_embd
        #print('video_embd_att',video_embd_att.shape)
        text_embd_att = text_embd_att + text_embd
        #print('text_embd_att',text_embd_att.shape)
        
        return video_embd_att, text_embd_att, vis_att_scores, text_att_scores

class S3D_clip(nn.Module):

    def __init__(self, num_classes=512, gating=True, space_to_depth=False,
                  word2vec_path='', init='uniform', token_to_word_path='data/dict.npy',
                  global_feature=False,gating_feature=False,resnet=False,r50=False):
        super(S3D_clip, self).__init__()
        self.num_classes = num_classes
        self.gating = gating
        self.space_to_depth = space_to_depth
        

        # if space_to_depth:
        #     self.conv1 = STConv3D(
        #                 24, 64, [2, 4, 4], stride=1, padding=(1, 2, 2), separable=False)
        # else:
        #     self.conv1 = STConv3D(
        #                 3, 64, [3, 7, 7], stride=2, padding=(1, 3, 3), separable=False)
        # self.conv_2b = STConv3D(64, 64, [1, 1, 1], separable=False)
        # self.conv_2c = STConv3D(64, 192, [3, 3, 3], padding=1, separable=True)
        # self.gating = SelfGating(192)
        # self.maxpool_2a = MaxPool3dTFPadding(
        #     kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # self.maxpool_3a = MaxPool3dTFPadding(
        #     kernel_size=(1, 3, 3), stride=(1, 2, 2), padding='SAME')
        # self.mixed_3b = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        # self.mixed_3c = InceptionBlock(self.mixed_3b.output_dim, 128, 128, 192, 32, 96, 64)
        # self.maxpool_4a = MaxPool3dTFPadding(
        #                 kernel_size=(3, 3, 3), stride=(2, 2, 2), padding='SAME')
        # self.mixed_4b = InceptionBlock(self.mixed_3c.output_dim, 192, 96, 208, 16, 48, 64)
        # self.mixed_4c = InceptionBlock(self.mixed_4b.output_dim, 160, 112, 224, 24, 64, 64)
        # self.mixed_4d = InceptionBlock(self.mixed_4c.output_dim, 128, 128, 256, 24, 64, 64)
        # self.mixed_4e = InceptionBlock(self.mixed_4d.output_dim, 112, 144, 288, 32, 64, 64)
        # self.mixed_4f = InceptionBlock(self.mixed_4e.output_dim, 256, 160, 320, 32, 128, 128)
        # self.maxpool_5a = self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(
        #                 kernel_size=(2, 2, 2), stride=(2, 2, 2), padding='SAME')
        # self.mixed_5b = InceptionBlock(self.mixed_4f.output_dim, 256, 160, 320, 32, 128, 128)
        # self.mixed_5c = InceptionBlock(self.mixed_5b.output_dim, 384, 192, 384, 48, 128, 128)
        # self.text_module = Sentence_Embedding(
        #                        num_classes,
        #                        os.path.join(os.path.dirname(__file__), token_to_word_path),
        #                        word2vec_path=os.path.join(os.path.dirname(__file__), word2vec_path))
        if resnet: #use resnet
            self.vis_dim_local = 2048
        else:
            self.vis_dim_local = 512#1024
        if r50:
            self.text_dim_local = 1024
        else:
            self.text_dim_local = 512#2048
        self.vis_proj_mat = nn.Sequential(nn.Linear(self.vis_dim_local, num_classes))
        self.lang_proj_mat = nn.Sequential(nn.Linear(self.text_dim_local, num_classes))
        
        self.ca_layer1 = CAEncoder()
        self.ca_layer2 = CAEncoder()
        self.sa_layer = SAEncoder()

        if r50:
            self.vis_dim = 1024
            self.text_dim = 1024 
        else:
            self.vis_dim = 512
            self.text_dim = 512
        if global_feature:
            if gating_feature: 
                self.vis_proj_mat_g = Gated_Embedding_Unit(self.text_dim, num_classes)
                self.lang_proj_mat_g = Gated_Embedding_Unit(self.text_dim, num_classes)
            else:
                self.vis_proj_mat_g = nn.Sequential(nn.Linear(self.vis_dim, num_classes))
                self.lang_proj_mat_g = nn.Sequential(nn.Linear(self.text_dim, num_classes))
        num_heads = 1

        if init == 'kaiming_normal':
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight,
                                            mode='fan_in',
                                            nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _space_to_depth(self, input):
      try:
          B, C, T, H, W = input.shape
      except:
          print('input.shape',input.shape)
      input = input.view(B, C, T // 2, 2, H // 2, 2, W // 2, 2)
      input = input.permute(0, 3, 5, 7, 1,  2, 4, 6)
      input = input.contiguous().view(B, 8 * C, T // 2, H // 2, W // 2)
      return input

    def forward(self, video, video_region, global_t, text, mask, mode='all', mixed5c=False, att=False):
      if mode == 'all':  
          num_samples = len(video)
          #text_embd = self.text_module(text, mask)
          #print('text',text.shape)
          text_embd = self.lang_proj_mat(text) #512->512

          #video_embd = self.forward_video(video)
          
          #num_temporal_steps = video_embd.size(1)

          video_embd = th.flatten(video, start_dim=1, end_dim=-2)

          video_embd = self.vis_proj_mat(video_embd) #[16, 98, 512], #16,16*49,512
          #print('video_embd',video_embd)
          #print('text_embd',text_embd)
          video_embd_att, text_embd_att, _, _ = self.ca_layer1(video_embd, text_embd, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          video_embd_att, text_embd_att, _, _ = self.sa_layer(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          final_video_embd_att, final_text_embd_att, _, _ = self.ca_layer2(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          #print('final_video_embd_att',final_video_embd_att.shape,'final_text_embd_att',final_text_embd_att.shape)
          # Computes loss
          mask = mask.unsqueeze(0).repeat(num_samples, 1, 1)
          final_text_embd_att = final_text_embd_att.masked_fill(mask.unsqueeze(-1) == False, 0.)
          final_text_embd_att = final_text_embd_att.sum(-2)
          final_text_embd_att = final_text_embd_att / mask.sum(-1).unsqueeze(-1).float()
          final_video_embd_att = final_video_embd_att.mean(-2)
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          return final_video_embd_att, final_text_embd_att
      elif mode == 'no_global':  
          #print('no global')
          num_samples = len(video)
          #text_embd = self.text_module(text, mask)
          #text_embd_g = th.max(text, dim=1)[0]
          text_embd_g = self.lang_proj_mat_g(global_t)


          text_embd = self.lang_proj_mat(text)
          

          #video_embd = self.forward_video(video)
          video_embd_g = video#th.mean(video_embd, dim=[1, 2, 3]) #[b,1024]
          #print('video',video.shape)
          video_embd_g= self.vis_proj_mat_g(video_embd_g)
          #print('video_embd_g',video_embd_g.shape)

          #num_temporal_steps = video_embd.size(1)
          video_embd = th.flatten(video_region, start_dim=1, end_dim=-2)
          video_embd = self.vis_proj_mat(video_embd) #[16, 98, 512]
          #print('video_embd',video_embd)
          #print('text_embd',text_embd)
          video_embd_att, text_embd_att, _, _ = self.ca_layer1(video_embd, text_embd, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          video_embd_att, text_embd_att, _, _ = self.sa_layer(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          final_video_embd_att, final_text_embd_att, _, _ = self.ca_layer2(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          #print('final_video_embd_att',final_video_embd_att.shape,'final_text_embd_att',final_text_embd_att.shape)
          # Computes loss
          mask = mask.unsqueeze(0).repeat(num_samples, 1, 1)
          final_text_embd_att = final_text_embd_att.masked_fill(mask.unsqueeze(-1) == False, 0.)
          final_text_embd_att = final_text_embd_att.sum(-2)
          final_text_embd_att = final_text_embd_att / mask.sum(-1).unsqueeze(-1).float()
          final_video_embd_att = final_video_embd_att.mean(-2)
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          return final_video_embd_att, final_text_embd_att, video_embd_g, text_embd_g

      elif mode == 'global':  
          num_samples = len(video)
          #text_embd = self.text_module(text, mask)
          text_embd_g = th.max(text, dim=1)[0]
          text_embd_g = self.lang_proj_mat_g(text_embd_g)


          text_embd = self.lang_proj_mat(text)
          

          #video_embd = self.forward_video(video)
          video_embd_g = video#th.mean(video_embd, dim=[1, 2, 3]) #[b,1024]
          #print('video',video.shape)
          video_embd_g= self.vis_proj_mat_g(video_embd_g)
          #print('video_embd_g',video_embd_g.shape)

          #num_temporal_steps = video_embd.size(1)
          video_embd = th.flatten(video_region, start_dim=1, end_dim=-2)
          video_embd = self.vis_proj_mat(video_embd) #[16, 98, 512]
          #print('video_embd',video_embd)
          #print('text_embd',text_embd)
          video_embd_att, text_embd_att, _, _ = self.ca_layer1(video_embd, text_embd, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          video_embd_att, text_embd_att, _, _ = self.sa_layer(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          final_video_embd_att, final_text_embd_att, _, _ = self.ca_layer2(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          #print('final_video_embd_att',final_video_embd_att.shape,'final_text_embd_att',final_text_embd_att.shape)
          # Computes loss
          mask = mask.unsqueeze(0).repeat(num_samples, 1, 1)
          final_text_embd_att = final_text_embd_att.masked_fill(mask.unsqueeze(-1) == False, 0.)
          final_text_embd_att = final_text_embd_att.sum(-2)
          final_text_embd_att = final_text_embd_att / mask.sum(-1).unsqueeze(-1).float()
          final_video_embd_att = final_video_embd_att.mean(-2)
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          return final_video_embd_att, final_text_embd_att, video_embd_g, text_embd_g
      elif mode == 'global_text':    
          num_samples = len(video)
          #text_embd = self.text_module(text, mask)
          #text_embd_g = th.max(text, dim=1)[0]
          text_embd_g = self.lang_proj_mat_g(global_t)


          text_embd = self.lang_proj_mat(text)
          

          #video_embd = self.forward_video(video)
          video_embd_g = video#th.mean(video_embd, dim=[1, 2, 3]) #[b,1024]
          #print('video',video.shape)
          video_embd_g= self.vis_proj_mat_g(video_embd_g)
          #print('video_embd_g',video_embd_g.shape)

          #num_temporal_steps = video_embd.size(1)
          video_embd = th.flatten(video_region, start_dim=1, end_dim=-2)
          video_embd = self.vis_proj_mat(video_embd) #[16, 98, 512]
          #print('video_embd',video_embd)
          #print('text_embd',text_embd)
          video_embd_att, text_embd_att, _, _ = self.ca_layer1(video_embd, text_embd, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          video_embd_att, text_embd_att, _, _ = self.sa_layer(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          final_video_embd_att, final_text_embd_att, _, _ = self.ca_layer2(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          #print('final_video_embd_att',final_video_embd_att.shape,'final_text_embd_att',final_text_embd_att.shape)
          # Computes loss
          mask = mask.unsqueeze(0).repeat(num_samples, 1, 1)
          final_text_embd_att = final_text_embd_att.masked_fill(mask.unsqueeze(-1) == False, 0.)
          final_text_embd_att = final_text_embd_att.sum(-2)
          final_text_embd_att = final_text_embd_att / mask.sum(-1).unsqueeze(-1).float()
          final_video_embd_att = final_video_embd_att.mean(-2)
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          return final_video_embd_att, final_text_embd_att, video_embd_g, text_embd_g
      elif mode == 'global_only':    
          num_samples = len(video)
          #text_embd = self.text_module(text, mask)
          #text_embd_g = th.max(text, dim=1)[0]
          text_embd_g = self.lang_proj_mat_g(global_t)


          text_embd = self.lang_proj_mat(text)
          

          #video_embd = self.forward_video(video)
          video_embd_g = video#th.mean(video_embd, dim=[1, 2, 3]) #[b,1024]
          #print('video',video.shape)
          video_embd_g= self.vis_proj_mat_g(video_embd_g)
          #print('video_embd_g',video_embd_g.shape)

          
          return video_embd_g, text_embd_g

      elif mode == 'global_l2':  
          num_samples = len(video)
          text_embd = self.text_module(text, mask)
          text_embd_g = th.max(text_embd, dim=1)[0]
          text_embd_g = self.lang_proj_mat_g(text_embd_g)
          text_embd = self.lang_proj_mat(text_embd)
          

          video_embd = self.forward_video(video)
          video_embd_g = th.mean(video_embd, dim=[1, 2, 3]) #[b,1024]
          video_embd_g= self.vis_proj_mat_g(video_embd_g)

          num_temporal_steps = video_embd.size(1)
          video_embd = th.flatten(video_embd, start_dim=1, end_dim=-2)
          video_embd = self.vis_proj_mat(video_embd) #[16, 98, 512]
          #print('video_embd',video_embd)
          #print('text_embd',text_embd)
          video_embd_att, text_embd_att, _, _ = self.ca_layer1(video_embd, text_embd, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          video_embd_att, text_embd_att, _, _ = self.sa_layer(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          final_video_embd_att, final_text_embd_att, final_video_embd_att_score, final_text_embd_att_score = self.ca_layer2(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          #print('final_video_embd_att',final_video_embd_att.shape,'final_text_embd_att',final_text_embd_att.shape)
          # Computes loss
          mask = mask.unsqueeze(0).repeat(num_samples, 1, 1)
          final_text_embd_att = final_text_embd_att.masked_fill(mask.unsqueeze(-1) == False, 0.)
          final_text_embd_att = final_text_embd_att.sum(-2)
          final_text_embd_att = final_text_embd_att / mask.sum(-1).unsqueeze(-1).float()
          final_video_embd_att = final_video_embd_att.mean(-2)
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          return final_video_embd_att, final_text_embd_att, video_embd_g, text_embd_g, final_video_embd_att_score, final_text_embd_att_score

      elif mode == 'global_l2_all':  
          num_samples = len(video)
          #text_embd = self.text_module(text, mask)
          text_embd_g = th.max(text, dim=1)[0]
          text_embd_g = self.lang_proj_mat_g(text_embd_g)


          text_embd = self.lang_proj_mat(text)
          

          #video_embd = self.forward_video(video) 
          video_embd_g = video#th.mean(video_embd, dim=[1, 2, 3]) #[b,1024]
          #print('video',video.shape)
          video_embd_g= self.vis_proj_mat_g(video_embd_g)
          #print('video_embd_g',video_embd_g.shape)

          #num_temporal_steps = video_embd.size(1)
          video_embd = th.flatten(video_region, start_dim=1, end_dim=-2)
          video_embd = self.vis_proj_mat(video_embd) #[16, 98, 512]
          #print('video_embd',video_embd.shape)
          #print('text_embd',text_embd.shape)
          video_embd_att, text_embd_att, v_cross_att_1, t_cross_att_1 = self.ca_layer1(video_embd, text_embd, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          video_embd_att, text_embd_att, v_self_att_1, t_self_att_1 = self.sa_layer(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          final_video_embd_att, final_text_embd_att, v_cross_att_2, t_cross_att_2 = self.ca_layer2(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          #print('final_video_embd_att',final_video_embd_att.shape,'final_text_embd_att',final_text_embd_att.shape)
          # Computes loss
          mask = mask.unsqueeze(0).repeat(num_samples, 1, 1)
          final_text_embd_att = final_text_embd_att.masked_fill(mask.unsqueeze(-1) == False, 0.)
          final_text_embd_att = final_text_embd_att.sum(-2)
          final_text_embd_att = final_text_embd_att / mask.sum(-1).unsqueeze(-1).float()
          final_video_embd_att = final_video_embd_att.mean(-2)
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          return final_video_embd_att, final_text_embd_att, video_embd_g, text_embd_g, v_cross_att_1, t_cross_att_1, v_self_att_1, t_self_att_1 , v_cross_att_2, t_cross_att_2


      elif mode == 'global_within':  
          num_samples = len(video)
          #text_embd = self.text_module(text, mask)
          text_embd_g = th.max(text, dim=1)[0]
          text_embd_g = self.lang_proj_mat_g(text_embd_g)


          text_embd = self.lang_proj_mat(text)
          

          #video_embd = self.forward_video(video)
          video_embd_g = video#th.mean(video_embd, dim=[1, 2, 3]) #[b,1024]
          #print('video',video.shape)
          video_embd_g= self.vis_proj_mat_g(video_embd_g)
          #print('video_embd_g',video_embd_g.shape)

          num_temporal_steps = video_region.size(1)
          video_embd = th.flatten(video_region, start_dim=1, end_dim=-2)
          video_embd = self.vis_proj_mat(video_embd) #[16, 98, 512]
          #print('video_embd',video_embd)
          #print('text_embd',text_embd)
          video_embd_att, text_embd_att, _, _ = self.ca_layer1(video_embd, text_embd, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          video_embd_att, text_embd_att, _, _ = self.sa_layer(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          final_video_embd_att, final_text_embd_att, _, _ = self.ca_layer2(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          #print('final_video_embd_att',final_video_embd_att.shape,'final_text_embd_att',final_text_embd_att.shape)
          # Computes loss
          mask = mask.unsqueeze(0).repeat(num_samples, 1, 1)
          final_text_embd_att = final_text_embd_att.masked_fill(mask.unsqueeze(-1) == False, 0.)
          final_text_embd_att = final_text_embd_att.sum(-2)
          final_text_embd_att = final_text_embd_att / mask.sum(-1).unsqueeze(-1).float()
          final_video_embd_att_region = final_video_embd_att
          final_video_embd_att = final_video_embd_att.mean(-2)
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          return final_video_embd_att, final_text_embd_att, video_embd_g, text_embd_g, final_video_embd_att_region

      elif mode == 'global_within_l2':  
          num_samples = len(video)
          text_embd = self.text_module(text, mask)
          text_embd_g = th.max(text_embd, dim=1)[0]
          text_embd_g = self.lang_proj_mat_g(text_embd_g)
          text_embd = self.lang_proj_mat(text_embd)
          

          video_embd = self.forward_video(video)
          video_embd_g = th.mean(video_embd, dim=[1, 2, 3]) #[b,1024]
          video_embd_g= self.vis_proj_mat_g(video_embd_g)

          num_temporal_steps = video_embd.size(1)
          video_embd = th.flatten(video_embd, start_dim=1, end_dim=-2)
          video_embd = self.vis_proj_mat(video_embd) #[16, 98, 512]
          #print('video_embd',video_embd)
          #print('text_embd',text_embd)
          video_embd_att, text_embd_att, _, _ = self.ca_layer1(video_embd, text_embd, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          video_embd_att, text_embd_att, _, _ = self.sa_layer(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          final_video_embd_att, final_text_embd_att, v_cross_att_2, t_cross_att_2 = self.ca_layer2(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          #print('final_video_embd_att',final_video_embd_att.shape,'final_text_embd_att',final_text_embd_att.shape)
          # Computes loss
          mask = mask.unsqueeze(0).repeat(num_samples, 1, 1)
          final_text_embd_att = final_text_embd_att.masked_fill(mask.unsqueeze(-1) == False, 0.)
          final_text_embd_att = final_text_embd_att.sum(-2)
          final_text_embd_att = final_text_embd_att / mask.sum(-1).unsqueeze(-1).float()
          final_video_embd_att_region = final_video_embd_att
          final_video_embd_att = final_video_embd_att.mean(-2)
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          return final_video_embd_att, final_text_embd_att, video_embd_g, text_embd_g, final_video_embd_att_region, v_cross_att_2, t_cross_att_2
      elif mode == 'global_within_l2_all':  
          num_samples = len(video)
          text_embd = self.text_module(text, mask)
          text_embd_g = th.max(text_embd, dim=1)[0]
          text_embd_g = self.lang_proj_mat_g(text_embd_g)
          text_embd = self.lang_proj_mat(text_embd)
          

          video_embd = self.forward_video(video)
          video_embd_g = th.mean(video_embd, dim=[1, 2, 3]) #[b,1024]
          video_embd_g= self.vis_proj_mat_g(video_embd_g)

          num_temporal_steps = video_embd.size(1)
          video_embd = th.flatten(video_embd, start_dim=1, end_dim=-2)
          video_embd = self.vis_proj_mat(video_embd) #[16, 98, 512]
          #print('video_embd',video_embd)
          #print('text_embd',text_embd)
          video_embd_att, text_embd_att, v_cross_att_1, t_cross_att_1 = self.ca_layer1(video_embd, text_embd, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          video_embd_att, text_embd_att, v_self_att_1, t_self_att_1 = self.sa_layer(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('video_embd_att',video_embd_att)
          #print('text_embd_att',text_embd_att)
          #print('video_embd_att',video_embd_att.shape,'text_embd_att',text_embd_att.shape)
          final_video_embd_att, final_text_embd_att, v_cross_att_2, t_cross_att_2 = self.ca_layer2(video_embd_att, text_embd_att, mask) #[16, 16, 98, 512]
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          #print('final_video_embd_att',final_video_embd_att.shape,'final_text_embd_att',final_text_embd_att.shape)
          # Computes loss
          mask = mask.unsqueeze(0).repeat(num_samples, 1, 1)
          final_text_embd_att = final_text_embd_att.masked_fill(mask.unsqueeze(-1) == False, 0.)
          final_text_embd_att = final_text_embd_att.sum(-2)
          final_text_embd_att = final_text_embd_att / mask.sum(-1).unsqueeze(-1).float()
          final_video_embd_att_region = final_video_embd_att
          final_video_embd_att = final_video_embd_att.mean(-2)
          
          #print('final_video_embd_att',final_video_embd_att)
          #print('final_text_embd_att',final_text_embd_att)
          return final_video_embd_att, final_text_embd_att, video_embd_g, text_embd_g, \
          v_cross_att_1, t_cross_att_1, v_self_att_1, t_self_att_1 , v_cross_att_2, t_cross_att_2, final_video_embd_att_region

      elif mode == 'eval':
          #print('in')
          #text_embd = self.text_module(text, mask)
          #video_embd = self.forward_video(video)
          #num_temporal_steps = video_embd.size(1)
          video_embd = th.flatten(video_region, start_dim=1, end_dim=-2)
          num_regions = video_embd.size(1)
          text_embd = self.lang_proj_mat(text)
          num_words = text_embd.size(1)
          video_embd =  self.vis_proj_mat(video_embd) #project region features
        
          text_embd = text_embd.repeat(len(video_embd), 1, 1)
          video_embd_att, text_embd_att, vis_scores, text_scores = self.ca_layer1(video_embd, text_embd, mask)
          video_embd_att, text_embd_att, sa_vis_scores, sa_text_scores = self.sa_layer(video_embd_att, text_embd_att, mask)
          final_video_embd_att, final_text_embd_att, vis_scores2, text_scores2 = self.ca_layer2(video_embd_att, text_embd_att, mask)
          #[length,length,98,512]
          mask = mask.repeat(len(video_embd), 1) 

          vis_scores2 = vis_scores2.masked_fill(mask.unsqueeze(1).repeat(1, vis_scores2.size(1), 1)==False, 0.)
          sa_text_scores = sa_text_scores.masked_fill(mask.unsqueeze(1).repeat(1, sa_text_scores.size(1), 1)==False, 0.)
          text_scores = text_scores.masked_fill(mask.unsqueeze(-1).repeat(1, 1, text_scores.size(-1))==False, 0.)
          text_scores2 = text_scores2.masked_fill(mask.unsqueeze(-1).repeat(1, 1, text_scores.size(-1))==False, 0.)
        #   print('vis_scores2',vis_scores2.shape) #[2, 98, 20]
        #   print('sa_text_scores',sa_text_scores.shape) #([2, 20, 20]
        #   print('text_scores',text_scores.shape) #([2, 20, 98]
          result = th.bmm(vis_scores2, sa_text_scores) #[2, 98, 20]
          #print('result',result.shape)
          result = th.bmm(result, text_scores) #[2, 98, 98]
          #print('result',result.shape)
          result = result.mean(1) #[2, 98]
          #print('result',result.shape)
          
          return result
      elif mode == 'eval_notrain':
          #print('in')
          #text_embd = self.text_module(text, mask)
          #video_embd = self.forward_video(video)
          #num_temporal_steps = video_embd.size(1)
          video_embd = th.flatten(video_region, start_dim=1, end_dim=-2) #b * region * 512
          num_regions = video_embd.size(1)
          text_embd = text#self.lang_proj_mat(text) # b * 20 * 512
          num_words = text_embd.size(1) 
          #video_embd =  self.vis_proj_mat(video_embd) #project region features
        
          text_embd = text_embd.repeat(len(video_embd), 1, 1)
          video_embd_att, text_embd_att, vis_scores, text_scores = self.ca_layer1(video_embd, text_embd, mask)
          video_embd_att, text_embd_att, sa_vis_scores, sa_text_scores = self.sa_layer(video_embd_att, text_embd_att, mask)
          
          final_video_embd_att, final_text_embd_att, vis_scores2, text_scores2 = self.ca_layer2(video_embd_att, text_embd_att, mask)
          #[length,length,98,512]
          mask = mask.repeat(len(video_embd), 1) 

          vis_scores2 = vis_scores2.masked_fill(mask.unsqueeze(1).repeat(1, vis_scores2.size(1), 1)==False, 0.)
          sa_text_scores = sa_text_scores.masked_fill(mask.unsqueeze(1).repeat(1, sa_text_scores.size(1), 1)==False, 0.)
          text_scores = text_scores.masked_fill(mask.unsqueeze(-1).repeat(1, 1, text_scores.size(-1))==False, 0.)
          text_scores2 = text_scores2.masked_fill(mask.unsqueeze(-1).repeat(1, 1, text_scores.size(-1))==False, 0.)
        #   print('vis_scores2',vis_scores2.shape) #[2, 98, 20]
        #   print('sa_text_scores',sa_text_scores.shape) #([2, 20, 20]
        #   print('text_scores',text_scores.shape) #([2, 20, 98]
          result = th.bmm(vis_scores2, sa_text_scores) #[2, 98, 20]
          #print('result',result.shape)
          result = th.bmm(result, text_scores) #[2, 98, 98]
          #print('result',result.shape)
          #text_embd = text_embd.permute(0,2,1)
          #print('text_embd',text_embd.shape)
          #print('video_embd',video_embd.shape)
          #result = th.bmm(video_embd, text_embd) #b * region *20
          result = result.mean(1) #[2, 98]
          #print('result',result.shape)
          
          return result
          
      elif mode == 'video':
          return self.forward_video(video, mixed5c=mixed5c)
      elif mode == 'text':
          return self.text_module(text)
      else:
          raise NotImplementedError

    def forward_video(self, inputs, mixed5c=False):
      #out = {}
      if self.space_to_depth:
        inputs = self._space_to_depth(inputs)
      # 'Conv2d_1a_7x7'
      net = self.conv1(inputs)
      if self.space_to_depth:
        net = net[:, :, 1:, 1:, 1:]
      #out['Conv2d_1a_7x7'] = net
      # 'MaxPool_2a_3x3'
      net = self.maxpool_2a(net)
      #out['MaxPool_2a_3x3'] = net
      #'Conv2d_2b_1x1'
      net = self.conv_2b(net)
      #out['Conv2d_2b_1x1'] = net
      # 'Conv2d_2c_3x3'
      net = self.conv_2c(net)
      #out['Conv2d_2c_3x3'] = net
      if self.gating:
          net = self.gating(net)
          #out['gating_1'] = net
      # 'MaxPool_3a_3x3'
      net = self.maxpool_3a(net)
      #out['MaxPool_3a_3x3'] = net
      # end_point = 'Mixed_3b'
      net = self.mixed_3b(net)
      #out['Mixed_3b'] = net
      # end_point = 'Mixed_3c'
      net = self.mixed_3c(net)
      #out['Mixed_3c'] = net
      # end_point = 'MaxPool_4a_3x3'
      net = self.maxpool_4a(net)
      #out['MaxPool_4a_3x3'] = net
      # end_point = 'Mixed_4b'
      net = self.mixed_4b(net)
      #out['Mixed_4b'] = net
      # end_point = 'Mixed_4c'
      net = self.mixed_4c(net)
    
      #out['Mixed_4c'] = net
      # end_point = 'Mixed_4d'
      net = self.mixed_4d(net)
      #out['Mixed_4d'] = net
      # end_point = 'Mixed_4e'
      net = self.mixed_4e(net)
      #out['Mixed_4e'] = net
      # end_point = 'Mixed_4f'
      net = self.mixed_4f(net)
        
      #out['Mixed_4f'] = net
      #end_point = 'MaxPool_5a_2x2'
      net = self.maxpool_5a(net)
      #out['MaxPool_5a_2x2'] = net
      # end_point = 'Mixed_5b'
      net = self.mixed_5b(net)
      #out['Mixed_5b'] = net
      # end_point = 'Mixed_5c'
      net = self.mixed_5c(net)  
    
      net = net.permute(0, 2, 3, 4, 1)
      return net
    
      #out['Mixed_5c'] = net
      #out['Avgpool'] = net
      net = th.mean(net, dim=[2, 3, 4])
      if mixed5c:
          return net
      net = self.fc(net)
      #out['final'] = net
      return net

if __name__ == "__main__":
    #text torch.Size([32, 20])
    #vid torch.Size([32, 3, 16, 224, 224])
    model = S3D()
    batch_size = 16
    video = th.rand((batch_size,3,16,224,224))
    # p, out = model(x)
    #video_embd = model.forward_video(video)
    ##print('video_embd',video_embd.shape) 
    # video_embd = th.flatten(video_embd, start_dim=1, end_dim=-2)
    # #print(video_embd.shape) 
    # num_regions = video_embd.size(1)
    # #print(num_regions) 
    text = th.zeros((batch_size,20), dtype=th.long)
    mask = th.ones((batch_size,20), dtype=th.bool)
    final_video_embd_att, final_text_embd_att = model(video,text,mask)
    #print('final_video_embd_att, final_text_embd_att',final_video_embd_att.shape, final_text_embd_att.shape)