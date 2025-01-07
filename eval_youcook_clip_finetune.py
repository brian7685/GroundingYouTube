import os
import random
import time
import sys
import numpy as np
import pickle

import torch
import torch.utils.data
import torch.nn.functional as F

from args import get_args
from youcook_interactions_loader import Youcook_DataLoader
from youcook_interactions_loader_clip import Youcook_DataLoader_clip
import clip
from clip_feature_noModule  import  _apply_clip_text_model
from clip_feature_noModule  import  _apply_clip_image_model
from clip_feature_noModule import _apply_clip_text_model_prompt

import s3dg_clip
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import scipy
import scipy.ndimage

def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    '''print('inter: ' + str(inter))
    print('union: ' + str(union))
    print('')'''
    return inter / union  # [A,B]

def main():
    args = get_args()
    print("=> loading checkpoint '{}'".format(args.checkpoint_eval))
    checkpoint = torch.load(args.checkpoint_eval)
    
    if args.gating_feature:
        print('gating features')
        model = s3dg_clip.S3D_clip(
            args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, 
            init=args.weight_init, global_feature=True,gating_feature=True,  resnet=args.resnet, r50=args.r50
        )
    else:
        model = s3dg_clip.S3D_clip(
                    args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, 
                    init=args.weight_init, global_feature=True, resnet=args.resnet, r50=args.r50
                )
    #model = torch.nn.DataParallel(model)
    #net_data = torch.load(args.pretrain_cnn_path, map_location='cpu')
    state_dict = checkpoint["state_dict"]
    new_dict = {}
    for k,v in state_dict.items():
        #k = k.replace('encoder_q.encoder.', 'backbone.')
        k = k.replace('module.', '')
        k = k.replace('text_word_embd.', 'text_module.word_embd.')
        k = k.replace('text_fc1.', 'text_module.fc1.')
        k = k.replace('lang_key_mat.', 'lang_proj_mat.')
        k = k.replace('key_mat.', 'vis_proj_mat.')
        k = k.replace('vis_attn_mat.', 'sa_layer.attn_mat.')
        #k = k.replace('', '')
        new_dict[k] = v
    state_dict = new_dict
    #model.load_state_dict(checkpoint["state_dict"])
    #model.load_state_dict(state_dict, strict=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.cuda()
    print('test clip loader')
    # Data loading code
    dataset = Youcook_DataLoader_clip(
        args,
        data=os.path.join(os.path.dirname(__file__), 'csv/validation_youcook.csv'),
        num_clip=args.num_windows_test,
        video_root=args.eval_video_root,
        fps=args.fps,
        num_frames=args.num_frames,
        size=args.video_size,
        crop_only=False,
        center_crop=True 
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, 
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=0,
    # )

    print('use CLIP')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.resnet:
        if args.r50:
            print('resnet 50')
            CLIP_model, preprocess = clip.load("RN50", device=device)
        else:
            print('resnet 101')
            CLIP_model, preprocess = clip.load("RN101", device=device)
    else:
        pretrained_dict = {}
        CLIP_model, preprocess = clip.load("ViT-B/32", device=device)
        checkpoint = torch.load(args.pretrain_clip)

        state_dict = checkpoint["state_dict"]
        new_dict = {}
        for k,v in state_dict.items():
            #k = k.replace('encoder_q.encoder.', 'backbone.')
            #print('bu k',k)
            #if 'text_module.fc1' in k:
            #    print('text_module.fc1 bu val',v[0])
                
            k = k.replace('module.', '')
            
            new_dict[k] = v
        checkpoint = new_dict
        for k in CLIP_model.state_dict():
            #k = k.replace('module.', '')
            # if 'DAVEnet.bn1' in k:
            #     print('DAVEnet.bn1 before val',model.state_dict()[0])
            # if 'lang_proj_mat_g' in k:
            #     print('lang_proj_mat_g before val',model.state_dict()[0])
            if k in checkpoint:
                pretrained_dict[k] = checkpoint[k]
            else:
                #print('not bu k',k)
                pretrained_dict[k] = model.state_dict()[k]
        print('loaded self-trained CLIP: ', args.pretrain_clip)
        CLIP_model.load_state_dict(pretrained_dict)

    num_valid = 0.
    num_correct = 0.
    skip = 0.
    with torch.no_grad():
        for i_batch, data in enumerate(tqdm(dataloader)):
            # if i_batch % 10 == 0:
            #     print(i_batch)
            name = data['name'][0]
            seg = data['segment']
            text = data['text'].cuda()
            
            video = data['video'].float().cuda().squeeze(0)
            frame_indices = data['frame_indices']
            video = video / 255.0
            mask = data['mask'].cuda()
            word_idx = data['idx'].cuda()
            gt = data['gt']
            width = data['width']
            height = data['height']
            #print('text',text.shape)
            #print('video',video.shape) #video torch.Size([2, 3, 16, 224, 224])
            #print('mask',mask.shape)
            batch_size = video.shape[0]
            image = video.permute(0, 2, 1,3, 4).half() 
            image = torch.flatten(image, start_dim=0, end_dim=1)
            if args.prompt:
                #print('prompt')
                text, mask, global_t = _apply_clip_text_model_prompt(CLIP_model, data, args.gpu, args.pin_memory)
                #mask = data["mask"].cuda()
            else:
                text, mask, global_t = _apply_clip_text_model(CLIP_model, data, args.gpu, args.pin_memory)

            global_v,local_v = _apply_clip_image_model(CLIP_model, image, args.gpu, args.pin_memory, args.resnet)
            if args.r50:
                global_v = global_v.view(batch_size,-1,1024)
            else:
                global_v = global_v.view(batch_size,-1,512) #[32, 16, 512]

            #mean pool, various pooling method should go here
            #print('global_v',global_v.shape)
            global_v= global_v.mean(-2)
            if args.resnet:
                local_v = local_v.view(batch_size,-1,49,2048)
            else:
                local_v = local_v.view(batch_size,-1,49,512) #[32, 16, 49, 512]
        
            if args.notrain:
                weights = model(global_v, local_v, global_t, text, mask, mode='eval_notrain')
            else:
                weights = model(global_v, local_v,global_t, text, mask, mode='eval')
            #print('weights')
            #print('weights',weights.shape)
            weights = torch.reshape(weights, (-1, 7, 7)) #reshape to time * height * width
            #print('weights',weights.shape)
            weights = weights.unsqueeze(0).unsqueeze(0) #[1,1,2,7,7]
            #print('weights',weights.shape)
            upsampled = F.interpolate(weights, size=(video.size(0) * video.size(2), height, width), mode='trilinear')
            #print('upsampled',upsampled.shape) #([1, 1, 16, 360, 640] upsample to video resolution
            upsampled = upsampled.squeeze()
            #print('upsampled.shape[0]',upsampled.shape[0])
            if upsampled.shape[0]==360:
                upsampled=upsampled.unsqueeze(0) 
            #print('upsampled sqz',upsampled.shape) #[1, 1, 16, 360, 640]
            
            selected = []
            for j in range(len(upsampled)):
                if j % 1 == 0:
                    tmp = upsampled[j]
                    selected.append(tmp.unsqueeze(0))
            selected = torch.cat(selected, dim=0)
            #print('selected',selected.shape) #[1, 1, 16, 360, 640]
            selected = selected.cpu().numpy()
            
            #print('gt.keys()',gt.keys(),len(gt.keys()),len(selected))
            # (xbr, ybr, xtl, ytl, outside, occluded)
            for j, frame_num in enumerate(list(gt.keys())): #eval when there is annotation
                curr_gt = [gt[frame_num]]
                if j < len(selected):
                    curr_frame = selected[j]                
                    
                    index = np.unravel_index(curr_frame.argmax(), curr_frame.shape)
                    #print('index',index)
                valid = False
                #print('curr_gt',curr_gt)
                for k in curr_gt:
                    xtl = k[0]
                    ytl = k[1]
                    xbr = k[2]
                    ybr = k[3]
                    outside = k[4]
                    occluded = k[5]
                    if outside == 1 or occluded == 1:
                        continue
                        
                    num_valid += 1.
                    
                    if index[1] >= xtl and index[1] <= xbr and index[0] >= ytl and index[0] <= ybr:
                        valid = True
                        break                   
                        
                if valid:
                    num_correct += 1.
            print('acc = num_correct / num_valid',num_correct / num_valid)

    acc = num_correct / num_valid
    acc *= 100.
    print('localization accuracy: ' + str(acc))

if __name__ == "__main__":
    main()
