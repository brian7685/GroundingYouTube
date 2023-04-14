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
from clip_feature  import  _apply_clip_text_model
from clip_feature  import  _apply_clip_image_model
from clip_feature import _apply_clip_text_model_prompt
from mining_loader import Mining_DataLoader

import s3dg_clip
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
import scipy
import scipy.ndimage
from pathlib import Path

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
    dataset = Mining_DataLoader(
        args,
        data=os.path.join(os.path.dirname(__file__), 'csv/validation_youcook.csv'),
        num_clip=args.num_windows_test,
        video_root=args.eval_video_root,
        fps=args.fps,
        num_frames=args.num_frames,
        size=args.video_size,
        crop_only=False,
        center_crop=True,
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
        CLIP_model, preprocess = clip.load("ViT-B/32", device=device)

    num_valid = 0.
    num_correct = 0.
    iou_total = 0
    iou_total_sat = 0
    iou_total_wide = 0
    iou_total_sat_n = 0
    iou_total_wide_n = 0
    meanAP = 0
    skip = 0.
    iou_write = set()
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
                    
                    print('index',index)
                    print('curr_frame',curr_frame.shape,curr_frame[index[0]][index[1]])
                    thre = args.thre#0.0001
                    curr_frame[curr_frame<thre]=0
                    curr_frame[curr_frame>=thre]=1
                    #l = max(0,index[0]
                    #r = index[1]

                # curr_frame_t = np.zeros((width,height))
                # l = max(0,index[1]-100)
                # r = min(width,index[1]+100)
                # t = max(0,index[0]-100)
                # b = min(height,index[0]+100)
                # for i in range(width):
                    
                #     if i>l and i<=r +1:
                #         curr_frame_t[i][t:b+1]=1
                valid = False
                print('curr_gt',curr_gt)

                for k in curr_gt:
                    xtl = int(k[0])
                    ytl = int(k[1])
                    xbr = int(k[2])
                    ybr = int(k[3])
                    outside = k[4]
                    occluded = k[5]
                    if outside == 1 or occluded == 1:
                        continue
                    
                    num_valid += 1.
                    
                    box_arr = np.zeros((width,height))
                    # for i in range(width):
                    #     for j in range(height):
                    #         if i>xtl and i<=xbr and j >ytl and j <=ybr:
                    #             box_arr[i][j]=1
                    for i in range(width):
                        
                        if i>xtl and i<=xbr +1:
                            box_arr[i][ytl:ybr+1]=1
                    #box_arr[xtl:xbr+1][ytl:ybr+1]=1
                    print('box_arr',box_arr,np.count_nonzero(box_arr),xtl,xbr,ytl,ybr)
                    if np.count_nonzero(box_arr)==0:
                        print('error')
                        break 
                    # result_path  = 'result_'+args.method
                    # Path(result_path).mkdir(parents=True, exist_ok=True)
                    # Path(result_path+'/save_mask').mkdir(parents=True, exist_ok=True)
                    # Path(result_path+'/gt_box').mkdir(parents=True, exist_ok=True)
                    # Path(result_path+'/max_point').mkdir(parents=True, exist_ok=True)
                    # Path(result_path+'/box_cor').mkdir(parents=True, exist_ok=True)
                    # Path(result_path+"/save_mask/"+video_id[0]).mkdir(parents=True, exist_ok=True)
                    # Path(result_path+"/gt_box/"+video_id[0]).mkdir(parents=True, exist_ok=True)
                    # Path(result_path+"/max_point/"+video_id[0]).mkdir(parents=True, exist_ok=True)
                    # Path(result_path+"/box_cor/"+video_id[0]).mkdir(parents=True, exist_ok=True)
                    # cv2.imwrite(result_path+"/save_mask/"+video_id[0]+'/'+str(frame_num)+".jpg", curr_frame*255)
                    # cv2.imwrite(result_path+"/gt_box/"+video_id[0]+'/'+str(frame_num)+".jpg", np.transpose(box_arr)*255)
                    # file2 = open(result_path+"/max_point/"+video_id[0]+'/'+str(frame_num)+".csv",'w')
                    # file2.write('x, y\n')
                    # file2.write(str(index[1])+', '+str(index[0])+'\n')
                    # file2.close()
                    # file3 = open(result_path+"/box_cor/"+video_id[0]+'/'+str(frame_num)+".csv",'w')
                    # file3.write('left, top, right, bottom\n')
                    # file3.write(str(int(k[0]))+', '+str(int(k[1]))+', '+str(int(k[2]))+', '+str(int(k[3]))+'\n')
                    # file3.close()

                    curr_frame_t = np.transpose(curr_frame)
                    # im = Image.fromarray(curr_frame_t)
                    # im.save('save_mask/'+str(frame_num)+".jpeg")
                    ##scipy.misc.imsave("save_mask/"+str(frame_num)+".jpg", curr_frame_t)
                    #import matplotlib

                    #matplotlib.image.imsave("save_mask/"+str(frame_num)+".jpg", curr_frame_t)          
                    


                    print('curr_frame_t',curr_frame_t,np.count_nonzero(curr_frame_t))
                    inter = curr_frame_t*box_arr
                    union = curr_frame_t+box_arr
                    union[union>0]=1
                    inter_s = np.count_nonzero(inter)
                    union_s = np.count_nonzero(union)
                    iou = inter_s/float(union_s)
                    print('iou',iou) 
                    if iou>0.3:
                        meanAP+=1

                    # file1.write(str(frame_num)+', '+str(iou)+', '+str(index[1])+', '+\
                    #     str(index[0])+', '+str(int(k[0]))+', '+str(int(k[1]))+', '+str(int(k[2]))\
                    #         +', '+str(int(k[3]))+'\n')
                    iou_total+=iou
                    if np.count_nonzero(box_arr)>50000:
                        iou_total_wide+=iou
                        iou_total_wide_n+=1
                    else:
                        iou_total_sat+=iou
                        iou_total_sat_n+=1

                    if index[1] >= xtl and index[1] <= xbr and index[0] >= ytl and index[0] <= ybr:
                        valid = True
                        break                   
                        
                if valid:
                    num_correct += 1.
            
            acc = num_correct / num_valid
            print('acc',acc)
            print('iou total',iou_total/num_valid)
            print('meanAP',meanAP/num_valid)
            
        #file1.close()
    acc = num_correct / num_valid
    acc *= 100.
    print('localization accuracy: ' + str(acc))
    print('meanAP',meanAP/num_valid)
    print('iou total sat',iou_total_sat/iou_total_sat_n)
    print('iou total wide',iou_total_wide/iou_total_wide_n)

if __name__ == "__main__":
    main()
