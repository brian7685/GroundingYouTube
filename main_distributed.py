import os
import random
import time
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
from pathlib import Path
from clip_feature_noModule  import  _apply_clip_text_model
from clip_feature_noModule  import _apply_clip_text_model_prompt
from clip_feature_noModule  import  _apply_clip_image_model
from my_sink import distributed_sinkhorn

from tqdm import tqdm
import s3dg
import s3dg_clip
import s3dg_clip_global
from args import get_args
from video_loader import HT100M_DataLoader
from video_loader_sentence import HT100M_DataLoader_sentence
from video_loader_clip import HT100M_DataLoader_clip
from video_loader_clip import HT100M_DataLoader_clip_prompt


from loss import MILNCELoss
from loss import MMS_loss


from loss import MILNCELoss_within
from loss import MILNCELoss_original
import sys

from metrics import compute_metrics
from youcook_interactions_loader import Youcook_DataLoader
from utils import AllGather
from utils import get_cosine_schedule_with_warmup

allgather = AllGather.apply

def nce(video_embd, text_embd):
    x = torch.matmul(text_embd, video_embd.t()) #16*512 512*16 => 16*16
    #x = get_sim_matrix(video_embd, text_embd, self.simtype, self.matchmap_pooltype)
    x = x.view(video_embd.shape[0], video_embd.shape[0], -1) #16*16*1
    #print('x',x,torch.nn.functional.normalize(x)) #logits -5.71, 18.6
    nominator = x * torch.eye(x.shape[0])[:, :, None].cuda()
    nominator = nominator.sum(dim=1)
    nominator = torch.logsumexp(nominator, dim=1)
    denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
    denominator = torch.logsumexp(denominator, dim=1)
    return torch.mean(denominator - nominator)

def main():
    args = get_args()
    assert args.eval_video_root != '' or not(args.evaluate)
    assert args.video_path != ''
    assert args.caption_root != ''
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    if args.world_size == -1 and "SLURM_NPROCS" in os.environ:
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
        hostfile = "dist_url." + jobid + ".txt"
        args.dist_url = "file://{}.{}".format(os.path.realpath(args.dist_file), jobid)
        print(
            "dist-url:{} at PROCID {} / {}".format(
                args.dist_url, args.rank, args.world_size
            )
        )
    else:
        raise NotImplementedError

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    print('dist',args.distributed)
    ngpus_per_node = torch.cuda.device_count()
    # args.world_size = ngpus_per_node
    # args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    # args.dist_url = "tcp://localhost:23458"
    # args.rank = 0

    if args.multiprocessing_distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        print('here')
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    print('gpu',gpu)
    
    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
            args.world_size = ngpus_per_node * args.world_size
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
    print('args.rank',args.rank)
    # create model
    if args.all:
        import s3dg_all
        print('S3D_all model')
        if args.globalF:
            print('global feature')
            model = s3dg_all.S3D_all(
                args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, 
                init=args.weight_init, global_feature=True
            )
        else:
            model = s3dg_all.S3D_all(
                args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, init=args.weight_init,
            )

    else:
        if args.globalF:
            if args.CLIP:
                if args.no_local:
                    print('CLIP with ONLY global feature')
                    if args.gating_feature:
                        model = s3dg_clip_global.S3D_clip(
                            args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, 
                            init=args.weight_init, global_feature=True,gating_feature=True,  resnet=args.resnet, r50=args.r50
                        )
                    else:
                        model = s3dg_clip_global.S3D_clip(
                        args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, 
                        init=args.weight_init, global_feature=True, resnet=args.resnet, r50=args.r50
                        )

                else:
                    print('CLIP with global feature')
                    if args.gating_feature:
                        print('gating features')
                        model = s3dg_clip.S3D_clip(
                            args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, 
                            init=args.weight_init, global_feature=True,gating_feature=True,  resnet=args.resnet, r50=args.r50
                        )
                    else:
                        
                        model = s3dg_clip.S3D_clip(
                            args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, 
                            init=args.weight_init, global_feature=True,gating_feature=False,  resnet=args.resnet, r50=args.r50
                        )
            else:   
                print('global feature')
                model = s3dg.S3D(
                    args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, 
                    init=args.weight_init, global_feature=True
                )

        else:
            if args.CLIP: 
                print('CLIP without global feature')
                model = s3dg_clip.S3D_clip(
                    args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, 
                    init=args.weight_init, global_feature=True, resnet=args.resnet, r50=args.r50
                )
            elif args.sc:
                print('sc model')
                import s3dg_sc
                model = s3dg_sc.S3D(
                    args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, init=args.weight_init,
                )
            elif args.cs:
                print('cs model')
                import s3dg_cs
                model = s3dg_cs.S3D(
                    args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, init=args.weight_init,
                )
            elif args.cc:
                print('cc model')
                import s3dg_cc
                model = s3dg_cc.S3D(
                    args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, init=args.weight_init,
                )
            else:
                model = s3dg.S3D(
                    args.num_class, space_to_depth=True, word2vec_path=args.word2vec_path, init=args.weight_init,
                )
    
    model_dict = model.state_dict()
    pretrained_dict = {}
    if args.gpu == 0:
        print('loading model')

    if args.pretrained_path!='':
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')#'cuda:{}'.format(args.gpu))
        state_dict = checkpoint["state_dict"]
        new_dict = {}
        for k,v in state_dict.items():
            #k = k.replace('encoder_q.encoder.', 'backbone.')
            #print('bu k',k)
            #if 'text_module.fc1' in k:
            #    print('text_module.fc1 bu val',v[0])
                
            k = k.replace('module.', '')
            k = k.replace('text_word_embd.', 'text_module.word_embd.')
            
            
            k = k.replace('text_fc1.', 'text_module.fc1.') # why commented?
            if not args.CLIP:
                k = k.replace('lang_key_mat.', 'lang_proj_mat.')
                k = k.replace('key_mat.', 'vis_proj_mat.')
            #k = k.replace('vis_attn_mat.', 'sa_layer.attn_mat.')
            if 'vis_attn_mat' in k:
                k_t = k
                k_t = k_t.replace('vis_attn_mat.', 'sa_layer.vis_attn_mat.')
                new_dict[k_t] = v
            k = k.replace('vis_attn_mat.', 'sa_layer.text_attn_mat.')
            #vis_attn_mat
            #print('bu k after',k)
            #k = k.replace('', '')
            new_dict[k] = v
        #print('new_dict',new_dict.keys())
        checkpoint = new_dict
        #print('checkpoint',checkpoint)
        for k in model.state_dict():
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
        if args.CLIP:
            model.load_state_dict(pretrained_dict, strict=False)
        else:
            model.load_state_dict(pretrained_dict, strict=True)

        if args.all:
            net_data_a = torch.load(args.pretrain_audio_path, map_location='cpu')

            model.load_state_dict(net_data_a, strict=False)


        if args.from_mil:
            checkpoint = torch.load(args.pretrain_mil_path, map_location='cpu')#'cuda:{}'.format(args.gpu))
            #state_dict = checkpoint["state_dict"]
            new_dict = {}
            for k,v in checkpoint.items():
                #k = k.replace('encoder_q.encoder.', 'backbone.')
                #print('mil',k)
                #k = k.replace('module.', '')
                if k=='fc.weight':
                    k = k.replace('fc.', 'vis_proj_mat_g.0.')
                if k=='fc.bias':
                    k = k.replace('fc.', 'vis_proj_mat_g.0.')
                #k = k.replace('fc.', 'vis_proj_mat_g.')
                k = k.replace('text_module.fc2.', 'lang_proj_mat_g.0.')
                #if 'text_module.fc1' in k:
                #    print('text_module.fc1 mil val',v[0])
                #print('mil_after',k)
                # k = k.replace('lang_key_mat.', 'lang_proj_mat.')
                # k = k.replace('key_mat.', 'vis_proj_mat.')
                # k = k.replace('vis_attn_mat.', 'sa_layer.attn_mat.')
                #k = k.replace('', '')
                #print('mil k after',k)

                new_dict[k] = v
            checkpoint = new_dict
            #print('mil checkpoint',checkpoint.keys())
            for k in model.state_dict():
                #k = k.replace('module.', '')
                # if 'DAVEnet.bn1' in k:
                #     print('DAVEnet.bn1 after val',model.state_dict()[0])
                # if 'lang_proj_mat_g' in k:
                #     print('lang_proj_mat_g before val',model.state_dict()[0])
                if k in checkpoint:
                    pretrained_dict[k] = checkpoint[k]
                    #print('k',k,checkpoint[k],model.state_dict()[k])
                    #pretrained_dict[k] = model.state_dict()[k]
                else:
                    #print('not mil',k)
                    pretrained_dict[k] = model.state_dict()[k]
            model.load_state_dict(pretrained_dict, strict=True)

            for k in model.state_dict():
                #k = k.replace('module.', '')
                if 'DAVEnet.bn1' in k:
                    print('DAVEnet.bn1 final val',v[0])

    #Freeze base encoders
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():    
        #if 'comma' not in name:
        if not args.onlyAudio:
            if 'sa_layer' in name:
                param.requires_grad = True
                #print('require grad',name)
            if 'vis_proj_mat' in name:
                param.requires_grad = True
                #print('require grad',name)
            if 'lang_proj_mat' in name:
                param.requires_grad = True
                #print('require grad',name)
        if args.all:
            if args.trainAudio:
                if 'DAVEnet' in name:
                    param.requires_grad = True
                    #print('require grad',name)
            if 'DAVEnet_fc' in name:
                param.requires_grad = True
                #print('require grad',name)
            #if 'text' not in name:
            

    if args.pretrain_cnn_path:
        net_data = torch.load(args.pretrain_cnn_path)
        model.load_state_dict(net_data)
    if args.distributed:
        if args.gpu is not None:
            print('args.gpu',args.gpu)
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
            args.num_thread_reader = int(args.num_thread_reader / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])#, find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    def set_bn_eval(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm3d') != -1:
              #print('classname',classname)
              m.eval()
    if args.fix_bn:
        print('fix_bn using')
        model.apply(set_bn_eval)

    if args.half:
        print('model half precision')
        model.half()

    # Data loading code
    if args.all:
        print('all')
        from video_loader_all_spatial import HT100M_DataLoader_all_spatial
        train_dataset = HT100M_DataLoader_all_spatial(
            csv='/nobackup/users/brian27/ECCV22/mil_nce/my_data/HowTo100M_miech_satori_audio_cook2.csv',#args.train_csv,
            video_root=args.video_path,
            caption_root=args.caption_root,
            min_time=args.min_time,
            fps=args.fps,
            num_frames=args.num_frames,
            size=args.video_size,
            crop_only=args.crop_only,
            center_crop=args.centercrop,
            random_left_right_flip=args.random_flip,
            num_candidates=args.num_candidates,
            n_pair=args.n_pair,
            num_audio_frames=args.howto_audio_frames,
            )
    else:
        if args.ht370:
            print('ht370K')
            # add sentence code here
            if args.sentence:
                print('sentence ht370')
                train_dataset = HT100M_DataLoader(
                csv='/nobackup/users/brian27/ECCV22/mil_nce/my_data/HowTo370K.csv',#args.train_csv,
                video_root=args.video_path,
                caption_root=args.caption_root,
                min_time=args.min_time,
                fps=args.fps,
                num_frames=args.num_frames,
                size=args.video_size,
                crop_only=args.crop_only,
                center_crop=args.centercrop,
                random_left_right_flip=args.random_flip,
                num_candidates=args.num_candidates,
                )
            elif args.CLIP:
                train_dataset = HT100M_DataLoader_clip(
                csv='/nobackup/users/brian27/ECCV22/mil_nce/my_data/HowTo370K.csv',#args.train_csv,
                video_root=args.video_path,
                caption_root=args.caption_root,
                min_time=args.min_time,
                fps=args.fps,
                num_frames=args.num_frames,
                size=args.video_size,
                crop_only=args.crop_only,
                center_crop=args.centercrop,
                random_left_right_flip=args.random_flip,
                num_candidates=args.num_candidates,
                )
            else:
                train_dataset = HT100M_DataLoader(
                csv='/nobackup/users/brian27/ECCV22/mil_nce/my_data/HowTo370K.csv',#args.train_csv,
                video_root=args.video_path,
                caption_root=args.caption_root,
                min_time=args.min_time,
                fps=args.fps,
                num_frames=args.num_frames,
                size=args.video_size,
                crop_only=args.crop_only,
                center_crop=args.centercrop,
                random_left_right_flip=args.random_flip,
                num_candidates=args.num_candidates,
                )
        else:
            if args.CLIP:
                print('clip data loader')
                if args.prompt:
                    print('clip prompt')
                    train_dataset = HT100M_DataLoader_clip(
                        csv='/nobackup/users/brian27/ECCV22/mil_nce/my_data/HowTo100M_miech_satori_audio_cook2.csv',#args.train_csv,
                        video_root=args.video_path,
                        caption_root=args.caption_root,
                        min_time=args.min_time,
                        fps=args.fps,
                        num_frames=args.num_frames,
                        size=args.video_size,
                        crop_only=args.crop_only,
                        center_crop=args.centercrop,
                        random_left_right_flip=args.random_flip,
                        num_candidates=args.num_candidates,
                        num_sec_control=args.num_sec_control,
                        fix_start=args.fix_start,
                        early_start=args.early_start
                    )
                elif args.ht_all:
                    print('howTo100M all')
                    train_dataset = HT100M_DataLoader_clip(
                        csv='/nobackup/users/brian27/ECCV22/mil_nce/data/HowTo100M_miech_satori_audio.csv',#args.train_csv,
                        video_root=args.video_path,
                        caption_root=args.caption_root,
                        min_time=args.min_time,
                        fps=args.fps,
                        num_frames=args.num_frames,
                        size=args.video_size,
                        crop_only=args.crop_only,
                        center_crop=args.centercrop,
                        random_left_right_flip=args.random_flip,
                        num_candidates=args.num_candidates,
                        num_sec_control=args.num_sec_control,
                        fix_start=args.fix_start,
                        early_start=args.early_start
                    )
                else:
                    train_dataset = HT100M_DataLoader_clip(
                        csv='/nobackup/users/brian27/ECCV22/mil_nce/my_data/HowTo100M_miech_satori_audio_cook2.csv',#args.train_csv,
                        video_root=args.video_path,
                        caption_root=args.caption_root,
                        min_time=args.min_time,
                        fps=args.fps,
                        num_frames=args.num_frames,
                        size=args.video_size,
                        crop_only=args.crop_only,
                        center_crop=args.centercrop,
                        random_left_right_flip=args.random_flip,
                        num_candidates=args.num_candidates,
                        num_sec_control=args.num_sec_control,
                        fix_start=args.fix_start,
                        early_start=args.early_start
                    )

            else:
                
                train_dataset = HT100M_DataLoader(
                    csv='/nobackup/users/brian27/ECCV22/mil_nce/my_data/HowTo100M_miech_satori_audio_cook2.csv',#args.train_csv,
                    video_root=args.video_path,
                    caption_root=args.caption_root,
                    min_time=args.min_time,
                    fps=args.fps,
                    num_frames=args.num_frames,
                    size=args.video_size,
                    crop_only=args.crop_only,
                    center_crop=args.centercrop,
                    random_left_right_flip=args.random_flip,
                    num_candidates=args.num_candidates,
                )
        


    # Test data loading code
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.distributed: 
        print('distributed 1 worker')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=1,#args.num_thread_reader,#1, #16
            pin_memory=args.pin_memory,
            sampler=train_sampler,
        )
    else:
        print('Not distributed, multi worker')
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            drop_last=True,
            num_workers=args.num_thread_reader,
            pin_memory=args.pin_memory,
            sampler=train_sampler,
        )

    # define loss function (criterion) and optimizer
    if args.MMS_loss:
        print('MMS loss')
        criterion = MMS_loss()
    else:
        criterion = MILNCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), args.lr)

    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, len(train_loader) * args.epochs)
    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoint', args.checkpoint_dir)
    if args.checkpoint_dir != '' and not(os.path.isdir(checkpoint_dir)) and args.rank == 0:
        #os.mkdir(checkpoint_dir)
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)


    if args.resume:
        print('resume from checkpoint',checkpoint_dir)
        checkpoint_path = get_last_checkpoint(checkpoint_dir)
        if checkpoint_path:
            log("=> loading checkpoint '{}'".format(checkpoint_path), args)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            
            log("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint["epoch"]), args)
        else:
            log("=> no checkpoint found at '{}'".format(args.resume), args)
    
    if args.cudnn_benchmark:
        cudnn.benchmark = True
    total_batch_size = args.world_size * args.batch_size 
    log(
        "Starting training loop for rank: {}, total batch size: {}".format(
            args.rank, total_batch_size
        ), args
    )
    scaler = 0#torch.cuda.amp.GradScaler(True)
    for epoch in range(args.start_epoch, args.epochs):
        print('checkpoint dir',checkpoint_dir)
        if epoch==0:
            if args.rank == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }, checkpoint_dir, epoch
                )
        if args.distributed:
            train_sampler.set_epoch(epoch)
        # train for one epoch
        if args.CLIP:
            import clip
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
            #CLIP_model = torch.nn.DataParallel(CLIP_model).cuda()
            train(train_loader, model, criterion, optimizer, scheduler, epoch, train_dataset, args, scaler, CLIP_model) # here add CLIP
        else:
            CLIP_model=0
            train(train_loader, model, criterion, optimizer, scheduler, epoch, train_dataset, args, scaler, CLIP_model)
        if args.rank == 0:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, checkpoint_dir, epoch + 1
            )

def train(train_loader, model, criterion, optimizer, scheduler, epoch, dataset, args, scaler,CLIP_model):
    running_loss = 0.0
    s = time.time()
    start = time.time()
    for i_batch, sample_batch in enumerate(tqdm(train_loader)):
        s_step = time.time()
        batch_loss = TrainOneBatch(model, optimizer, scheduler, sample_batch, criterion, args, scaler,CLIP_model)
        d_step = time.time() - s_step
        running_loss += batch_loss
        if (i_batch + 1) % args.n_display == 0 and args.verbose and args.rank == 0:
            d = time.time() - s
            log(
                "Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Training loss: %.4f, Learning rate: %.6f"
                % (
                    epoch + 1,
                    d,
                    args.batch_size * args.world_size * float(i_batch) / len(dataset),
                    running_loss / args.n_display,
                    optimizer.param_groups[0]['lr'],
                ), args
            )
            print("Epoch %d, Elapsed Time: %.3f, Epoch status: %.4f, Training loss: %.4f, Learning rate: %.6f" % (
                    epoch + 1,
                    d,
                    args.batch_size * args.world_size * float(i_batch) / len(dataset),
                    running_loss / args.n_display,
                    optimizer.param_groups[0]['lr'],
                ))
            running_loss = 0.0
            s = time.time()
        start_load = time.time()
    end = time.time()
    total = end - start
    if args.rank == 0:
        print('time elapsed: ' + str(total))

def TrainOneBatch(model, opt, scheduler, data, loss_fun, args, scaler, CLIP_model):
    
    video = data["video"].float().cuda(args.gpu, non_blocking=args.pin_memory)
    video = video / 255.0
    # Clip should go here
    if args.CLIP:
        import clip
        #print('CLIP model')
        #image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
        batch_size = video.shape[0]
        image = video.permute(0, 2, 1,3, 4).half() 
        image = torch.flatten(image, start_dim=0, end_dim=1)
        #print('image shape',image.shape)
        #image_features =  CLIP_model.visual.forward(image)
        #image_features = image_features.view(batch_size,16,512)

        #print('image_features shape',image_features.shape)
        #print('data["text"]',data["raw_text"])
        if args.prompt:
            
            text, mask, global_t = _apply_clip_text_model_prompt(CLIP_model, data, args.gpu, args.pin_memory)
            #mask = data["mask"].cuda(args.gpu, non_blocking=args.pin_memory)
            #print('old mask',mask[0])
        else:
            text, mask, global_t = _apply_clip_text_model(CLIP_model, data, args.gpu, args.pin_memory)
        #print('txt',text.shape)
        if args.half:
            text = text.half()
        #print('text feature',text.shape)  #[32, 20, 512]
        #print('text mask',mask.shape) #[32, 20]
        #print('~mask',mask) 
        
        global_v,local_v = _apply_clip_image_model(CLIP_model,
            image, args.gpu, args.pin_memory, args.resnet)
        if args.r50:
            global_v = global_v.view(batch_size,-1,1024)
        else:
            global_v = global_v.view(batch_size,-1,512) #[32, 16, 512]

        if args.hitchhiker_local:
            if args.hitchhiker_norm:
                global_v_n = global_v / global_v.norm(dim=-1, keepdim=True)
                global_t_n = global_t / global_t.norm(dim=-1, keepdim=True)
                global_sim = torch.matmul(global_v_n, global_t_n.unsqueeze(2)).squeeze(2)/0.1 #32*16

            else:
                global_sim = torch.matmul(global_v, global_t.unsqueeze(2)).squeeze(2)/0.1 #32*16
            
            #global_sim = global_v@global_t.t()/0.1 #32*16*32
            m = nn.Softmax(dim=1)
            global_sim_soft = m(global_sim).unsqueeze(2).unsqueeze(3)
            #print('global_sim',global_sim_soft,global_sim_soft.shape)
        #mean pool, various pooling method should go here
        #print('global_v',global_v.shape)
        
        if args.resnet:
            local_v = local_v.view(batch_size,-1,49,2048)
        else:
            local_v = local_v.view(batch_size,-1,49,512) #[32, 16, 49, 512]
            if args.one_frame:
                frame_num = local_v.shape[1]
                mid_f = int(frame_num/2)
                local_v = local_v[:,mid_f,:,:]
            
            if args.hitchhiker_local:
                local_v = local_v*global_sim_soft
                if args.hitchhiker_local_pool:  
                    local_v= local_v.mean(1, keepdim=True)

            if args.longer_frame:
                m = nn.Softmax(dim=1)
                temp_length = local_v.shape[1]
                if args.global_select:
                    if args.project:
                        with torch.no_grad():
                            video_embd, text_embd,video_embd_g, text_embd_g = \
                                model(global_v,local_v, global_t, text, mask, mode='global_text')
                            global_v_n = global_v / global_v.norm(dim=-1, keepdim=True)
                            global_t_n = global_t / global_t.norm(dim=-1, keepdim=True)
                            global_sim = torch.matmul(global_v_n, global_t_n.unsqueeze(2)).squeeze(2)/0.1
                            
                            global_sim_soft = m(global_sim)#.unsqueeze(2).unsqueeze(3)

                    else:
                        global_v_n = global_v / global_v.norm(dim=-1, keepdim=True)
                        global_t_n = global_t / global_t.norm(dim=-1, keepdim=True)
                        global_sim = torch.matmul(global_v_n, global_t_n.unsqueeze(2)).squeeze(2)/0.1
                        
                        global_sim_soft = m(global_sim)#.unsqueeze(2).unsqueeze(3)
                    num_prune=8
                    frame_rank = torch.argsort(global_sim_soft, dim=1, descending=True)[:,:num_prune]#.squeeze(2).squeeze(2)
                    #print('frame_rank',frame_rank.shape)
                    #print('local_v before',local_v.shape)
                    batch_size = local_v.shape[0]
                    tmp_local = torch.empty((batch_size,8,49,512), dtype=torch.float32)
                    for i in range(batch_size):
                        tmp_local[i] = torch.index_select(local_v[i], 0, frame_rank[i])
                    local_v = tmp_local
                    #print('local_v',local_v.shape)
                elif args.local_select:
                    if args.project:
                        with torch.no_grad():
                            video_embd, text_embd,video_embd_g, text_embd_g = \
                                model(global_v,local_v, text, text, mask, mode='global_text') 
                            global_v_n = video_embd_g / video_embd_g.norm(dim=-1, keepdim=True) #32*16*512
                            #print('global_v',global_v.shape)
                            local_t_n = text_embd_g / text_embd_g.norm(dim=-1, keepdim=True) #32*20*512
                            local_t_n = local_t_n.permute(0,2,1) ##32*512*20
                            global_sim = torch.matmul(global_v_n, local_t_n).mean(-1)/0.1 #32*16*20
                            
                            global_sim_soft = m(global_sim)#.unsqueeze(2).unsqueeze(3)
                    elif args.project_old:
                        #print('old project')
                        with torch.no_grad():
                            video_embd, text_embd,video_embd_g, text_embd_g = \
                                model(global_v,local_v, global_t, text, mask, mode='global_text') 
                            global_v_n = video_embd_g / video_embd_g.norm(dim=-1, keepdim=True) #32*16*512
                            #print('global_v',global_v.shape)
                            local_t_n = text_embd / text_embd.norm(dim=-1, keepdim=True) #32*20*512
                            local_t_n = local_t_n.permute(0,2,1) ##32*512*20
                            global_sim = torch.matmul(global_v_n, local_t_n).mean(-1)/0.1 #32*16*20
                            
                            global_sim_soft = m(global_sim)#.unsqueeze(2).unsqueeze(3)
                    else:

                        global_v_n = global_v / global_v.norm(dim=-1, keepdim=True) #32*16*512
                        #print('global_v',global_v.shape)
                        local_t_n = text / text.norm(dim=-1, keepdim=True) #32*20*512
                        local_t_n = local_t_n.permute(0,2,1) ##32*512*20
                        global_sim = torch.matmul(global_v_n, local_t_n).mean(-1)/0.1 #32*16*20
                        
                        global_sim_soft = m(global_sim)#.unsqueeze(2).unsqueeze(3)
                    num_prune=8
                    # Todo
                    if args.self_select:
                        #global_t #32*512
                        text_temp = text.permute(0,2,1)
                        global_t_temp = global_t.unsqueeze(1)
                        #print('global_t',global_t_temp.shape,text_temp.shape)
                        text_sim = torch.matmul(global_t_temp, text_temp).mean(-2) #64*20
                        #print('text_sim',text_sim.shape)
                        text_rank = torch.argsort(text_sim, dim=1, descending=True)#[:,:num_prune]
                        #*text #32*20*512
                        txt_tmp = torch.empty((batch_size,20,512), dtype=torch.float32).cuda(args.gpu, non_blocking=args.pin_memory)
                        local_t_n_tmp = local_t_n.permute(0,2,1)
                        #print('local_t_n',local_t_n_tmp.shape)
                        print('text_rank',text_rank,text_rank.shape)
                        for i in range(batch_size):
                            txt_tmp[i] = torch.index_select(local_t_n_tmp[i], 0, text_rank[i])
                        #local_t_n_tmp = txt_tmp
                        #txt_tmp = txt_tmp[:num_prune]
                        local_t_n = txt_tmp.permute(0,2,1)

                    if args.sink:
                        frame_rank = torch.empty((batch_size,8), dtype=torch.long).cuda(args.gpu, non_blocking=args.pin_memory)
                        for i in range(batch_size):
                            mm_sim = torch.matmul(global_v_n[i], local_t_n[i])/0.1

                            Q, argmaxes = distributed_sinkhorn(mm_sim)
                            #print('argmaxes',argmaxes,argmaxes.shape)
                            selected  =argmaxes[:num_prune]
                            
                            frame_rank[i] = selected#torch.sort(selected)
                            #print('selected',selected)

                    else:
                        frame_rank = torch.argsort(global_sim_soft, dim=1, descending=True)[:,:num_prune]
                    # dummy = frame_rank.unsqueeze(2).unsqueeze(3).expand(frame_rank.size(0), \
                    # frame_rank.size(1), local_v.size(2), local_v.size(3))
                    # batch_size = local_v.shape[0]
                    # local_v = torch.gather(local_v, 1, dummy)
                    tmp_local = torch.empty((batch_size,8,49,512), dtype=torch.float32)
                    for i in range(batch_size):
                        tmp_local[i] = torch.index_select(local_v[i], 0, frame_rank[i])
                    local_v = tmp_local
                    
                else:
                    speed = args.speed
                    indices = torch.arange(0, temp_length, speed).long().cuda(args.gpu, non_blocking=args.pin_memory)
                    #print('indices',indices)
                    local_v = torch.index_select(local_v, 1, indices)
                    #print('local_v',local_v.shape)
                    #indices = torch.tensor([0, 2])

            if args.local_mean:
                local_v= local_v.mean(1, keepdim=True)

            #else:
        if args.max_pool:
            #print('max_pool')
            #global_v = global_v.permute(0,2,1)
            global_v= F.normalize(torch.max(global_v, 1)[0])#global_v.mean(-2)
            #print('global_v',global_v.shape)
        else:
            global_v= global_v.mean(-2)
        
        #print('global_v',global_v.shape) #[512, 512]
        #print('local_v',local_v.shape)

        #text_clip = clip.tokenize(data["raw_text"]).cuda(args.gpu, non_blocking=args.pin_memory)
        #print('text_clip',text_clip)
        #text_features = CLIP_model.encode_text(text_clip)
        #print('text_features',text_features,text_features.shape) #3,512 [32, 512]


    else: # S3D word2vec
        text = data["text"].cuda(args.gpu, non_blocking=args.pin_memory)
        idx = data["idx"].cuda(args.gpu, non_blocking=args.pin_memory)
        mask = data["mask"].cuda(args.gpu, non_blocking=args.pin_memory)
        text = text.view(-1, text.shape[-1])

    if args.all:
        audio = data['audio'].float().cuda(args.gpu, non_blocking=args.pin_memory) # NOTE: named it text for now
        audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    
    #print('video shape',video.shape) #video shape torch.Size([24, 3, 16, 224, 224])
    #print('text shape',text.shape) #[32, 20]
    
    

    

    opt.zero_grad()
    #with torch.set_grad_enabled(False):
    with torch.set_grad_enabled(True):
        #print('video',video,'text',text)
        if args.all:
            if args.globalF:
                if args.global_within:
                    video_embd, audio_embd, text_embd, video_embd_g, audio_embd_g, text_embd_g, video_embd_region  \
                        = model(video, audio, text, mask, mode='global_within') 
                    #print('video_embd_region',video_embd_region.shape) #[96, 24, 98, 512]
                else:
                    #print('not global within')
                    if args.l2_norm:

                        video_embd, audio_embd, text_embd, video_embd_g, audio_embd_g, text_embd_g \
                            = model(video, audio, text, mask, mode='global') 

                    else:

                        video_embd, audio_embd, text_embd, video_embd_g, audio_embd_g, text_embd_g \
                            = model(video, audio, text, mask, mode='global') 
            else:

                video_embd, audio_embd, text_embd = model(video, audio, text, mask) 
        else:
            if args.globalF:

                if args.global_within:
                    if args.l2_norm:
                        #global_l2
                        video_embd, text_embd,video_embd_g, text_embd_g, video_embd_region, video_att, text_att \
                        = model(video, text, mask, mode='global_within_l2')
                    elif args.l2_norm_all:
                        #global_l2
                        video_embd, text_embd,video_embd_g, text_embd_g, \
                        v_cross_att_1, t_cross_att_1, v_self_att_1, t_self_att_1 , v_cross_att_2, t_cross_att_2,video_embd_region  \
                         = model(video, text, mask, mode='global_within_l2_all')
                    else:
                        if args.CLIP:
                            
                            video_embd, text_embd,video_embd_g, text_embd_g, video_embd_region = \
                            model(global_v,local_v, text, mask, mode='global_within') 
                        else:
                            video_embd, text_embd, video_embd_g, text_embd_g, video_embd_region  \
                            = model(video, text, mask, mode='global_within') 
                else:


                    if args.CLIP:
                        if args.l2_norm_all:
                            #global_l2
                            video_embd, text_embd,video_embd_g, text_embd_g, \
                            v_cross_att_1, t_cross_att_1, v_self_att_1, t_self_att_1 , v_cross_att_2, t_cross_att_2\
                            = model(global_v, local_v, text, mask, mode='global_l2_all')
                        elif args.no_local:
                            #print('no_local')
                            video_embd_g, text_embd_g = \
                            model(global_v, global_t, mode='global_only') 
                        elif args.global_text:
                            #print('global text')
                            video_embd, text_embd,video_embd_g, text_embd_g = \
                            model(global_v,local_v, global_t, text, mask, mode='global_text') 

                        
                        
                        else:
                            video_embd, text_embd,video_embd_g, text_embd_g = \
                            model(global_v,local_v,global_t, text, mask, mode='global') 
                    else:
                        if args.l2_norm:
                            #global_l2
                            video_embd, text_embd,video_embd_g, text_embd_g, video_att, text_att = model(video, text, mask, mode='global_l2')
                        elif args.l2_norm_all:
                            #global_l2
                            video_embd, text_embd,video_embd_g, text_embd_g, \
                            v_cross_att_1, t_cross_att_1, v_self_att_1, t_self_att_1 , v_cross_att_2, t_cross_att_2\
                            = model(video, text, mask, mode='global_l2_all')
                        else:
                            video_embd, text_embd,video_embd_g, text_embd_g = model(video, text, mask, mode='global') 
            else:
                if args.CLIP:
                    #print('no glob')
                    video_embd, text_embd,video_embd_g, text_embd_g = \
                            model(global_v,local_v, global_t, text, mask, mode='no_global') 
                else:
                    video_embd, text_embd = model(video, text, mask) 
        
         
        if args.gpu==None:
            if not args.no_local:
                video_embd = video_embd.view(4,video_embd.shape[1],video_embd.shape[1],512)
                text_embd = text_embd.view(4,text_embd.shape[1],text_embd.shape[1],512)
            if args.all:
                #print('video_embd',video_embd.shape,'text_embd',text_embd.shape)   
                #print('audio_embd',audio_embd.shape)
                audio_embd = audio_embd.view(4,audio_embd.shape[1],audio_embd.shape[1],512)
        #print('video_embd',video_embd.shape,'text_embd',text_embd.shape)   
        if args.gpu==None: # no distributed
            if args.all:
                loss=0
                if not args.no_local:
                    for i in range(4):
                        loss += loss_fun(video_embd[i], text_embd[i]) + \
                        args.VALoss * loss_fun(video_embd[i], audio_embd[i]) \
                            + args.ATLoss * loss_fun(audio_embd[i], text_embd[i])

                if args.globalF:
                    #cosine similarity
                    # video_embd_g = F.normalize(video_embd_g)
                    # text_embd_g = F.normalize(text_embd_g)
                    # x = torch.matmul(text_embd_g, video_embd_g.t())
                    # #x = get_sim_matrix(video_embd, text_embd, self.simtype, self.matchmap_pooltype)
                    # x = x.view(video_embd_g.shape[0], video_embd_g.shape[0], -1)
                    #print('x',x,torch.nn.functional.normalize(x))
                    loss += nce(video_embd_g,audio_embd_g)+nce(video_embd_g,text_embd_g)+nce(text_embd_g,audio_embd_g)
                    if args.global_within:
                        #loss+=0 ##TODO
                        video_embd_region = video_embd_region.view(4,video_embd.shape[1],video_embd.shape[1],2,49,512)
                        #video_embd_region torch.Size([4, 24, 24, 2, 49, 512])
                        video_embd_region = video_embd_region.mean(-2) #4, 24, 24, 2,  512
                        #print('video_embd_region',video_embd_region.shape)#
                        video_embd_region = video_embd_region.permute(0, 3,1,2, 4)
                        #print('i',i)
                        for i in range(4):
                            loss += loss_fun(video_embd_region[i][0], video_embd_region[i][1]) 
            else:
                loss=0
                #print('video_embd',video_embd.shape) #video_embd torch.Size([4, 16, 16, 512])
                #nominator = video_embd[0] * torch.eye(video_embd[0].shape[0])[:, :,].cuda()
                #video_embd[0]
                if  args.multi_loss:
                    vid_list = []
                    for i in range(4):
                        vid_list.append(torch.transpose(torch.diagonal(video_embd[i],0), 0, 1)) #16
                    v_local = torch.stack(vid_list)
                    v_local = v_local.view(video_embd.shape[0]*video_embd.shape[1],512)
                    #print('v_local',v_local.shape)
                    text_list = []
                    for i in range(4):
                        text_list.append(torch.transpose(torch.diagonal(text_embd[i],0), 0, 1)) #16
                    t_local = torch.stack(text_list)
                    t_local = t_local.view(text_embd.shape[0]*text_embd.shape[1],512)
                    #print('t_local',t_local.shape)
                    loss += nce(video_embd_g,v_local)
                    loss += nce(t_local,text_embd_g)
                    if  args.cross_consist:
                        loss += nce(video_embd_g,t_local)
                        loss += nce(v_local,text_embd_g)
                    #print('diagonal',torch.diagonal(video_embd[0],dim1=0, dim2=1).shape)
                if not args.no_local:
                    loss += loss_fun(video_embd[0], text_embd[0])+loss_fun(video_embd[1], text_embd[1])+ \
                    loss_fun(video_embd[2], text_embd[2])+loss_fun(video_embd[3], text_embd[3])
                if args.globalF:
                    #print('video_embd_g',video_embd_g.shape)
                    loss += nce(video_embd_g,text_embd_g)
                    
                    

                    if args.global_within:
                        #loss+=0 ##TODO
                        if args.CLIP:
                            video_embd_region = video_embd_region.view(4,video_embd.shape[1],video_embd.shape[1],8,49,512)
                            video_embd_region = video_embd_region.mean(-2)
                            if args.video_mil:
                                video_embd_region = video_embd_region.view(4,-1,8,512)
                                video_embd_region = video_embd_region.permute(0, 2,1,3)
                                for i in range(4):
                                    text_embd_temp = text_embd.view(4,-1,512)
                                    video_mil_loss = nce(text_embd_temp[i], video_embd_region[i]) 
                                    #print('video_mil_loss',video_mil_loss)
                                    loss += video_mil_loss#MILNCELoss_original(text_embd_temp[i], video_embd_region[i]) 

                            if args.video_mil_within:
                                video_embd_region = video_embd_region.view(4,-1,8,512)
                                video_embd_region = video_embd_region.permute(0, 2,1,3)
                                for i in range(4):
                                    #text_embd_temp = text_embd.view(4,-1,512)
                                    video_mil_loss = nce(video_embd_region[i][0], video_embd_region[i][1:]) 
                                    #print('video_mil_within_loss',video_mil_loss)
                                    loss += video_mil_loss#MILNCELoss_original(text_embd_temp[i], video_embd_region[i])
                            else:
                                video_embd_region = video_embd_region.permute(0, 3,1,2, 4)
                                for i in range(4):
                                    loss += loss_fun(video_embd_region[i][:4].mean(0), video_embd_region[i][4:].mean(0)) 
                        else:
                            video_embd_region = video_embd_region.view(4,video_embd.shape[1],video_embd.shape[1],2,49,512)
                            #video_embd_region torch.Size([4, 24, 24, 2, 49, 512])
                            video_embd_region = video_embd_region.mean(-2) #4, 24, 24, 2,  512
                            #print('video_embd_region',video_embd_region.shape)#
                            video_embd_region = video_embd_region.permute(0, 3,1,2, 4)
                            for i in range(4):
                                loss += loss_fun(video_embd_region[i][0], video_embd_region[i][1]) 

                        if args.l2_norm:
                            att_norm_loss = -torch.log(torch.norm(video_att)) - torch.log(torch.norm(text_att))
                            #print('att_norm_loss',att_norm_loss)
                            loss+= att_norm_loss*args.sparsity # was actually 20 instead of 10

                        elif args.l2_norm_all: 
                            #v_cross_att_1, t_cross_att_1, v_self_att_1, t_self_att_1 , v_cross_att_2, t_cross_att_2
                            v_cross_att_1_norm = torch.norm(v_cross_att_1,2,-1)
                            t_cross_att_1_norm = torch.norm(t_cross_att_1,2,-1)
                            v_cross_att_2_norm = torch.norm(v_cross_att_2,2,-1)
                            t_cross_att_2_norm = torch.norm(t_cross_att_2,2,-1)
                            att_norm_loss = -torch.log(torch.mean(v_cross_att_1_norm)) - torch.log(torch.mean(t_cross_att_1_norm)) \
                            #-torch.log(torch.norm(t_self_att_1)) - torch.log(torch.norm(t_self_att_1)) \
                            -torch.log(torch.mean(v_cross_att_2_norm)) - torch.log(torch.mean(t_cross_att_2_norm)) 
                            #print('att_norm_loss',att_norm_loss)
                            loss+= att_norm_loss*args.sparsity
                    else:
                        if args.l2_norm:
                            video_att_norm = torch.norm(video_att,2,-1)
                            text_att_norm = torch.norm(text_att,2,-1)
                            att_norm_loss = -torch.log(torch.mean(video_att_norm)) - torch.log(torch.mean(text_att_norm))
                            #print('att_norm_loss',att_norm_loss)
                            loss+= att_norm_loss*args.sparsity
                        
                        elif args.l2_norm_all:
                            #print('v_cross_att_1',v_cross_att_1)
                            #print('v_cross_att_1_norm',torch.norm(v_cross_att_1))
                            #print('v_cross_att_1_norm_log',torch.log(torch.norm(v_cross_att_1)))
                            #v_cross_att_1, t_cross_att_1, v_self_att_1, t_self_att_1 , v_cross_att_2, t_cross_att_2
                            v_cross_att_1_norm = torch.norm(v_cross_att_1,2,-1)
                            t_cross_att_1_norm = torch.norm(t_cross_att_1,2,-1)
                            v_cross_att_2_norm = torch.norm(v_cross_att_2,2,-1)
                            t_cross_att_2_norm = torch.norm(t_cross_att_2,2,-1)
                            att_norm_loss = -torch.log(torch.mean(v_cross_att_1_norm)) - torch.log(torch.mean(t_cross_att_1_norm)) \
                            #-torch.log(torch.norm(v_self_att_1)) - torch.log(torch.norm(t_self_att_1)) \
                            -torch.log(torch.mean(v_cross_att_2_norm)) - torch.log(torch.mean(t_cross_att_2_norm)) 
                            
                            #print('att_norm_loss',att_norm_loss)
                            

                            loss+= att_norm_loss*args.sparsity

                            if args.l2_norm_self:
                                v_self_att_1_norm = torch.norm(v_self_att_1,2,-1)
                                t_self_att_1_norm = torch.norm(t_self_att_1,2,-1)
                                att_norm_loss = -torch.log(torch.mean(v_self_att_1_norm)) - \
                                torch.log(torch.mean(t_self_att_1_norm))
                                loss+= att_norm_loss*args.sparsity

        else: #is distributed
            #video_embd = allgather(video_embd, args)
            #text_embd = allgather(text_embd, args)
            if not args.no_local:
                loss = loss_fun(video_embd, text_embd)
            if args.globalF:
                video_embd_g = allgather(video_embd_g, args)
                text_embd_g = allgather(text_embd_g, args)
                #print('text_embd_g',text_embd_g.shape)   
                #log("text_embd_g '{}'".format(text_embd_g.shape), args)
                if args.MMS_loss:
                    #log("MMS loss '{}'".format(text_embd_g.shape), args)
                    loss = loss_fun(video_embd_g, text_embd_g)
                else:
                    loss = nce(video_embd_g,text_embd_g)

            if args.global_within:
                #loss+=0 ##TODO
                #print('video_embd_region',video_embd_region.shape) #32, 32, 98, 512
                video_embd_region = video_embd_region.view(video_embd.shape[0],video_embd.shape[0],2,49,512)
                #video_embd_region torch.Size([4, 24, 24, 2, 49, 512])
                video_embd_region = video_embd_region.mean(-2) #4, 24, 24, 2,  512
                #print('video_embd_region',video_embd_region.shape)#
                video_embd_region = video_embd_region.permute(2, 0,1, 3) 
                #for i in range(4):
                loss += loss_fun(video_embd_region[0], video_embd_region[1]) 
        if torch.isnan(loss):
            print('vid',data["vid"])   
            file1 = open('nan_files/'+data["vid"][0]+'.txt','w')
            file1.close()
    #print('loss',loss.item())
    loss.backward()
    opt.step()
    scheduler.step()
    
    return loss.item()

def evaluate(test_loader, model, epoch, args, dataset_name):
    all_txt_embd = []
    all_video_embd = []
    model.eval()
    if args.rank == 0:  
        log('Evaluating on {}'.format(dataset_name), args)
    with torch.no_grad():
        for i_batch, data in enumerate(test_loader):
            text = data['text'].cuda()
            video = data['video'].float().cuda()
            video = video / 255.0
            video = video.view(-1, video.shape[2], video.shape[3], video.shape[4], video.shape[5])
            video_embd, text_embd = model(video, text)
            video_embd = video_embd.view(text_embd.shape[0], args.num_windows_test, text_embd.shape[1])
            video_embd = video_embd.mean(dim=1)
            video_embd = allgather(video_embd, args)
            text_embd = allgather(text_embd, args)
            if args.rank == 0:
                text_embd = text_embd.cpu().numpy()
                video_embd = video_embd.cpu().numpy()
                all_txt_embd.append(text_embd)
                all_video_embd.append(video_embd)
    model.train()
    if args.rank == 0:
        all_txt_embd = np.concatenate(all_txt_embd, axis=0)
        all_video_embd = np.concatenate(all_video_embd, axis=0)
        metrics = compute_metrics(np.dot(all_txt_embd, all_video_embd.T))
        log('Epoch {} results: {}'.format(epoch, metrics), args)

def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=20):
    torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch)))
    # Don't remove previous
    # if epoch - n_ckpt >= 0: 
    #     oldest_ckpt = os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch - n_ckpt)) 
    #     if os.path.isfile(oldest_ckpt):
    #         os.remove(oldest_ckpt)
            
def save_store_checkpoint(state, checkpoint_dir, epoch, n_ckpt=10):
    torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch)))

def get_last_checkpoint(checkpoint_dir):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, 'epoch*.pth.tar'))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        print('load this check',all_ckpt[-1])
        return all_ckpt[-1]
    else:
        return ''

def log(output, args):
    with open(os.path.join(os.path.dirname(__file__), 'log' , args.checkpoint_dir + '.txt'), "a") as f:
        f.write(output + '\n')

if __name__ == "__main__":
    main()
