# What, when, and where? -- Self-Supervised Spatio-Temporal Grounding in Untrimmed Multi-Action Videos from Narrated Instructions
******************************************************

This repo has the implementation of our paper: [What, when, and where? -- Self-Supervised Spatio-Temporal Grounding in Untrimmed Multi-Action Videos from Narrated Instructions](https://arxiv.org/abs/2303.16990)

![figure](figure.png)


## Getting Started


```
$ pip install requirements.txt
$ conda create -n stg --file req.txt
```

**************************************************************

## Datasets

### Youcook inter

### Mining YouTube

### Grounding YouTube

**************************************************************

## Train Model

```
$ python -u main_distributed.py  --batch_size=64  \
--lr=1e-4 --epochs=15 --globalF  \
--CLIP --fps 1 --num_frames 8 --num_sec_control 8 --longer_frame --local_select \
--sink --checkpoint_dir=nce_b64_globalF_CLIP_1fps_8frame_num_sec_control8_local_select_sink
```
**************************************************************


## Test Model

```
$ CUDA_VISIBLE_DEVICE=1 python -W ignore eval_mining_clip_iou.py \
--eval_video_root $video_path \
--youcook2_annotations_path mining_anno/seg.json \
--interactions_annotations_path mining_anno/id2xy_box.json \
--checkpoint_eval \
checkpoint/nce_b64_globalF_CLIP_1fps_8frame_num_sec_control8_local_select_sink/epoch0009.pth.tar

```

Evaluate youcook-inter with finetuned clip
```
CUDA_VISIBLE_DEVICE=1 python -W ignore eval_youcook_clip_finetune.py \
--eval_video_root /nobackup/users/brian27/ECCV22/mil_nce/my_data/youcook/validation_all/ \
--youcook2_annotations_path youcookii_annotations_trainval.json \
--interactions_annotations_path YouCook2-Interactions/final_dataset_annotations.pkl \
--interactions_segments_path YouCook2-Interactions/final_dataset_segments.pkl  \
--pretrain_clip \
/nobackup/users/brian27/ECCV22/video-grounding-narrations/checkpoint/nce_b96_globalF_CLIP_1fps_8frame_finetune_0912/epoch0033.pth.tar \
--checkpoint_eval \
/nobackup/users/brian27/ECCV22/video-grounding-narrations/checkpoint/init_selfCLIP_single_train/epoch0030.pth.tar
```

**************************************************************

## Checkpoints


CLIP backbone finetuned on HowTo100M [Google Dirve](https://drive.google.com/file/d/1PDCySq8qAlm9dqxJE-DkpO1w2mjren7W/view?usp=drive_link)

Model weights [Google Dirve](https://drive.google.com/file/d/135ivdZTKA_F-UwwRzGYPSMeG1W3-4H4A/view?usp=drive_link)

**************************************************************


