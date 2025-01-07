# What, when, and where? -- Self-Supervised Spatio-Temporal Grounding in Untrimmed Multi-Action Videos from Narrated Instructions
******************************************************

This repo has the implementation of our paper: [What, when, and where? -- Self-Supervised Spatio-Temporal Grounding in Untrimmed Multi-Action Videos from Narrated Instructions](https://arxiv.org/abs/2303.16990)

![figure](figure.png)


## Getting Started


```
$ conda create -n stg --file req.txt
$ conda activate stg
```

**************************************************************

## Datasets
```
 ├── data
     └── Youcook2
         ├── annotation
         ├── validation
                └── *.mp4  (442 videos)
     └── GroundingYouTube
         ├── annotation
         ├── test
                └── *.mp4  ( videos)
```

### [YouCook2-Interactions](https://github.com/rxtan2/video-grounding-narrations?tab=readme-ov-file)

Raw videos are [here](https://huggingface.co/datasets/lmms-lab/YouCook2), we use the validation set 

### [Mining YouTube](https://github.com/hildekuehne/Mining_YouTube_dataset)

### [Grounding YouTube](https://github.com/brian7685/STG)

Raw videos for Mining YouTube and Grounding YouTube are [here](https://github.com/brian7685/STG)

**************************************************************

## Checkpoints
 
Put the checkpoints under checkpoint folder
 
```
 ├── checkpoint
     └── finetuned_CLIP_howto.pth.tar
     └── GroundingWeights.pth.tar
```

CLIP backbone finetuned on HowTo100M [Google Dirve](https://drive.google.com/file/d/1PDCySq8qAlm9dqxJE-DkpO1w2mjren7W/view?usp=drive_link)

Model weights [Google Dirve](https://drive.google.com/file/d/135ivdZTKA_F-UwwRzGYPSMeG1W3-4H4A/view?usp=drive_link)


**************************************************************


## Test Model

Evaluate youcook-inter with finetuned clip
```
CUDA_VISIBLE_DEVICE=1 python -W ignore eval_youcook_clip_finetune.py \
--eval_video_root data/Youcook2/validation/ \
--youcook2_annotations_path data/Youcook2/annotation/youcookii_annotations_trainval.json \
--interactions_annotations_path data/Youcook2/annotation/YouCook2-Interactions/final_dataset_annotations.pkl \
--interactions_segments_path data/Youcook2/annotation/YouCook2-Interactions/final_dataset_segments.pkl  \
--pretrain_clip checkpoint/finetuned_CLIP_howto.pth.tar \
--checkpoint_eval checkpoint/GroundingWeights.pth.tar
```

Evaluate GroundingYouTube with finetuned clip
```
CUDA_VISIBLE_DEVICE=1 python -W ignore eval_mining_clip_iou_finetune.py \
--eval_video_root data/GroundingYouTube/test/ \
--youcook2_annotations_path data/GroundingYouTube/grounding_anno/seg.json \
--interactions_annotations_path data/GroundingYouTube/grounding_anno/id2xy_box.json \
--pretrain_clip checkpoint/finetuned_CLIP_howto.pth.tar \
--checkpoint_eval checkpoint/GroundingWeights.pth.tar
```

**************************************************************




If you're using GroundingYouTube in your research or applications, please cite using this BibTeX:

```
@InProceedings{Chen_2024_CVPR,
    author    = {Chen, Brian and Shvetsova, Nina and Rouditchenko, Andrew and Kondermann, Daniel and Thomas, Samuel and Chang, Shih-Fu and Feris, Rogerio and Glass, James and Kuehne, Hilde},
    title     = {What When and Where? Self-Supervised Spatio-Temporal Grounding in Untrimmed Multi-Action Videos from Narrated Instructions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {18419-18429}
}
```