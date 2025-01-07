import argparse

def get_args(description='MILNCE'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--train_csv',
        type=str,
        default='csv/howto100m_videos.csv',
        help='train csv')
    parser.add_argument(
        '--video_path',
        type=str,
        default='/nobackup/projects/public/howto100m/parsed_videos/',
        help='video_path')
    parser.add_argument(
        '--caption_root',
        type=str,
        default='data/howto100m_csv',
        help='video_path')
    parser.add_argument(
        '--checkpoint_root',
        type=str,
        default='checkpoint',
        help='checkpoint dir root')
    parser.add_argument(
        '--checkpoint_eval',
        type=str,
        default='./pretrained_checkpoints/epoch0005.pth.tar',
        help='checkpoint dir root')
    parser.add_argument(
        '--pretrain_clip',
        type=str,
        default='./pretrained_checkpoints/epoch0005.pth.tar',
        help='clip checkpoint dir root')
    parser.add_argument(
        '--log_root',
        type=str,
        default='log',
        help='log dir root')
    parser.add_argument(
        '--eval_video_root',
        type=str,
        default='/nobackup/users/brian27/ECCV22/mil_nce/my_data/youcook/validation/',
        help='root folder for the video at for evaluation')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='',
        help='checkpoint model folder')
    parser.add_argument(
        '--youcook2_annotations_path',
        type=str,
        default="/research/rxtan/neurips_rebuttal/pretrained_s3d_baseline/youcookii_annotations_trainval.json",
        help='path to original youcook2 instructions')
    parser.add_argument(
        '--interactions_annotations_path',
        type=str,
        default="/research/rxtan/neurips_rebuttal/reuben/final_dataset_annotations.pkl",
        help='path to youcook2-interaction annotations')
    parser.add_argument(
        '--interactions_segments_path',
        type=str,
        default="/research/rxtan/neurips_rebuttal/reuben/final_dataset_segments.pkl",
        help='path to youcook2-interaction segments')
    parser.add_argument(
        '--optimizer', type=str, default='adam', help='opt algorithm')
    parser.add_argument('--pretrained_path', type=str, default='',
                                help='CNN weights inits')
    parser.add_argument('--pretrain_cnn_path', type=str, default='',
                                help='CNN weights inits')
    parser.add_argument('--weight_init', type=str, default='uniform',
                                help='CNN weights inits')
    parser.add_argument('--thre', type=float, default=0.0001,
                                help='')
    parser.add_argument('--num_thread_reader', type=int, default=20,
                                help='')
    parser.add_argument('--num_class', type=int, default=512,
                                help='upper epoch limit')
    parser.add_argument('--num_candidates', type=int, default=1,
                                help='num candidates for MILNCE loss')
    parser.add_argument('--batch_size', type=int, default=512,
                                help='batch size')
    parser.add_argument('--num_windows_test', type=int, default=4,
                                help='number of testing windows')
    parser.add_argument('--batch_size_val', type=int, default=32,
                                help='batch size eval')
    parser.add_argument('--momemtum', type=float, default=0.9,
                                help='SGD momemtum')
    parser.add_argument('--n_display', type=int, default=10,
                                help='Information display frequence')
    parser.add_argument('--num_frames', type=int, default=16,
                                help='random seed')
    parser.add_argument('--video_size', type=int, default=224,
                                help='random seed')
    parser.add_argument('--crop_only', type=int, default=1,
                                help='random seed')
    parser.add_argument('--centercrop', type=int, default=0,
                                help='random seed')
    parser.add_argument('--random_flip', type=int, default=1,
                                help='random seed')
    parser.add_argument('--verbose', type=int, default=1,
                                help='')
    parser.add_argument('--warmup_steps', type=int, default=5000,
                                help='')
    parser.add_argument('--min_time', type=float, default=5.0,
                                help='')
    parser.add_argument(
        '--dist_url', type=str, default='tcp://localhost:23456', help='')
    parser.add_argument(
        '--word2vec_path', type=str, default='data/word2vec.pth', help='')
    parser.add_argument('--howto_audio_frames', type=int, default=1000,
                            help='number of frames to use for loading howto100m audio')
        
    parser.add_argument('--fps', type=int, default=5, help='')
    parser.add_argument('--cudnn_benchmark', type=int, default=0,
                                help='')
    parser.add_argument('--n_pair', type=int, default=1,
                                help='')
    parser.add_argument(
        '--pretrain_audio_path',
        type=str,
        default='',
        help='')    
    parser.add_argument(
        '--pretrain_mil_path',
        type=str,
        default='',
        help='')                          
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--finetune', action='store_true',
                        help='resume training from last checkpoint')    
    parser.add_argument('--cross_consist', action='store_true',
                        help='resume training from last checkpoint')  
                     
    parser.add_argument('--MMS_loss', dest='MMS_loss', action='store_true',
                        help='resume training from last checkpoint') 
                        
    parser.add_argument('--global_within', dest='global_within', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--gating_feature', dest='gating_feature', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--self_select', dest='self_select', action='store_true',
                        help='resume training from last checkpoint') 
                        
    parser.add_argument('--one_frame', dest='one_frame', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--no_local', dest='no_local', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--ht_all', dest='ht_all', action='store_true',
                        help='resume training from last checkpoint') 
                        
    parser.add_argument('--global_text', dest='global_text', type=int, default=1,
                        help='resume training from last checkpoint') 
    parser.add_argument('--sink', dest='sink', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--project', dest='project', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--project_old', dest='project_old', action='store_true',
                        help='resume training from last checkpoint') 
                        
    parser.add_argument('--prompt', dest='prompt', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--cpu', dest='cpu', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--l2_norm_all', dest='l2_norm_all', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--l2_norm_self', dest='l2_norm_self', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--ht370', dest='ht370', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--notrain', dest='notrain', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--fix_start', dest='fix_start', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--early_start', dest='early_start', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--l2_norm', dest='l2_norm', action='store_true',
                        help='resume training from last checkpoint') 
                        #l2_norm l2_norm_all
    parser.add_argument('--onlyAudio', dest='onlyAudio', action='store_true',
                        help='resume training from last checkpoint') 
    parser.add_argument('--globalF', dest='globalF', action='store_true',
                        help='resume training from last checkpoint')                    
    parser.add_argument('--trainAudio', dest='trainAudio', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--all', dest='all', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--sentence', dest='sentence', action='store_true',
                        help='resume training from last checkpoint')
                        
    parser.add_argument('--from_mil', dest='from_mil', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--fix_bn', dest='fix_bn', action='store_true',
                help='Use matchmap triplet loss (more memory efficient since O(n))')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--sc', dest='sc', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--cc', dest='cc', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--cs', dest='cs', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--r50', dest='r50', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('--sparsity', dest='sparsity', default=1, type=float,
                        help='resume training from last checkpoint')
    parser.add_argument('--ATLoss', dest='ATLoss', default=1, type=float,
                        help='resume training from last checkpoint')
    parser.add_argument('--VALoss', dest='VALoss', default=1, type=float,
                        help='resume training from last checkpoint')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--hitchhiker_local', dest='hitchhiker_local', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--hitchhiker_local_pool', dest='hitchhiker_local_pool', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--hitchhiker_norm', dest='hitchhiker_norm', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--global_select', dest='global_select', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--local_select', dest='local_select', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--video_mil', dest='video_mil', action='store_true',
                        help='use pre-trained model')      
             
    parser.add_argument('--num_sec_control', dest='num_sec_control', default=0, type=int,
                        help='resume training from last checkpoint')
                        
    parser.add_argument('--local_mean', dest='local_mean', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--longer_frame', dest='longer_frame', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--video_mil_within', dest='video_mil_within', action='store_true',
                        help='use pre-trained model')
                        
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                        help='use pin_memory')
    parser.add_argument('--resnet', dest='resnet', action='store_true',
                        help='use pin_memory')
    parser.add_argument('--CLIP', dest='CLIP', action='store_true',
                        help='use pin_memory')
    parser.add_argument('--max_pool', dest='max_pool', action='store_true',
                        help='use pin_memory')                    
    parser.add_argument('--half', dest='half', action='store_true',
                        help='use pin_memory')
    parser.add_argument('--multi_loss', dest='multi_loss', action='store_true',
                        help='use pin_memory')
    parser.add_argument('--speed', default=2, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--method', default='', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-file', default='dist-file', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()
    return args