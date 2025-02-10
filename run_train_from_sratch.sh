# setting the environment for NCCL timeout, as validation on full dataset causes waiting of training dataloaders!!!

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

accelerate launch scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v3_1_80g.yaml  #--auto_resume
accelerate launch scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v5_1_80g_seg_only.yaml # only use the segmenataion
accelerate launch scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v5_1_80g_seg_only_fix_te_1cls.yaml # only use the segmenataion and freeze the text encoder
accelerate launch scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v5_1_80g_fix_te_mlp_fusion_try1.yaml # only use the segmenataion and freeze the text encoder
# CUDA_VISIBLE_DEVICES=0,1
# accelerate launch --main_process_port 29600 scripts/run_train.py --config ct_clip_ori_hpc_1.yaml

# debug
# python scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v1_1_div_0_80g.yaml --debug
python scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v3_1_80g_debug.yaml --debug
python scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v3_1_80g_debug_resume.yaml --resume /mnt/input/CT-CLIP/ctclip_vit_open_seg_hpc_v4_1_80g_final2/checkpoints/CTClip.30000.pt

# local 3195 machine debug
python scripts/run_train.py --config ct_clip_vit_open_seg_v2_1_debug_30_80g.yaml --debug

# resume training
accelerate launch scripts/run_train.py --config ct_clip_ori_hpc_2.yaml --resume ../ckpts/CTClip.80500.pt

# debug

python scripts/run_train.py --config ct_clip_debug_30_v2.yaml

