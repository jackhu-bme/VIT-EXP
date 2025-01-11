accelerate launch scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v4_1_80g.yaml  #--auto_resume
# accelerate launch scripts/run_train.py --config ct_clip_ori_hpc_1.yaml

# debug
# python scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v1_1_div_0_80g.yaml --debug
python scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v3_1_80g_debug.yaml --debug

# local 3195 machine debug
python scripts/run_train.py --config ct_clip_vit_open_seg_v2_1_debug_30_80g.yaml --debug

# resume training
accelerate launch scripts/run_train.py --config ct_clip_ori_hpc_2.yaml --resume ../ckpts/CTClip.80500.pt

# debug

python scripts/run_train.py --config ct_clip_debug_30_v2.yaml

