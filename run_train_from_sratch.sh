accelerate launch scripts/run_train.py --config ct_clip_vit_seg_hpc_v3_1_div_0_80g_cl_revised_ref.yaml --auto_resume
# accelerate launch scripts/run_train.py --config ct_clip_ori_hpc_1.yaml

# debug
# python scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v1_1_div_0_80g.yaml --debug
python scripts/run_train.py --config ct_clip_vit_open_seg_hpc_v2_1_div_0_80g.yaml --debug

# resume training
accelerate launch scripts/run_train.py --config ct_clip_ori_hpc_2.yaml --resume ../ckpts/CTClip.80500.pt

# debug

python scripts/run_train.py --config ct_clip_debug_30_v2.yaml

