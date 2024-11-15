accelerate launch scripts/run_train.py --config ct_clip_vit_hpc_v2_1.yaml
# accelerate launch scripts/run_train.py --config ct_clip_ori_hpc_1.yaml


# resume training
accelerate launch scripts/run_train.py --config ct_clip_ori_hpc_2.yaml --resume ../ckpts/CTClip.80500.pt

# debug

python scripts/run_train.py --config ct_clip_debug_30_v2.yaml

