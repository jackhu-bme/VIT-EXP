# accelerate launch --use_fsdp scripts/run_train.py --config ct_clip_ori_hpc_1.yaml
accelerate launch scripts/run_train.py --config ct_clip_ori_hpc_2.yaml


# resume training
accelerate launch scripts/run_train_resume.py --config ct_clip_ori_hpc_2.yaml --resume ../ckpts/CTClip.80500.pt
