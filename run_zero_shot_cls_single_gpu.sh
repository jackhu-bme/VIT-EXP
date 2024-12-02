export CONFIG=configs/train_from_scratch/ct_clip_vit_hpc_v3_1.yaml
export MODEL=/mnt/input/CT-CLIP-VIT/train_from_scratch_vit_hpc_1_dim_384/2024-11-22_01-06-13/checkpoints/CTClip.200000.pt
export RESULTS=results_fast_inference_zeroshot_ctvit_full


python scripts/run_zero_shot_cls_single_gpu.py --config ${CONFIG} --model ${MODEL} --results ${RESULTS} 