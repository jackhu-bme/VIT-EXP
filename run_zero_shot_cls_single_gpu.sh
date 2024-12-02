export CONFIG=configs/train_from_scratch/ct_clip_vit_hpc_v3_1.yaml
export MODEL=../output_train_scratch/train_from_scratch_vit_hpc_1_dim_384/2024-11-16_06-57-22/checkpoints/CTClip.100000.pt
export RESULTS=results_inference_zeroshot_ctvit


python scripts/run_zero_shot_cls_single_gpu.py --config ${CONFIG} --model ${MODEL} --results ${RESULTS} 