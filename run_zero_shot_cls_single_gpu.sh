export CONFIG=configs/train_from_scratch/ct_clip_vit_hpc_v3_1_div_0_revised_cl.yaml
export MODEL=/mnt/input/CT-CLIP-VIT/train_from_scratch_vit_hpc_1_dim_384_div0_cl_revised/2024-12-05_14-02-26/checkpoints/CTClip.40000.ptls
export RESULTS=results_fast_inference_zeroshot_ctvit_full_cl_revised


python scripts/run_zero_shot_cls_single_gpu.py --config ${CONFIG} --model ${MODEL} --results ${RESULTS} 