source activate mar
module load anaconda3_gpu
# export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "/sw/external/libraries/cudnn-linux-x86_64-8.9.0.131_cuda11-archive/lib" | paste -sd ':' -)
/u/zongzhex/.conda/envs/mar/bin/torchrun --nnodes=1 --nproc_per_node=4 main_mar.py --patch_size 5 --online_eval \
--model mar_base --epochs 400 --warmup_epochs 100 --batch_size 32 --blr 1.0e-4