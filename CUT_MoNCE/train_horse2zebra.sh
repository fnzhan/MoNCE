srun -o file.out -e file.err -p dsta --mpi=pmi2 \
--gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=horse2zebra_hard0.1 \
--kill-on-bad-exit=1 \
-w SG-IDC1-10-51-2-37 \
python3 train.py \
--dataroot /mnt/lustre/fnzhan/projects/aaai2022/CUT/datasets/horse2zebra \
--name horse2zebra \
--CUT_mode CUT \
--cost_type 'hard' \
--eps 0.1 \
--neg_term_weight 1.0 \
--lambda_NCE 1.0 \
--batch_size 12 \
--gpu_ids '0' &
#--continue_train \
#--epoch_count 391 \
#--epoch 390
# -o file.out -e file.err
