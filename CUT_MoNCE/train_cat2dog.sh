srun -o file.out -e file.err -p dsta --mpi=pmi2 \
--gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=cut_cat2dog_inverse \
--kill-on-bad-exit=1 \
-w SG-IDC1-10-51-2-37 \
python3 train.py \
--dataroot /mnt/lustre/fnzhan/datasets/CVPR2022/cat2dog \
--name cat2dog_inverse1_p1_n1.5 \
--CUT_mode CUT \
--lambda_NCE 1.0 \
--ot_weight 382.0 \
--batch_size 12 \
--gpu_ids '0' &
#--continue_train
