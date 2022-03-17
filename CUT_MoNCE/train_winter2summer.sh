srun -o file.out -e file.err -p dsta --mpi=pmi2 \
--gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=conce_winter2summer \
--kill-on-bad-exit=1 \
-w SG-IDC1-10-51-2-37 \
python3 train.py \
--dataroot /mnt/lustre/fnzhan/datasets/CVPR2022/winter2summer \
--name winter2summer_inverse1_p1_n0.5_vgg2 \
--CUT_mode CUT \
--lambda_NCE 2 \
--batch_size 12 \
--gpu_ids '0' \
--ot_weight 127.5 &
#--continue_train \
#--epoch_count 391 \
#--epoch 390
# -o file.out -e file.err
