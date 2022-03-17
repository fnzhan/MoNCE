srun -o file.out -e file.err -p dsta --mpi=pmi2 \
--gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=spade_conce_ade20k \
--kill-on-bad-exit=1 \
-w SG-IDC1-10-51-2-37 \
python3 train.py \
--name ade20k \
--dataset_mode ade20k \
--dataroot /mnt/lustre/fnzhan/datasets/CVPR2022/ade20k/ \
--batchSize 12 \
--niter 100 \
--niter_decay 100 \
--ot_weight 128 \
--lambda_vgg 50 \
--continue_train &
#--which_epoch 150
