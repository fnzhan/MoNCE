srun -p dsta --mpi=pmi2 \
--gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=test_ade20k \
--kill-on-bad-exit=1 \
-w SG-IDC1-10-51-2-37 \
python3 test.py \
--name ade20k \
--dataset_mode ade20k \
--dataroot /mnt/lustre/fnzhan/datasets/CVPR2022/ade20k/ \
--batchSize 10 \
#--continue_train
