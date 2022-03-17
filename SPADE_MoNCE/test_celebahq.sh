srun -p dsta --mpi=pmi2 \
--gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=test_ade20k \
--kill-on-bad-exit=1 \
-w SG-IDC1-10-51-2-37 \
python3 test.py \
--name celebahq \
--dataset_mode celebahq \
--dataroot '/mnt/lustre/fnzhan/datasets/ICCV2021/CelebAMask-HQ' \
--batchSize 10 \
#--continue_train
