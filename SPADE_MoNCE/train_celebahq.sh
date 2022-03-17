srun -o file.out -e file.err -p dsta --mpi=pmi2 \
--gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=conce_celebahq \
--kill-on-bad-exit=1 \
-w SG-IDC1-10-51-2-37 \
python3 train.py \
--name celebahq \
--dataset_mode celebahq \
--dataroot '/mnt/lustre/fnzhan/datasets/ICCV2021/CelebAMask-HQ' \
--batchSize 20 \
--niter 30 \
--niter_decay 30 \
--ot_weight 128 \
--lambda_vgg 100 \
--continue_train &
# -o file.out -e file.err 
