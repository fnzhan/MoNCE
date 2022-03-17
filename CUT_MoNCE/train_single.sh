srun -p dsta --mpi=pmi2 \
--gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=cut_single \
--kill-on-bad-exit=1 \
-w SG-IDC1-10-51-2-37 \
python3 train.py \
--model sincut \
--name singleimage_monet_etretat1.5 \
--dataroot ./datasets/single_image_monet_etretat \
--batch_size 128 \
--gpu_ids '0' \
--ot_weight 382.0 \
#--continue_train
