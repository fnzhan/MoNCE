python3 train.py \
--dataroot /mnt/lustre/fnzhan/datasets/CVPR2022/cityscapes \
--name cityscapes_eps0.5_p1_n0.5 \
--CUT_mode CUT \
--lambda_NCE 1 \
--batch_size 12 \
--gpu_ids '0' \
--ot_weight 128.0 &
#--continue_train

