srun -p dsta --mpi=pmi2 \
--gres=gpu:1 -n1 --ntasks-per-node=1 \
--job-name=cut \
--kill-on-bad-exit=1 \
-w SG-IDC1-10-51-2-37 \
python3 train.py \
--dataroot ./datasets/grumpifycat \
--name grumpycat_CUT \
--CUT_mode CUT \
--continue_train
