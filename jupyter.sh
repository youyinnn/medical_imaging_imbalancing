nohup jupyter notebook > jpt.log 2>&1 &
nohup tensorboard --logdir='.' --port=26657 > tsb.log 2>&1 &
echo $! > save_pid.log
