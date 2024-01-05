kill -9 `cat save_jpt_pid.txt`
rm save_jpt_pid.txt
cp jpt.log jpt.$(date +%s).bk.log
rm jpt.log