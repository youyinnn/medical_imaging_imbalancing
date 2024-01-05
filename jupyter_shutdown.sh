kill -9 `cat save_pid.log`
rm save_pid.log

cp jpt.log jpt.$(date +%s).bk.log
rm jpt.log

cp tsb.log tsb.$(date +%s).bk.log
rm tsb.log