for start_task in 0 10 20 30 40; do
    python lats.py --start_task $start_task --task_num 10 --save_folder runtime_logs &
done