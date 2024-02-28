for user_idx in 0 1 2 3 4 5 6 7 8 9; do
    $python reflexion.py \
            --num_trials 3 \
            --num_envs 50 \
            --run_name "reflexion_run_logs_ua2webshop/run_user_${user_idx}" \
            --use_memory \
            --user_id $user_idx &
    sleep 5
done