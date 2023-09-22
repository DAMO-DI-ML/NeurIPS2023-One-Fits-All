model=GPT4TS

seq_len=12
pred_len=6
test_seq_len=12
test_pred_len=6

python main_test.py \
    --root_path ./datasets/m4/ \
    --test_root_path ./datasets/m3/ \
    --data_path m4_yearly_dataset.tsf \
    --test_data_path m3_yearly_dataset.tsf \
    --model_id m3Yearly'_'$model \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $seq_len \
    --test_pred_len $test_pred_len \
    --label_len 0 \
    --batch_size 512 \
    --test_batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 20 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func smape \
    --dropout 0 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --patch_size 1 \
    --stride 1 \
    --print_int 1000


seq_len=24
pred_len=8
test_seq_len=24
test_pred_len=8

python main_test.py \
    --root_path ./datasets/m4/ \
    --test_root_path ./datasets/m3/ \
    --data_path m4_quarterly_dataset.tsf \
    --test_data_path m3_quarterly_dataset.tsf \
    --model_id m3Quarterly'_'$model \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $test_seq_len \
    --test_pred_len $test_pred_len \
    --label_len 0 \
    --batch_size 512 \
    --test_batch_size 128 \
    --learning_rate 0.01 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func mape \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --patch_size 1 \
    --stride 1 \
    --print_int 1000


seq_len=24
pred_len=18
test_seq_len=24
test_pred_len=18

python main_test.py \
    --root_path ./datasets/m4/ \
    --test_root_path ./datasets/m3/ \
    --data_path m4_monthly_dataset.tsf \
    --test_data_path m3_monthly_dataset.tsf \
    --model_id tourismMonthly'_'$model \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $test_seq_len \
    --test_pred_len $test_pred_len \
    --label_len 0 \
    --batch_size 2048 \
    --test_batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func mape \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --patch_size 2 \
    --stride 2 \
    --print_int 1000 


seq_len=16
pred_len=8
test_seq_len=16
test_pred_len=8

python main_test.py \
    --root_path ./datasets/m4/ \
    --test_root_path ./datasets/m3/ \
    --data_path m4_monthly_dataset.tsf \
    --test_data_path m3_other_dataset.tsf \
    --model_id m3Other_$model \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $test_seq_len \
    --test_pred_len $test_pred_len \
    --label_len 0 \
    --batch_size 2048 \
    --test_batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func mape \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --patch_size 2 \
    --stride 2 \
    --print_int 1000