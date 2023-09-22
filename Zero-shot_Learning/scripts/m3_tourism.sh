
model=GPT4TS
source_data=m3
target_data=tourism

seq_len=12
pred_len=4
test_seq_len=12
test_pred_len=4


python main_test.py \
    --root_path ./datasets/m3/ \
    --test_root_path ./datasets/tourism/ \
    --data_path m3_yearly_dataset.tsf \
    --test_data_path tourism_yearly_dataset.tsf \
    --model_id tourismYearly_$source_data'_'$model'_'$gpt_layer'_lr'$lr'_decay'$decay'_dropout'$dropout'_epoch'$epoch'_percent'$percent \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $seq_len \
    --test_pred_len $test_pred_len \
    --label_len 0 \
    --batch_size 512 \
    --test_batch_size 128 \
    --learning_rate 0.002 \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func mape \
    --dropout 0 \
    --percent 100 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --patch_size 1 \
    --stride 1 \
    --print_int 1000 \
    --itr 3 \
    --is_gpt 1

seq_len=24
pred_len=8
test_seq_len=24
test_pred_len=8
model=GPT4TS
source_data=m3
target_data=tourism

python main_test.py \
    --root_path ./datasets/m3/ \
    --test_root_path ./datasets/tourism/ \
    --data_path m3_quarterly_dataset.tsf \
    --test_data_path tourism_quarterly_dataset.tsf \
    --model_id tourismQuarterly_$source_data'_'$model'_'$gpt_layer'_lr'$lr'_percent'$percent'_batch'$batch_size \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $test_seq_len \
    --test_pred_len $test_pred_len \
    --label_len 0 \
    --batch_size 128 \
    --test_batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func mape \
    --percent 100 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --dropout 0 \
    --patch_size 2 \
    --stride 2 \
    --print_int 1000 \
    --itr 3 \
    --is_gpt 1

seq_len=36
pred_len=24
test_seq_len=36
test_pred_len=24

model=GPT4TS
source_data=m3
target_data=tourism

python main_test.py \
    --root_path ./datasets/m3/ \
    --test_root_path ./datasets/tourism/ \
    --data_path m3_monthly_dataset.tsf \
    --test_data_path tourism_monthly_dataset.tsf \
    --model_id tourismMonthly_$source_data'_'$model'_'$gpt_layer'_lr'$lr'_percent'$percent'_batch'$batch_size \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $seq_len \
    --test_pred_len $test_pred_len \
    --label_len 0 \
    --batch_size 32 \
    --test_batch_size 128 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func mape \
    --percent 100 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --dropout 0 \
    --patch_size 1 \
    --stride 1 \
    --print_int 1000 \
    --itr 3 \
    --is_gpt 1