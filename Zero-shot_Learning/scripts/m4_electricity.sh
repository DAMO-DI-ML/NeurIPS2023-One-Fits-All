export CUDA_VISIBLE_DEVICES=0
    
model=GPT4TS

seq_len=30
pred_len=168
test_seq_len=30
test_pred_len=168

python main_test.py \
    --root_path ./datasets/m4/ \
    --test_root_path ./datasets/electricity_tsf/ \
    --data_path m4_hourly_dataset.tsf \
    --test_data_path electricity_hourly_dataset.tsf \
    --model_id m4Hourly_$model'_'$seq_len'_'$pred_len \
    --data tsf_data \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --test_seq_len $test_seq_len \
    --test_pred_len $test_pred_len \
    --label_len 0 \
    --batch_size 512 \
    --test_batch_size 128 \
    --learning_rate 0.005 \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 16 \
    --d_ff 512 \
    --loss_func smape \
    --percent 100 \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --patch_size 2 \
    --stride 1 \
    --print_int 1000