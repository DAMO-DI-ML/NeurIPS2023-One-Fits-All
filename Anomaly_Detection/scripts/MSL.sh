

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/MSL \
  --model_id MSL \
  --model GPT4TS \
  --data MSL \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --gpt_layer 6 \
  --d_model 768 \
  --d_ff 8 \
  --patch_size 1 \
  --stride 1 \
  --enc_in 55 \
  --c_out 55 \
  --anomaly_ratio 2 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --train_epochs 10