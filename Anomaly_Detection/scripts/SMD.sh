

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMD \
  --model_id SMD \
  --model GPT4TS \
  --data SMD \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --d_model 768 \
  --d_ff 768 \
  --gpt_layer 6 \
  --enc_in 38 \
  --c_out 38 \
  --anomaly_ratio 0.5 \
  --batch_size 128 \
  --patch_size 1 \
  --stride 1 \
  --train_epochs 5