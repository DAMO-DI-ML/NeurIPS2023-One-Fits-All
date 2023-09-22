

python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --root_path ./dataset/SMAP \
  --model_id SMAP \
  --model GPT4TS \
  --data SMAP \
  --features M \
  --seq_len 100 \
  --pred_len 0 \
  --gpt_layer 6 \
  --d_model 768 \
  --d_ff 768 \
  --patch_size 1 \
  --stride 1 \
  --enc_in 25 \
  --c_out 25 \
  --anomaly_ratio 1 \
  --batch_size 128 \
  --learning_rate 0.0005 \
  --train_epochs 10
  