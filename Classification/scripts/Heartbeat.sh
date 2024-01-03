export CUDA_VISIBLE_DEVICES=1
for lr in 0.0005
do
for patch in 32
do
for stride in 16
do

python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name Heartbeat \
    --records_file Classification_records.xls \
    --data_dir ./datasets/Heartbeat \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 50 \
    --lr $lr \
    --patch_size $patch \
    --stride $stride \
    --optimizer RAdam \
    --d_model 768 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy

done
done
done