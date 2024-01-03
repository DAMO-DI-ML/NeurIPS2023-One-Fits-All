export CUDA_VISIBLE_DEVICES=1
for lr in 0.002
do
for patch in 8
do
for stride in 2
do

python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name Handwriting \
    --records_file Classification_records.xls \
    --data_dir ./datasets/Handwriting \
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
    --key_metric accuracy \
    --lr_step 10,20,30,40 \
    --lr_factor 0.6

done
done
done