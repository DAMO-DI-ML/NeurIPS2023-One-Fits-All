for lr in 0.001
do
for patch in 16
do
for stride in 16
do

python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name UWaveGestureLibrary \
    --records_file Classification_records.xls \
    --data_dir ./datasets/UWaveGestureLibrary \
    --data_class tsra \
    --pattern TRAIN \
    --val_pattern TEST \
    --epochs 50 \
    --lr 0.001 \
    --patch_size 16 \
    --stride 16 \
    --optimizer RAdam \
    --d_model 768 \
    --pos_encoding learnable \
    --task classification \
    --key_metric accuracy

done
done
done