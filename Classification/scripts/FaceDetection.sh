
for patch in 4
do
for stride in 2
do
for lr in 0.0001
do

python src/main.py \
    --output_dir experiments \
    --comment "classification from Scratch" \
    --name FaceDetection \
    --records_file Classification_records.xls \
    --data_dir ./datasets/FaceDetection \
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