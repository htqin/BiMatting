DataDir=${1}
model=bimobilenet
echo model-variant=${model}
num_workers=4
prefix=${DataDir}/matting-data
save_dir_local=checkpoints
log_dir_local=logs

echo num_workers=${num_workers}

# Stage 1
echo train-stage-1 start: $(date "+%Y-%m-%d-%H-%M-%S")
python3 train.py \
    --model-variant ${model} \
    --prefix ${prefix} \
    --num-workers ${num_workers} \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 15 \
    --learning-rate-backbone 0.0001 \
    --learning-rate-aspp 0.0002 \
    --learning-rate-decoder 0.0002 \
    --learning-rate-refiner 0 \
    --checkpoint-dir ${save_dir_local}/stage1 \
    --log-dir ${log_dir_local}/stage1 \
    --epoch-start 0 \
    --epoch-end 20 \
    --disable-validation
echo train-stage-1 finished: $(date "+%Y-%m-%d-%H-%M-%S")

# Stage 2
echo train-stage-2 start: $(date "+%Y-%m-%d-%H-%M-%S")
python3 train.py \
    --model-variant ${model} \
    --prefix ${prefix} \
    --num-workers ${num_workers} \
    --dataset videomatte \
    --resolution-lr 512 \
    --seq-length-lr 20 \
    --learning-rate-backbone 0.00005 \
    --learning-rate-aspp 0.0001 \
    --learning-rate-decoder 0.0001 \
    --learning-rate-refiner 0 \
    --checkpoint ${save_dir_local}/stage1/epoch-19.pth \
    --checkpoint-dir ${save_dir_local}/stage2 \
    --log-dir ${log_dir_local}/stage2 \
    --epoch-start 20 \
    --epoch-end 22 \
    --disable-validation
echo train-stage-2 finished: $(date "+%Y-%m-%d-%H-%M-%S")
    
# Stage 3
echo train-stage-3 start: $(date "+%Y-%m-%d-%H-%M-%S")
python3 train.py \
    --model-variant ${model} \
    --prefix ${prefix} \
    --num-workers ${num_workers} \
    --dataset videomatte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 40 \
    --seq-length-hr 6 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00001 \
    --learning-rate-refiner 0.0002 \
    --checkpoint ${save_dir_local}/stage2/epoch-21.pth \
    --checkpoint-dir ${save_dir_local}/stage3 \
    --log-dir ${log_dir_local}/stage3 \
    --epoch-start 22 \
    --epoch-end 23 \
    --disable-validation
echo train-stage-3 finished: $(date "+%Y-%m-%d-%H-%M-%S")

# Stage 4
echo train-stage-4 start: $(date "+%Y-%m-%d-%H-%M-%S")
python3 train.py \
    --model-variant ${model} \
    --prefix ${prefix} \
    --num-workers ${num_workers} \
    --dataset imagematte \
    --train-hr \
    --resolution-lr 512 \
    --resolution-hr 2048 \
    --seq-length-lr 20 \
    --seq-length-hr 6 \
    --learning-rate-backbone 0.00001 \
    --learning-rate-aspp 0.00001 \
    --learning-rate-decoder 0.00005 \
    --learning-rate-refiner 0.0002 \
    --checkpoint ${save_dir_local}/stage3/epoch-22.pth \
    --checkpoint-dir ${save_dir_local}/stage4 \
    --log-dir ${log_dir_local}/stage4 \
    --epoch-start 23 \
    --epoch-end 28 \
   --disable-validation
echo train-stage-4 finished: $(date "+%Y-%m-%d-%H-%M-%S")
