DATA_DIR=${1}
OUTPUT_DIR=${2}
checkpoint_path=${3}

model=bimobilenet
num_workers=32
checkpoint_name=${checkpoint_path##*/}
checkpoint_name=${checkpoint_name%%.*}

mkdir ${OUTPUT_DIR}
OUTPUT_DIR=$OUTPUT_DIR/${checkpoint_name}
mkdir ${OUTPUT_DIR}

echo DATA_DIR=${DATA_DIR}
echo OUTPUT_DIR=${OUTPUT_DIR}
echo checkpoint_path=${checkpoint_path}
echo checkpoint_name=${checkpoint_name}
echo num_workers=1

# test videomatte_1920x1080
echo test-videomatte_1920x1080 start: $(date "+%Y-%m-%d-%H-%M-%S")
mkdir ${OUTPUT_DIR}/videomatte_1920x1080
echo test-videomatte_1920x1080-inference start: $(date "+%Y-%m-%d-%H-%M-%S")
python3 multi_inference.py \
    --variant=${model} \
    --checkpoint=${checkpoint_path} \
    --device cuda \
    --input-source "${DATA_DIR}/videomatte_1920x1080/" \
    --output-type jpg_sequence \
    --output-alpha "${OUTPUT_DIR}/videomatte_1920x1080/" \
    --output-foreground "${OUTPUT_DIR}/videomatte_1920x1080/" \
    --seq-chunk 1 \
    --disable-progress 

echo test-videomatte_1920x1080-evaluation start: $(date "+%Y-%m-%d-%H-%M-%S")
python3 evaluation/evaluate_hr.py \
    --pred-dir "${OUTPUT_DIR}/videomatte_1920x1080" \
    --true-dir "${DATA_DIR}/videomatte_1920x1080" \
    --log-dir "${OUTPUT_DIR}" \
    --num-workers 1 
echo test-videomatte_1920x1080 finished: $(date "+%Y-%m-%d-%H-%M-%S")

# test videomatte_512x288
echo test-videomatte_512x288 start: $(date "+%Y-%m-%d-%H-%M-%S")
echo test-videomatte_512x288-inference start: $(date "+%Y-%m-%d-%H-%M-%S")
mkdir ${OUTPUT_DIR}/videomatte_512x288

python3 multi_inference.py \
    --variant=${model} \
    --checkpoint=${checkpoint_path} \
    --device cuda \
    --input-source "${DATA_DIR}/videomatte_512x288/" \
    --output-type png_sequence \
    --output-alpha "${OUTPUT_DIR}/videomatte_512x288/" \
    --output-foreground "${OUTPUT_DIR}/videomatte_512x288/" \
    --seq-chunk 1 \
    --disable-progress 

echo test-videomatte_512x288-evaluation start: $(date "+%Y-%m-%d-%H-%M-%S")
python3 evaluation/evaluate_lr.py \
    --pred-dir "${OUTPUT_DIR}/videomatte_512x288" \
    --true-dir "${DATA_DIR}/videomatte_512x288" \
    --log-dir "${OUTPUT_DIR}" \
    --num-workers 1 
echo test-videomatte_512x288 finished: $(date "+%Y-%m-%d-%H-%M-%S")
