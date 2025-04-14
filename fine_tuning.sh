#!/usr/bin/env bash

# Thiết lập GPU sử dụng
export CUDA_VISIBLE_DEVICES=0 # 0 nếu như bạn có GPU của nvidia :v

log() {
    echo "$@"
}

# Tạo thư mục cần thiết, 
DATASET_DIR="/content/ckpts/your_training_dataset"
mkdir -p "$DATASET_DIR"
# Bắt buộc phải có thư mục data/your_dataset chứa các file .wav, file .txt tương ứng, các bạn tự xử lý
mkdir -p data/your_dataset

# Định nghĩa các tham số huấn luyện
EXP_NAME="F5TTS_100h_Custom"
DATASET_NAME="your_training_dataset"
BATCH_SIZE=4000
NUM_WOKERS=16
WARMUP_UPDATES=20000
SAVE_UPDATES=10000
LAST_UPDATES=10000
PRETRAIN_CKPT="ckpts/your_training_dataset/model_500000.pt"

# Tạo các biến stage để quản lý pipeline, bước nào đã chạy rồi thì không cần chạy lại
stage=0
stop_stage=5

# Chuẩn hoá sample_rate, bỏ qua stage này nếu audio của bạn đã ở định dạng 24Khz
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    log "Convert sample rate: data/your_dataset ..."
    python convert_sr.py
fi

# Chuẩn bị dữ liệu audio_name và text tương ứng
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    log "Preparing metadata at: data/your_dataset ..."
    python prepare_metadata.py
fi

# Bổ sung từ vựng trong bộ dữ liệu của bạn chưa có trong từ vựng của mô hình pretrained
if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
    log "Checking missing token in pretrained vocab ... "
    python check_vocab_pretrained.py
fi

# Mở rộng embedding của mô hình pretrained để hỗ trợ bộ từ vựng mới
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
    log "Extend embedding pretrained with new vocab ... "
    python extend_embedding_pretrained.py
fi

# Trích xuất đặc trưng
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
    log "Feature extraction ... "
    python src/f5_tts/train/datasets/prepare_csv_wavs.py "$DATASET_DIR" "$DATASET_DIR" --workers "$NUM_WOKERS"
fi

# Chạy quá trình fine-tuning
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
    log "Start fine-tuning F5-TTS with your dataset ... "
    python src/f5_tts/train/finetune_cli.py \
        --exp_name "$EXP_NAME" \
        --dataset_name "$DATASET_NAME" \
        --batch_size_per_gpu "$BATCH_SIZE" \
        --num_warmup_updates "$WARMUP_UPDATES" \
        --save_per_updates "$SAVE_UPDATES" \
        --last_per_updates "$LAST_UPDATES" \
        --finetune \
        --log_samples \
        --pretrain "$PRETRAIN_CKPT"
    ### Nếu bạn muốn training với nhiều gpu, sử dụng câu lệnh bên dưới:
    # accelerate launch python src/f5_tts/train/finetune_cli.py \
    #     --exp_name "$EXP_NAME" \
    #     --dataset_name "$DATASET_NAME" \
    #     --batch_size_per_gpu "$BATCH_SIZE" \
    #     --num_warmup_updates "$WARMUP_UPDATES" \
    #     --save_per_updates "$SAVE_UPDATES" \
    #     --last_per_updates "$LAST_UPDATES" \
    #     --finetune \
    #     --log_samples \
    #     --pretrain "$PRETRAIN_CKPT"
fi

log "Fine-tuning F5-TTS done."