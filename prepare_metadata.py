"""
Mô-đun chuẩn bị metadata cho tập dữ liệu huấn luyện.
Tạo file metadata.csv và vocab từ tập dữ liệu âm thanh + TextGrid.
"""

import os
import glob
import shutil
import soundfile as sf
import textgrid
from tqdm import tqdm

# Đường dẫn dữ liệu
DATASET_DIR = "data/your_dataset"
TRAINING_DIR = "data/your_training_dataset"
WAVS_DIR = os.path.join(TRAINING_DIR, "wavs")
METADATA_PATH = os.path.join(TRAINING_DIR, "metadata.csv")
VOCAB_PATH = os.path.join(TRAINING_DIR, "vocab_your_dataset.txt")

# Tạo thư mục đích nếu chưa tồn tại
os.makedirs(WAVS_DIR, exist_ok=True)

def get_audio_duration(wav_path: str) -> float:
    audio_data, sr = sf.read(wav_path)
    return len(audio_data) / sr

def extract_text_from_textgrid(textgrid_path):
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_path)
        tier = tg[0]  # Lấy layer đầu tiên
        text = " ".join([interval.mark.strip() for interval in tier if interval.mark.strip() != ""])
        return text
    except Exception as e:
        print(f"[Lỗi đọc TextGrid] {textgrid_path}: {e}")
        return None

def process_dataset():
    wav_paths = glob.glob(os.path.join(DATASET_DIR, "*.wav"))
    wav_paths = sorted(wav_paths)[:5000]
    tokens = set()

    with open(METADATA_PATH, "w", encoding="utf8") as fw:
        fw.write("audio_id|text\n")  # Header

        for wav_path in tqdm(wav_paths, desc="Processing dataset"):
            wav_name = os.path.basename(wav_path)
            wav_id = os.path.splitext(wav_name)[0]
            textgrid_path = os.path.join(DATASET_DIR, f"{wav_id}.TextGrid")

            # Bỏ qua nếu không có file TextGrid tương ứng
            if not os.path.exists(textgrid_path):
                continue

            # Trích xuất văn bản từ TextGrid
            text = extract_text_from_textgrid(textgrid_path)
            if not text:
                continue

            text = text.lower().replace("_", " ")
            text = " ".join(text.split())

            duration = get_audio_duration(wav_path)
            if duration < 1 or duration > 30 or len(text.split()) < 3:
                continue

            # Copy file wav vào thư mục chuẩn
            wav_dest_path = os.path.join(WAVS_DIR, wav_name)
            shutil.copy(wav_path, wav_dest_path)

            fw.write(f"wavs/{wav_name}|{text}\n")

            tokens.update(text)

    # Ghi vocab
    with open(VOCAB_PATH, "w", encoding="utf8") as fw_vocab:
        fw_vocab.write("\n".join(sorted(tokens)))

    print(f"✅ Metadata lưu tại: {METADATA_PATH}")
    print(f"✅ Vocab lưu tại: {VOCAB_PATH}")

if __name__ == "__main__":
    process_dataset()
