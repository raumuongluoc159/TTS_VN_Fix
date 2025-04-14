"""
Mô-đun chuẩn bị metadata cho tập dữ liệu huấn luyện.
Tạo file metadata.csv và vocab từ tập dữ liệu âm thanh.
"""

import os
import glob
import shutil
import soundfile as sf
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
    """
    Tính thời lượng của file audio.

    Args:
        wav_path (str): Đường dẫn file WAV.

    Returns:
        float: Thời lượng của file (giây).
    """
    audio_data, sr = sf.read(wav_path)
    return len(audio_data) / sr


def process_dataset():
    """
    Duyệt qua tất cả file WAV, copy vào thư mục mới, tạo metadata và vocab.
    """
    wav_paths = glob.glob(os.path.join(DATASET_DIR, "*.wav"))
    tokens = set()

    with open(METADATA_PATH, "w", encoding="utf8") as fw:
        for wav_path in tqdm(wav_paths, desc="Processing dataset"):
            wav_name = os.path.basename(wav_path)
            wav_dest_path = os.path.join(WAVS_DIR, wav_name)

            # Copy file âm thanh sang thư mục mới
            shutil.copy(wav_path, wav_dest_path)

            # Đọc nội dung text
            txt_path = wav_path.replace(".wav", ".txt")
            if not os.path.exists(txt_path):
                continue

            with open(txt_path, "r", encoding="utf8") as fr:
                text = fr.readline().strip().lower()
                text = text.replace("_", " ")
                text = " ".join(text.split())

            # Bỏ qua file không đạt yêu cầu
            duration = get_audio_duration(wav_path)
            if duration < 1 or duration > 30 or len(text.split()) < 3:
                continue

            # Ghi vào metadata.csv
            fw.write(f"wavs/{wav_name}|{text}\n")

            # Thu thập token cho vocab
            tokens.update(text)

    # Ghi vocab vào file
    with open(VOCAB_PATH, "w", encoding="utf8") as fw_vocab:
        fw_vocab.write("\n".join(sorted(tokens)))

    print(f"Metadata lưu tại: {METADATA_PATH}")
    print(f"Vocab lưu tại: {VOCAB_PATH}")


if __name__ == "__main__":
    process_dataset()
