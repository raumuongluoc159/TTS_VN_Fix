import glob
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path
from shutil import move

from tqdm import tqdm


def convert_sr(audio_path: str) -> None:
    """
    Chuyển đổi tần số lấy mẫu của file âm thanh thành 24kHz.
    """
    audio_path = Path(audio_path)
    output_path = audio_path.with_name(f"{audio_path.stem}_24k.wav")
    subprocess.run(
        ["sox", str(audio_path), "-r", "24000", "-c", "1", str(output_path)],
        check=True
    )


def remove_original(audio_path: str) -> None:
    """
    Xóa file gốc nếu nó không phải là file đã được chuyển đổi sang 24kHz.
    """
    if "_24k.wav" not in audio_path:
        os.remove(audio_path)


def rename_audio(audio_path: str) -> None:
    """
    Xóa hậu tố '_24k' khỏi tên file để đặt lại tên cho đúng chuẩn.
    """
    audio_path = Path(audio_path)
    new_path = audio_path.with_name(audio_path.stem.replace("_24k", "") + ".wav")
    move(audio_path, new_path)


def process_audio_files(function, wav_paths):
    """
    Xử lý các file âm thanh với hàm được chỉ định sử dụng đa luồng.
    """
    with Pool(processes=16) as pool:
        list(tqdm(pool.imap(function, wav_paths), total=len(wav_paths)))


if __name__ == "__main__":
    dataset_path = "data/your_dataset/*.wav"

    # Chuyển đổi sample rate
    process_audio_files(convert_sr, glob.glob(dataset_path))

    # Xóa file gốc
    process_audio_files(remove_original, glob.glob(dataset_path))

    # Đổi tên file
    process_audio_files(rename_audio, glob.glob(dataset_path))