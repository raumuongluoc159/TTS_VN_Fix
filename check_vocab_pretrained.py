"""
Kiểm tra vocab trước khi pretraining hoặc fine-tuning mô hình lớn (LLM hoặc Speech).
Mục tiêu: Đảm bảo bộ vocab bao phủ đầy đủ token của Tiếng Việt.
"""

import os

# Định nghĩa đường dẫn file vocab
PRETRAINED_VOCAB_PATH = "data/Emilia_ZH_EN_pinyin/vocab.txt"
DATASET_VOCAB_PATH = "data/your_training_dataset/vocab_your_dataset.txt"
OUTPUT_VOCAB_PATH = "data/your_training_dataset/vocab.txt"


def load_vocab(file_path: str) -> list:
    """
    Đọc danh sách token từ file vocab.

    Args:
        file_path (str): Đường dẫn đến file vocab.

    Returns:
        list: danh sách các token trong file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File không tồn tại: {file_path}")

    with open(file_path, "r", encoding="utf8") as file:
        return [line.replace("\n", "") for line in file]


def save_vocab(file_path: str, vocab: list):
    """
    Lưu danh sách token vào file vocab.

    Args:
        file_path (str): Đường dẫn file đầu ra.
        vocab (list): Danh sách token cần lưu.
    """
    with open(file_path, "w", encoding="utf8") as file:
        file.writelines(f"{token}\n" for token in vocab)


def process_vocab():
    """
    Kiểm tra và mở rộng vocab nếu cần thiết.
    """
    # Load vocab từ file
    tokens_pretrained = load_vocab(PRETRAINED_VOCAB_PATH)
    tokens_your_dataset = load_vocab(DATASET_VOCAB_PATH)

    # Tìm token trong dataset nhưng không có trong pretrained
    tokens_missing = []

    for token in tokens_your_dataset:
        if token not in tokens_pretrained:
            tokens_missing.append(token)

    print(f"Số token thiếu trong vocab pretrained: {len(tokens_missing)}")

    # Tạo vocab mới và lưu lại
    new_vocab = tokens_pretrained + list(tokens_missing)
    save_vocab(OUTPUT_VOCAB_PATH, new_vocab)

    print(f"Vocab mới đã được lưu tại {OUTPUT_VOCAB_PATH}, tổng số token: {len(new_vocab)}")


if __name__ == "__main__":
    process_vocab()