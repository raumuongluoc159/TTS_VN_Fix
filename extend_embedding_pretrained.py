"""
Mô-đun mở rộng embedding của mô hình bằng cách thêm token mới vào vocab.
Áp dụng khi fine-tuning mô hình F5-TTS.
"""

import os
import random
import torch
from cached_path import cached_path
from safetensors.torch import load_file

# Định nghĩa seed để đảm bảo tái lập kết quả
SEED = 666

def set_random_seed(seed: int):
    """ Đặt seed cho các thư viện ngẫu nhiên để đảm bảo reproducibility. """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_vocab(file_path: str) -> list:
    """ Đọc danh sách token từ file vocab. """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with open(file_path, "r", encoding="utf8") as file:
        return [line.strip() for line in file.readlines()]


def expand_model_embeddings(ckpt_path: str, new_ckpt_path: str, num_new_tokens: int = 42):
    """
    Mở rộng embedding của mô hình bằng cách thêm token mới.

    Args:
        ckpt_path (str): Đường dẫn đến file checkpoint gốc.
        new_ckpt_path (str): Đường dẫn để lưu checkpoint đã mở rộng.
        num_new_tokens (int): Số lượng token mới cần thêm vào.
    """
    if ckpt_path.endswith(".safetensors"):
        ckpt = load_file(ckpt_path, device="cpu")
        ckpt = {"ema_model_state_dict": ckpt}
    elif ckpt_path.endswith(".pt"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
    else:
        raise ValueError("Định dạng checkpoint không được hỗ trợ. Chỉ hỗ trợ .safetensors hoặc .pt")

    ema_sd = ckpt.get("ema_model_state_dict", {})
    embed_key_ema = "ema_model.transformer.text_embed.text_embed.weight"

    if embed_key_ema not in ema_sd:
        raise KeyError(f"Không tìm thấy khóa {embed_key_ema} trong checkpoint.")

    old_embed_ema = ema_sd[embed_key_ema]
    vocab_old, embed_dim = old_embed_ema.shape
    vocab_new = vocab_old + num_new_tokens

    def expand_embeddings(old_embeddings: torch.Tensor) -> torch.Tensor:
        """ Mở rộng embeddings bằng cách thêm vector mới. """
        new_embeddings = torch.zeros((vocab_new, embed_dim))
        new_embeddings[:vocab_old] = old_embeddings
        new_embeddings[vocab_old:] = torch.randn((num_new_tokens, embed_dim))
        return new_embeddings

    ema_sd[embed_key_ema] = expand_embeddings(ema_sd[embed_key_ema])
    torch.save(ckpt, new_ckpt_path)


if __name__ == "__main__":
    # Thiết lập seed ngẫu nhiên
    set_random_seed(SEED)

    # Đường dẫn file vocab
    TOKEN_PRETRAINED_PATH = "data/Emilia_ZH_EN_pinyin/vocab.txt"
    TOKEN_NEW_PATH = "data/your_training_dataset/vocab.txt"

    # Load vocab
    tokens_pretrained = load_vocab(TOKEN_PRETRAINED_PATH)
    tokens_new = load_vocab(TOKEN_NEW_PATH)

    # Số lượng token mới cần thêm
    vocab_size_new = len(tokens_new) - len(tokens_pretrained)

    # Đường dẫn checkpoint
    ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
    new_ckpt_path = "ckpts/your_training_dataset/pretrained_model_1200000.pt"

    # Mở rộng embedding
    expand_model_embeddings(ckpt_path, new_ckpt_path, num_new_tokens=vocab_size_new)

    print(f"Checkpoint đã được mở rộng và lưu tại: {new_ckpt_path}")