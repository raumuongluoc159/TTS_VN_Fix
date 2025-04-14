f5-tts_infer-cli \
--model "F5TTS_Base" \
--ref_audio ref.wav \
--ref_text "nội dung cấu hình của bạn, dữ liệu huấn luyện bạn đang sử dụng, phiên bản TTS, Python, Tensor, PyTorch và Cuda" \
--gen_text "Xã hội này, chỉ có cần cù bù siêng năng, chỉ có làm thì mới có ăn, những cái loại không làm mà đòi có ăn thì ăn cục gạch" \
--speed 1.0 \
--vocoder_name vocos \
--vocab_file data/your_training_dataset/vocab.txt \
--ckpt_file ckpts/your_training_dataset/model_last.pt \