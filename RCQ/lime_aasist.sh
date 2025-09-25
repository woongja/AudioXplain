CUDA_VISIBLE_DEVICES=MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd python Lime_Word_RCQ_aasist.py \
  --protocol /home/woongjae/AudioXplain/result/aasist_PS.txt \
  --audio_root /home/woongjae/AudioXplain/Datasets/wavs \
  --model_path /home/woongjae/XAI/SSL_aasist/Best_LA_model_for_DF.pth \
  --num_samples 200 \
  --batch_size 128 \
  --log_path /home/woongjae/AudioXplain/RCQ/lime_rcq.log