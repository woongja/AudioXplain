CUDA_VISIBLE_DEVICES=MIG-ad433dcf-e7b9-5a99-a0fa-6fdf3033b7cd python Lime_Word_RCQ_aasist.py \
  --protocol /home/woongjae/AudioXplain/porotocol/aasist_ps_spoof.txt \
  --audio_root /home/woongjae/AudioXplain/Datasets/PartialSpoof/database/eval/con_wav \
  --model_path /home/woongjae/AudioXplain/SSL_aasist/Best_LA_model_for_DF.pth \
  --num_samples 200 \
  --batch_size 64 \
  --log_path /home/woongjae/AudioXplain/RCQ/lime_rcq_aasist.log \
  --vad_dir /home/woongjae/AudioXplain/Datasets/PartialSpoof/database/vad_20sil/eval \
  --save_csv /home/woongjae/AudioXplain/result/lime_rcq_aasist_results.csv \
  --save_json /home/woongjae/AudioXplain/result/lime_rcq_aasist_results.json