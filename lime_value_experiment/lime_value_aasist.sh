CUDA_VISIBLE_DEVICES=1 python lime_value_aasist.py \
  --protocol /home/woongjae/XAI/protocol/aasist_ps_spoof.txt \
  --audio_root /home/woongjae/XAI/Datasets/PartialSpoof/database/eval/con_wav \
  --model_path /home/woongjae/XAI/SSL_aasist/Best_LA_model_for_DF.pth \
  --num_samples 200 \
  --batch_size 64 \
  --log_path /home/woongjae/XAI/RCQ/lime_rcq_aasist.log \
  --vad_dir /home/woongjae/XAI/Datasets/PartialSpoof/database/vad_20sil/eval \
  --save_csv /home/woongjae/XAI/result/lime_value_aasist_results.csv \
  --save_json /home/woongjae/XAI/result/lime_value_aasist_results.json