CUDA_VISIBLE_DEVICES=MIG-56c6e426-3d07-52cb-aa59-73892edacb69 python -m RCQ.Lime_Word_RCQ_conformertcm \
  --protocol /home/woongjae/AudioXplain/porotocol/conformer_ps_spoof.txt \
  --audio_root /home/woongjae/AudioXplain/Datasets/PartialSpoof/database/eval/con_wav \
  --model_path /home/woongjae/AudioXplain/tcm_add/avg_5_best.pth \
  --num_samples 200 \
  --batch_size 128 \
  --log_path /home/woongjae/AudioXplain/RCQ/lime_rcq_conformertcm.log \
  --vad_dir /home/woongjae/AudioXplain/Datasets/PartialSpoof/database/vad_20sil/eval \
  --save_csv /home/woongjae/AudioXplain/result/lime_rcq_conformertcm_results.csv \
  --save_json /home/woongjae/AudioXplain/result/lime_rcq_conformertcm_results.json