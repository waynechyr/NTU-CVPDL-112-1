
python main.py --output_dir output2 -c config/DINO/DINO_4scale.py --coco_path COCODIR  --options dn_scalar=100 embed_init_tgt=TRUE dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False dn_box_noise_scale=1.0 --pretrain_model_path checkpoint0011_4scale.pth --finetune_ignore label_enc.weight class_embed

#checkpoint0011_4scale.pth
