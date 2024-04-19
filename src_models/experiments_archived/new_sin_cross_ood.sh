python new_sin_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 27 --model_path "./saved_models/single_cross_1.pt" --experiment_name new_single_cross_1_ood

python new_sin_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --end_sub 9 --model_path "./saved_models/single_cross_2.pt" --experiment_name new_single_cross_2_ood

python new_sin_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 9 --end_sub 18 --model_path "./saved_models/single_cross_3.pt" --experiment_name new_single_cross_3_ood

python new_sin_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 18 --end_sub 27 --model_path "./saved_models/single_cross_4.pt" --experiment_name new_single_cross_4_ood

python new_sin_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 24 --end_sub 33 --model_path "./saved_models/single_cross_5.pt" --experiment_name new_single_cross_5_ood

python new_sin_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 6 --end_sub 15 --model_path "./saved_models/single_cross_6.pt" --experiment_name new_single_cross_6_ood

# python new_sin_cross.py --cuda --pretrained --model_name=mn10_im --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 27 --experiment_name new_single_cross_1_im

# python new_sin_cross.py --cuda --pretrained --model_name=mn10_im --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --end_sub 9 --experiment_name new_single_cross_2_im

# python new_sin_cross.py --cuda --pretrained --model_name=mn10_im --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 9 --end_sub 18 --experiment_name new_single_cross_3_im

# python new_sin_cross.py --cuda --pretrained --model_name=mn10_im --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 18 --end_sub 27 --experiment_name new_single_cross_4_im

# python new_sin_cross.py --cuda --pretrained --model_name=mn10_im --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 24 --end_sub 33 --experiment_name new_single_cross_5_im

# python new_sin_cross.py --cuda --pretrained --model_name=mn10_im --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 6 --end_sub 15 --experiment_name new_single_cross_6_im

# python new_sin_cross.py --cuda --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 27 --experiment_name new_single_cross_1_raw

# python new_sin_cross.py --cuda --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --end_sub 9 --experiment_name new_single_cross_2_raw

# python new_sin_cross.py --cuda --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 9 --end_sub 18 --experiment_name new_single_cross_3_raw

# python new_sin_cross.py --cuda --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 18 --end_sub 27 --experiment_name new_single_cross_4_raw

# python new_sin_cross.py --cuda --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 24 --end_sub 33 --experiment_name new_single_cross_5_raw

# python new_sin_cross.py --cuda  --batch_size=32 --lr=0.00005 --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 6 --end_sub 15 --experiment_name new_single_cross_6_raw