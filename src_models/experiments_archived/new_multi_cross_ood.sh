python new_multi_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 27 --model_path "./saved_models/multi_cross_1.pt" --imu_model_path "./saved_models/multi_cross_1_imu.pt" --experiment_name new_multi_cross_1_ood

python new_multi_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --end_sub 9 --model_path "./saved_models/multi_cross_2.pt" --imu_model_path "./saved_models/multi_cross_2_imu.pt" --experiment_name new_multi_cross_2_ood

python new_multi_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 9 --end_sub 18 --model_path "./saved_models/multi_cross_3.pt" --imu_model_path "./saved_models/multi_cross_3_imu.pt" --experiment_name new_multi_cross_3_ood

python new_multi_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 18 --end_sub 27 --model_path "./saved_models/multi_cross_4.pt" --imu_model_path "./saved_models/multi_cross_4_imu.pt" --experiment_name new_multi_cross_4_ood

python new_multi_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 24 --end_sub 33 --model_path "./saved_models/multi_cross_5.pt" --imu_model_path "./saved_models/multi_cross_5_imu.pt" --experiment_name new_multi_cross_5_ood

python new_multi_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 6 --end_sub 15 --model_path "./saved_models/multi_cross_6.pt" --imu_model_path "./saved_models/multi_cross_6_imu.pt" --experiment_name new_multi_cross_6_ood

python new_multi_cross_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 6 --end_sub 15 --model_path "./saved_models/multi_cross_6.pt" --imu_model_path "./saved_models/multi_cross_6_imu.pt" --experiment_name new_multi_cross_6_ood --modality "multi" --setting "cross"