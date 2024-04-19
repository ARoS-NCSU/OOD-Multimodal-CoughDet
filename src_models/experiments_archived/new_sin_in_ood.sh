python new_sin_in_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --split_prop 0.7 --model_path "./saved_models/single_in_1.pt" --experiment_name new_single_in_1_ood

python new_sin_in_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --split_prop 0.6 --model_path "./saved_models/single_in_2.pt" --experiment_name new_single_in_2_ood

python new_sin_in_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --split_prop 0.5 --model_path "./saved_models/single_in_3.pt" --experiment_name new_single_in_3_ood

python new_sin_in_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --split_prop 0.3 --model_path "./saved_models/single_in_4.pt" --experiment_name new_single_in_4_ood

python new_sin_in_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --split_prop 0.2 --model_path "./saved_models/single_in_5.pt" --experiment_name new_single_in_5_ood

python new_sin_in_ood.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --split_prop 0.1 --model_path "./saved_models/single_in_6.pt" --experiment_name new_single_in_6_ood