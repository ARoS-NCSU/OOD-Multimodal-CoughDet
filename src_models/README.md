# Robust Multi Modal Cough Detection

This repository is adapted from 
* [Efficient Large-Scale Audio Tagging Via Transformer-To-CNN Knowledge Distillation](https://github.com/fschmid56/EfficientAT)
* [ViM: Out-Of-Distribution with Virtual-logit Matching](https://github.com/haoqiwang/vim)


### To run training:
Run `train_test.py` to train the model. 
Besides training parameters, `--setting` indicates in-subject (in) or cross-subject (cross) settings. `--modality` indicates singal-modal (single) or multimodal (multi) models. To run in-subject setting, `--split_prop` need to be setted. To run cross-subject setting, `--start_sub` or `--end_sub` need to be setted.

For example, we want to train 6 multimodal models under in-subject setting:
```
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00005 --n_epochs 3 --mixup_alpha 0 --num_class 3 --split_prop 0.7 --experiment_name multi_in_1 --setting in --modality multi
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00005 --n_epochs 3 --mixup_alpha 0 --num_class 3 --split_prop 0.6 --experiment_name multi_in_2 --setting in --modality multi
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00005 --n_epochs 3 --mixup_alpha 0 --num_class 3 --split_prop 0.4 --experiment_name multi_in_3 --setting in --modality multi
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00005 --n_epochs 3 --mixup_alpha 0 --num_class 3 --split_prop 0.3 --experiment_name multi_in_4 --setting in --modality multi
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00005 --n_epochs 3 --mixup_alpha 0 --num_class 3 --split_prop 0.2 --experiment_name multi_in_5 --setting in --modality multi
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00005 --n_epochs 3 --mixup_alpha 0 --num_class 3 --split_prop 0.1 --experiment_name multi_in_6 --setting in --modality multi
```
To run 6 experiments for multimodal models under cross-subject setting:
```
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00004 --n_epochs 3 --mixup_alpha 0 --num_class 3 --start_sub 27 --experiment_name multi_cross_1 --setting cross --modality multi
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00004 --n_epochs 3 --mixup_alpha 0 --num_class 3 --end_sub 9 --experiment_name multi_cross_2 --setting cross --modality multi
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00004 --n_epochs 3 --mixup_alpha 0 --num_class 3 --start_sub 9 --end_sub 18 --experiment_name multi_cross_3 --setting cross --modality multi
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00004 --n_epochs 3 --mixup_alpha 0 --num_class 3 --start_sub 18 --end_sub 27 --experiment_name multi_cross_4 --setting cross --modality multi
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00004 --n_epochs 3 --mixup_alpha 0 --num_class 3 --start_sub 24 --end_sub 33 --experiment_name multi_cross_5 --setting cross --modality multi
python train_test.py --cuda --pretrained --model_name=mn10_as --batch_size=32 --lr=0.00004 --n_epochs 3 --mixup_alpha 0 --num_class 3 --start_sub 6 --end_sub 15 --experiment_name multi_cross_6 --setting cross --modality multi
```


### To run OOD detection task:
Set up model paths for different types of model.

For example, this command tests the OOD performance for multimodal models under cross-subject setting.
```
python ood_detection.py --cuda --pretrained --model_name=mn10_as --n_epochs 30 --num_class 3 --mixup_alpha 0 --start_sub 6 --end_sub 15 --model_path "./saved_models/multi_cross_6.pt" --imu_model_path "./saved_models/multi_cross_6_imu.pt" --experiment_name new_multi_cross_6_ood --modality "multi" --setting "cross"
````
