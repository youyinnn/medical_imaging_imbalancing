## Cases

1. ISIC
2. MedMNIST
3. X-Ray Chest

## Exp Design

0. Select the data augmentation methods as the balancing;

1. Check label distribution;
2. Train models without balancing;
3. Observe the XAI result;
4. Train models with balancing;

## Objective

1. Use XAI to detect whether a model is trained with an imbalanced dataset and:
   1. Which class is imbalanced;
   2. How much is the imbalance;

## Scripts

```bash
python main.py fit --config config/pytorch_resnet50_base.yaml --config config/pytorch_resnet50_32_128.yaml
python main.py fit --config config/pytorch_resnet50_base.yaml --config config/pytorch_resnet50_224_128.yaml
python main.py fit --config config/timm_mixer_in21k_base.yaml --config config/timm_mixer_in21k_224_128.yaml
```
