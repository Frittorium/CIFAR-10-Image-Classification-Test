# CIFAR-10 Image Classification: Transfer Learning vs. Custom CNN

## Dataset
CIFAR-10 contains 60,000 32x32 RGB images across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). The 50,000 training images are split into 45,000 train and 5,000 validation. The official 10,000-image test set is held out for final evaluation.

## Objective
Multi-class image classification. The project compares how well a pretrained ResNet18 adapts to CIFAR-10 at different levels of fine-tuning against a small CNN trained from scratch.

## Preprocessing
- Pixel values scaled to [0, 1]
- Per-channel normalization using CIFAR-10 mean and standard deviation
- Images kept at native 32x32 resolution
- No data augmentation was applied

## Model Architecture
The custom CNN has three convolutional blocks, each with two 3x3 convolutions, batch normalization, ReLU, max pooling, and 0.25 dropout. Channel widths are 32, 64, and 128. A 256-unit fully connected layer with 0.5 dropout feeds the 10-class output.

## Design Decisions
- **Transfer learning baselines:** show how much pretrained ImageNet features help on a small-image dataset.
- **Progressive unfreezing:** isolates the effect of fine-tuning deeper layers.
- **Batch norm and dropout in the custom CNN:** stabilize training and reduce overfitting with a compact model.
- **Macro-averaged precision, recall, and F1:** account for all classes equally alongside accuracy.
- **Torchvision for data loading:** reliable download without extra dependencies.

## Experiments
Four models were trained under identical settings (Adam, lr 1e-3, batch size 128, 5 epochs):

| Model | Trainable layers |
|---|---|
| ResNet18 | `fc` only |
| ResNet18 | `layer4` + `fc` |
| ResNet18 | `layer3`, `layer4` + `fc` |
| Custom CNN | All (from scratch) |

Each run tracks train/validation loss and accuracy, plus validation precision, recall, and F1 per epoch, followed by a final test-set evaluation.

## Results

| Model | Trainable params | Accuracy | Precision | Recall | F1 | Test loss |
|---|---|---|---|---|---|---|
| ResNet18 (fc only) | 5,130 | 0.4595 | 0.4624 | 0.4595 | 0.4525 | 1.5737 |
| ResNet18 (layer4 + fc) | 8,398,858 | 0.6962 | 0.7013 | 0.6962 | 0.6960 | 1.4506 |
| ResNet18 (layer3-4 + fc) | 10,498,570 | 0.7889 | 0.7916 | 0.7889 | 0.7893 | 0.9438 |
| Custom CNN | 815,018 | **0.8066** | **0.8157** | **0.8066** | **0.8070** | **0.5569** |

**Findings**
- The custom CNN performed best on every metric, with about 13x fewer trainable parameters than the best ResNet18 variant.
- Accuracy rose steadily as more ResNet18 layers were unfrozen (46% → 70% → 79%). This suggests ImageNet features transfer poorly to 32x32 images without adaptation.
- The custom CNN's test loss (0.56) was well below the ResNet18 variants (0.94 to 1.57), so its predictions were also better calibrated.
- Training time was similar across the unfrozen models (39 to 56 s), so the custom CNN's advantage did not come at a large compute cost.
- All models were trained for only 5 epochs without augmentation, so the ranking may change with longer training.