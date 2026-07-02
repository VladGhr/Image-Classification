# Image Classification: From MLP to Transfer Learning

A PyTorch project that tackles multi-class image classification on two very different datasets, built as a single configurable pipeline. Rather than chasing one model, it works through a progression — a plain multilayer perceptron, a convolutional network built from scratch, dedicated handling for class imbalance, and finally transfer learning with a pretrained ResNet — and measures the effect of each design decision through controlled ablation studies.

## What it does

The notebook trains and evaluates classifiers for two seven-class problems, and you switch between them by changing a single configuration flag:

- **Skin lesion classification** (codenamed *"Vai de pielea mea"*) — dermatoscopic images sorted into seven diagnostic categories (`akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`). This dataset is heavily imbalanced, which motivates a lot of the work later in the project.
- **Facial expression recognition** (codenamed *"You're on candid camera"*) — face images labelled with one of seven emotions: surprise, fear, disgust, happiness, sadness, anger, and neutral.

Everything downstream — image sizes, normalization statistics, class names, augmentation strength, and even the size of the first convolution kernel — is read from a per-dataset config block, so the same code runs on either problem without edits.

## The approach

The project is organized as a sequence of experiments, each building on the last and each accompanied by a small ablation that isolates the impact of one choice.

It starts with a **multilayer perceptron** as a baseline — a fully-connected network that flattens the image and passes it through two hidden layers (1024 and 512 units) with batch normalization and ReLU. Here the ablation asks a simple question: does dropout actually help? The two variants, with and without dropout, are trained under identical conditions and compared.

Next comes a **convolutional network built from scratch**, four convolutional blocks widening from 32 to 256 channels, each followed by pooling, and finished with adaptive average pooling and a small classifier head. This stage runs two separate ablations: one that toggles batch normalization on and off, and one that sweeps four augmentation regimes — none, geometric only, color only, and the full combination — to see how much data augmentation is worth on each dataset.

Because the skin dataset in particular is so imbalanced, a whole section is devoted to **fighting class imbalance**. Four strategies are pitted against each other on the same CNN: a plain cross-entropy baseline, class-weighted cross-entropy, focal loss, and a weighted random sampler that rebalances the batches. Each is scored the same way so the trade-offs are visible side by side.

The final modeling stage is **transfer learning**. A ResNet-18 (with a MobileNetV2 option) pretrained on ImageNet has its backbone frozen and its classification head replaced, then is fine-tuned on the target dataset with its batch-norm layers kept in evaluation mode. The ablation here looks at learning-rate scheduling: a linear warmup followed by cosine annealing versus cosine annealing alone.

## How performance is measured

Every model is evaluated consistently. Alongside accuracy, the pipeline reports macro- and micro-averaged F1 scores and a full per-class classification report, which matters a great deal on the imbalanced skin data where raw accuracy can be misleading. For each run it also renders training-history curves (loss and accuracy over epochs), a confusion matrix, and — where several configurations are compared — a bar chart summarizing the ablation. All of these are saved as images into a per-dataset outputs folder.

To keep comparisons fair, a fixed random seed is set across Python, NumPy, and PyTorch before every experiment, and the class-weighting and normalization statistics are computed directly from the training split.

## How it's organized

The notebook flows top to bottom through numbered sections: setup and configuration, exploring the dataset archive, the dataset and split builders, the augmentation and transform definitions, the shared training and evaluation utilities, then the four modeling stages (MLP, CNN, imbalance handling, and fine-tuning), and finally the generation of a Kaggle submission file. The custom `ImageCSVDataset` class is generic over a list of image paths and targets, and handles both labelled data (for training and local testing) and unlabelled data (for the remote Kaggle test set).

## Running it

The project targets a CUDA-enabled GPU but falls back to CPU automatically. It expects PyTorch and torchvision (the notebook installs the CUDA 12.1 builds), plus NumPy, pandas, Pillow, Matplotlib, and scikit-learn.

To use it, unzip the dataset you want next to the notebook so its folder matches the `root` path in the config (`./vai-de-pielea-mea` for skin, `./you-re-on-candid-camera` for emotions), set the `DATASET` variable at the top to either `"skin"` or `"emotions"`, and run the cells in order. Each dataset is expected to provide CSV split files (train, local test, and a remote test set) that map image files to labels. Trained-model artifacts, plots, and the final `submission.csv` are written to `outputs/<dataset>/`.

## Notes

The two datasets deliberately stress different things: the skin data is a lesson in imbalanced medical classification where the minority classes are the ones that matter most, while the emotions data is a cleaner but subtler recognition task. Running the same pipeline across both is part of the point — it shows which techniques generalize and which are dataset-specific.
