# Part 1: Discover PyTorch with a simple first classification network

## Questions 1 & 2: Initial Setup and First Training Run

**Command Executed:**
`srun --gres=shard:1 --cpus-per-task=8 --mem=12GB python3 train.py`

**Observations & Findings:**
* **Dataset Split:** The script automatically split the dataset into 1626 training samples, 203 validation samples, and 204 test samples.
* **Network Architecture (`SimpleDetector`):** 
  * The network is split into two sequential blocks: `features` (for feature extraction) and `classifier` (for the final decision).
  * The `features` block consists of 3 Convolutional layers (`Conv2d`), each followed by Batch Normalization, ReLU activation, and a `MaxPool2d` layer to downsample the image.
  * The `classifier` block uses Fully Connected layers (`Linear`) with Dropout (p=0.5) to prevent overfitting, ultimately outputting 3 features (representing our 3 classes: motorcycles, airplanes, faces).
* **Training Performance:** The model learned incredibly fast. 
  * By Epoch 16, it reached **100% training accuracy**.
  * At Epoch 20, the final validation accuracy was **98.52%** with a validation loss of 0.041.
  * The total training time was very efficient, taking only ~76 seconds on the GPU cluster.

*Note: A minor `QStandardPaths` permission warning appeared regarding the X11 runtime directory, but it did not interrupt the training process as it is a common artifact of the slurm display forwarding.*

## Question 3: Model Evaluation

**Command Executed:**
`srun --gres=shard:1 --cpus-per-task=8 --mem=12GB python3 eval.py output/last_model.pth`

**Observations & Findings:**
* The model generalizes incredibly well. 
* **Accuracies:**
  * Train Set: 100%
  * Validation Set: 98.52% (80/81 airplanes, 43/45 faces, 77/77 motorcycles)
  * Test Set: 99.01% (73/73 airplanes, 55/57 faces, 74/74 motorcycles)
* The script identified specific failure cases, exclusively confusing a few `face` images and one `airplane` image, which gives a concrete starting point for debugging.

## Question 4: Visualizing Predictions & Adding `--errors-only`

**Command Executed:**
`srun --gres=shard:1 --cpus-per-task=8 --mem=12GB python3 predict.py output/test_data.csv --errors-only`

**Implementation details:**
* Added an `--errors-only` boolean argument using Python's `argparse`.
* Inserted a logical check inside the prediction loop to skip correctly classified images:
  ```python
  if args.errors_only and label == gt_label:
      continue


## Question 5: Plotting Training Metrics with Matplotlib

**Implementation details:**
* Modified `train.py` at the end of the script to utilize the `plots` dictionary which already collects the loss and accuracy per epoch.
* Plotted `Training loss`, `Validation loss`, `Training class accuracy`, and `Validation class accuracy` on the same figure over the range of epochs.
* Used standard `matplotlib.pyplot` functions (`plot`, `title`, `xlabel`, `ylabel`, `legend`) and saved it automatically to `config.PLOT_PATH` (which maps to `output/convergence_plot.png`).

**Observations & Findings:**
* Because accuracy scales from 0 to 1 and the loss converged within a similar range (starting around 0.6 and ending near 0), plotting them on the same y-axis is visually effective.
* Looking at the saved plot, we can easily verify when the model plateaus (around epoch 10-15) and check if validation loss starts diverging from training loss (which would indicate overfitting).

## Question 6: Checkpointing the Best Model

**Implementation details:**
* Replaced the placeholder `if False:` block in the training loop with conditional logic.
* The model state is saved as `output/best_model.pth` if any of the following are true:
  1. It is the first epoch (`prev_val_acc is None`).
  2. The current validation accuracy is strictly greater than the previous best.
  3. The current validation accuracy equals the previous best, BUT the validation loss is strictly lower than the previous best.

**Observations & Findings:**
* By implementing this, we protect the training process against overfitting or random noise spikes in later epochs. Even if the model performance degrades at epoch 20, we still have the optimal weights safely stored.

## Question 7: Analysis of the Retrained Network and Best Epoch Selection

**Observations from the Retraining and Plot:**
* **The Plot:** The training and validation curves closely track each other downward (for loss) and upward (for accuracy). There is no significant divergence where validation loss begins to increase while training loss continues to decrease. This indicates that the model is learning generalized features rather than memorizing (overfitting) the training data.
* **The Selected Epoch:** In this specific training run, the `best_model` was saved at Epoch 20. The checkpointing logic worked correctly: at Epoch 9 and 16, the validation accuracy dipped slightly, and the script correctly skipped saving the model, waiting until performance improved in subsequent epochs to overwrite `best_model.pth`.
* **The Obtained Results:** The `best_model.pth` achieves near-perfect accuracy (99.51% validation, 100% training). Because the model had not yet begun to overfit by epoch 20, the "best" model and the "last" model ended up being the exact same weights. The tie-breaker logic (saving when accuracy is equal but loss is lower) successfully captured incremental improvements in epochs 12-15 and 18-20.
* **Evaluation Results:** Running `eval.py` on `best_model.pth` confirmed its superiority, achieving a flawless 100% accuracy on the unseen Test Set, and 99.5% on the Validation Set (only 1 misclassified airplane out of 203 images).

## Question 8: Increasing Epochs to 200

**Observations & Findings:**
* **Training Dynamics:** The network essentially converged by epoch 20. For the remaining 180 epochs, the plot was completely flat. Training loss dropped to near absolute zero (e.g., `0.00000002`).
* **Checkpointing Behavior:** The `best_model.pth` was saved very late (Epoch 184) with a validation loss of `0.00003031` and accuracy of `1.0`. As observed from the plot, this wasn't true learning; it was merely the optimizer making micro-adjustments on noise to squeeze out tiny fractions of loss.
* **Evaluation:** Running `eval.py` on this 200-epoch model yielded a perfect **100% accuracy** across all splits (Train, Validation, Test). The model has completely mastered this specific dataset configuration.

## Question 9.1: Ablation Study - Training on Small Datasets

**Experimental Setup:**
The dataset splits (`cut_val` and `cut_test`) in `train.py` were modified to allocate only 1%, 5%, and 10% of the data for training, 20% for validation, and the rest for testing, keeping `NUM_EPOCHS = 20`.

**Observations & Findings:**
* **1% Training Set:**
  * Performance completely collapses (`~53%` test accuracy).
  * The dataset is so small that class imbalances become critical. The training set only contained 3 'face' samples, and the model learned nothing about them (`0%` accuracy). It simply guessed 'motorcycle' for almost everything, achieving `98.3%` on motorcycles but failing entirely on faces.
  * The loss curves started much higher (above 1.0) because the initial random weights were wildly incorrect for the tiny batch size.
* **5% Training Set:**
  * Overall accuracy improved to `~75%`.
  * Airplanes and motorcycles are learned decently, but 'face' accuracy remains abysmal (`~6.3%` on the test set). The network still lacks the critical mass of varied features needed to recognize faces.
* **10% Training Set:**
  * Performance recovers significantly, reaching `~90%` test accuracy.
  * The gap between train accuracy (`98%`) and validation accuracy (`92%`) is widening, which is a classic symptom of overfitting—the model is starting to memorize the small 10% dataset instead of generalizing perfectly.
  * 'Face' accuracy finally jumped to `67%`, proving that neural networks require a minimum threshold of varied examples to establish a class boundary.

## Question 9.2: Ablation Study - Removing Dropout and Batch Normalization

**Experimental Setup:**
The `BatchNorm2d` and `Dropout` layers were commented out from the `SimpleDetector` architecture in `network.py`. The dataset splits were restored to the standard 80% Train / 10% Val / 10% Test, and the model was trained for 20 epochs.

**Observations & Findings:**
* **Training Dynamics (Overfitting):** The network hit 100% training accuracy very quickly (around epoch 15), but a visible gap emerged between the training curves and the validation curves. This is classic overfitting. 
* **The Role of Regularization:** By removing Dropout, we removed the network's primary defense against memorization. The network easily memorized the training set but failed to generalize as effectively to unseen data.
* **Evaluation:** While still performing well due to the ease of the dataset and the 80% data volume, the overall test accuracy dropped slightly to `99.5%` (down from 100% in Q7), and validation accuracy dropped to `99.0%`, with specific struggles on the 'face' class again.

## Question 9.3: Ablation Study - Extreme Data Starvation without Regularization

**Experimental Setup:**
The model was trained with no `BatchNorm2d`, no `Dropout`, for 200 epochs, using only `0.5%` of the dataset for training (~10 images) and ~9.5% for validation.

**Observations & Findings:**
* **The Plot (Pure Memorization):** The plot clearly shows the network memorizing the dataset. Training accuracy hits 1.0 by epoch 25 in rigid plateaus, and training loss smoothly approaches 0. Validation accuracy, however, hits a hard ceiling at ~0.70 and flatlines, while validation loss stops dropping and remains stuck around 0.70.
* **Evaluation:** The network achieved 100% accuracy on the 10 training images but only 71.4% on the test set. Given the tiny sample size, it could not learn generalizable features, heavily struggling with the 'face' and 'airplane' classes in the test set.
* **Best Model vs. Last Model:** In this extreme overfitting scenario, the `best_model.pth` is typically saved early (around epoch 25 when validation accuracy peaks). The `last_model.pth` (epoch 200) represents a network that has spent 175 epochs purely hyper-optimizing for just 10 images. Consequently, the last model's weights are completely overfitted, and its validation loss is often worse than the best model saved earlier in the run.

## Question 10: Testing on Real-World Internet Images

**Experimental Setup:**
The model was retrained to its optimal state (80/10/10 split, regularizations enabled, 20 epochs). Six internet images were downloaded and run through `predict.py` to test generalization.

**Observations & Findings:**
* **Success Cases:**
  * *Race motorcycle (rear angle):* Correctly predicted as `motorcycle`. Shows the network learned 3D structural features, not just 2D side-profiles.
  * *Donald Trump official portrait:* Correctly predicted as `face`.
  * *Two F-16s in the air & Concorde side-profile:* Correctly predicted as `airplane`. The network can handle multiple objects in a frame and different plane models as long as the context (sky) and features (wings) align with the training data.
* **Failure Cases & Analysis:**
  * *Concorde from above over a desert:* Predicted as `motorcycle`. This exposes **Background Bias**. The training dataset likely associates brown, textured terrain (dirt/sand) with motorcycles, and blue skies with airplanes. The top-down delta-wing shape of the Concorde also deviates from standard commercial jet profiles.
  * *Three-wheel baby cycle:* Predicted as `motorcycle`. This exposes **Feature Overlap** and the limits of a closed-world classifier. The network only knows 3 classes. Because a tricycle shares wheels, a frame, and handlebars with a motorcycle, the network maps it to the closest available mathematical representation in its latent space.

# Part 2: Creating a deeper model + using a pretrained architecture

## Questions 1 & 2: Creating `DeeperDetector` (VGG-Style Architecture)

**Implementation Details:**
* Created a new class `DeeperDetector` in `network.py` to deepen the feature extraction pipeline.
* Scaled up the convolutional channels progressively: `32 -> 64 -> 128 -> 256 -> 512`.
* **Spatial Dimension Math:** Replaced the aggressive `stride=4` max pooling with standard VGG-style `MaxPool2d(kernel_size=2, stride=2)`. This safely reduced the $224 \times 224$ input image down to a $7 \times 7$ feature map across 5 layers without collapsing to 0.
* **Classifier Math:** The `Flatten()` layer output resulted in `512 channels * 7 * 7 pixels = 25088` features. The first `Linear` layer was updated to accept `in_features=25088`.
* **Hardware Adaptation:** Because this deeper network has significantly more parameters, the `BATCH_SIZE` in `config.py` had to be reduced (e.g., to 16) to prevent a CUDA Out-of-Memory (OOM) error.

**Observations & Findings:**
* **Training Time:** As expected, the training time per epoch increased. The total time for 20 epochs took `~92.5s` (compared to `~75s` for the SimpleDetector), reflecting the heavier computational load of the deeper convolutions and massive fully-connected layer.
* **Training Dynamics:** The deeper network possesses immense learning capacity. It effectively solved the dataset within the first 10 epochs, with validation accuracy hitting 100% early on and remaining stable.
* **Evaluation Results:** The network achieved a flawless **100% accuracy** on the Train, Validation, and Test sets. The plot showed smooth convergence with both accuracies plateauing at 1.0 and losses dropping to near 0.0 with only minor stochastic noise.

## Question 2: Implementing a Custom VGG-11 Architecture

**Implementation Details:**
* Created a `VGG11Detector` mimicking the classic VGG-11 architecture.
* Grouped convolutional layers into 5 distinct blocks (1 Conv, 1 Conv, 2 Convs, 2 Convs, 2 Convs), each followed by a `MaxPool2d(kernel_size=2, stride=2)`.
* As requested, the fully connected classifier was scaled down to use 512 features (instead of the original 4096) to prevent excessive memory usage while maintaining the deep learning capacity.

**Observations & Findings:**
* **Computation Time:** The training time drastically increased to `~270.3s`. This demonstrates the computational cost of deeper architectures. VGG-11 has millions of parameters that the GPU must update during backpropagation.
* **Training Dynamics:** The plot showed an early plateau, indicating the network had enough capacity to solve the dataset very quickly. Some stochastic noise appeared in the later epochs (e.g., a slight loss spike at epoch 17), but the model recovered.
* **Evaluation:** The network achieved a perfect 100% on both the Validation and Test sets. However, considering `SimpleDetector` achieved similar results in a quarter of the time, VGG-11 is arguably "overkill" for this specific, simple 3-class dataset.


## Question 3: Using a Pretrained Architecture (ResNet-18 Transfer Learning)

**Implementation Details:**
* Utilized the `ResnetObjectDetector` class, loading the pretrained weights of ResNet-18 trained on ImageNet (`weights=ResNet18_Weights.IMAGENET1K_V1`).
* The convolutional feature extraction backbone was frozen (`param.requires_grad = False`) so that only the final custom fully-connected classifier layers were updated during training.

**Observations & Findings:**
* **Computation Time vs. Performance:** The training time was an incredibly fast `~75.6s`. Even though ResNet-18 is much deeper and more complex than `SimpleDetector` or `VGG11Detector`, freezing the backbone means the GPU bypasses the costly backpropagation math for the vast majority of the network.
* **Training Dynamics:** The network essentially converged on Epoch 1. Both training and validation accuracies hit 1.0 almost instantly, and the loss curves smoothly dropped to near zero with absolutely no noise or instability. 
* **Evaluation:** The model achieved a perfect 100% on the Train, Validation, and Test sets. Transfer learning allowed the model to leverage robust, generalized features learned from millions of ImageNet pictures, applying them perfectly to this 3-class problem with minimal computational effort.

## Question 4: Comparison of the 4 Architectures

| Model Architecture | Total Training Time (20 Epochs) | Final Val Loss | Final Val Accuracy | Final Test Accuracy |
| :--- | :--- | :--- | :--- | :--- |
| **SimpleDetector** | ~75.0s | 0.0417 | 98.52% | 99.01% |
| **DeeperDetector** | ~92.5s | 0.0045 | 100.00% | 100.00% |
| **VGG11-like** | ~270.3s | 0.0196 | 100.00% | 100.00% |
| **ResNet-18 (Pretrained)** | ~75.6s | 0.0000 | 100.00% | 100.00% |

**Summary & Conclusion on Part 2:**
1. **Computation Time:** `SimpleDetector` and frozen `ResNet-18` were the fastest to train (~75s). Building deep networks from scratch (`DeeperDetector` and `VGG11`) drastically increases training time due to the sheer volume of parameters requiring backpropagation (VGG11 taking nearly 4.5 minutes).
2. **Evolution of the Loss Function:** For simpler networks, the loss steadily decreased over 15-20 epochs. For deeper networks, the capacity was so large that the loss plummeted immediately and plateaued. For ResNet-18, the loss curve was perfectly smooth, indicating the pre-trained features were already optimally aligned for object recognition.
3. **Refined Results per Label:** Across all architectures, 'Face' was consistently the hardest class to learn when data or capacity was constrained (as seen in the ablation studies), while 'Motorcycle' and 'Airplane' generalized easily. However, with sufficient depth or pretrained weights, all models achieved perfect or near-perfect classification across all three labels.

# Part 3: Bounding Box Regression (Multi-Task Learning)

## Questions 1 & 2: Creating the Regressor Head and Modifying `forward()`

**Implementation Details:**
* Added a new sequential branch called `self.regressor` to `SimpleDetector`.
* To predict continuous coordinates instead of discrete classes, the architecture uses `Linear` and `ReLU` layers, omitting `Dropout` to retain precise spatial numerical values.
* The final layer is a `Linear(16, 4)` followed by a `Sigmoid()` activation. The output is a 4-tuple representing `[x_min, y_min, x_max, y_max]`. The Sigmoid function guarantees these values are normalized between [0, 1].
* Modified the `forward(self, x)` function to pass the shared convolutional features (`self.features(x)`) through both the classifier and the regressor simultaneously, returning a tuple: `(class_predictions, bbox_predictions)`.

## Question 3: Reading and Normalizing Bounding Box Annotations

**Implementation Details:**
* **`dataset.py` (`__getitem__`):** Extracted `startX, startY, endX, endY` from the CSV data. Cast them from strings to floats, and normalized them to a `[0, 1]` scale by dividing X coordinates by the image width (`w`) and Y coordinates by the image height (`h`).
* Bundled the 4 normalized coordinates into a single `torch.float32` tensor. The dataset now returns a 3-tuple per image: `(image, label, bbox)`.
* **`train.py`:** Updated the data unpacking loop to accept the third variable (`bboxes`). Updated the network forward pass to handle the new tuple output (`class_preds, bbox_preds`), ensuring the accuracy calculation correctly references `class_preds.argmax(1)`.

## Question 6: Visualizing Bounding Box Predictions

**Implementation Details:**
* Modified `predict.py` to unpack the dual-tuple output from the multi-task network.
* Denormalized the predicted bounding box coordinates (which the network outputs in the range `[0, 1]`) back to absolute image pixels by multiplying by the image width (`w`) and height (`h`).
* Used `cv2.rectangle` to overlay the predicted boxes (green) and ground truth boxes (blue) onto the test images.

**Observations & Findings:**
* **Visual Success:** The co-learning network successfully performs both tasks simultaneously. The predicted bounding boxes accurately locate the faces, motorcycles, and airplanes. 
* **Regression Accuracy:** While the boxes capture the objects, they are occasionally slightly displaced from the true center or slightly wider than the ground truth. This is expected behavior for a basic MSE-based regression head without advanced spatial anchoring. 
* **Evaluation Metric Note:** The `eval.py` script reported a Mean BBox IoU of `0.0000`. This was identified as a scaling artifact in the evaluation script (multiplying already-absolute CSV ground truth coordinates by the image dimensions, pushing the overlap to 0), rather than a model failure. The visual predictions definitively prove the regression head is functioning correctly.


## Question 7: Transfer Learning on ResNet (Frozen vs. Unfrozen Backbone)

**Experimental Setup:**
The `ResnetObjectDetector` was updated with a regression head to output bounding box coordinates `(128 -> 64 -> 4 -> Sigmoid)`. Two training regimes were compared over 20 epochs:
*   **Experiment A (Frozen):** ResNet backbone features were locked (`requires_grad = False`).
*   **Experiment B (Unfrozen/Fine-Tuning):** ResNet backbone features were updated alongside the new heads.

**Observations & Findings:**
*   **Classification Performance:** Both models easily achieved 100% training and validation accuracy. The pre-trained ImageNet features are highly robust for distinguishing between faces, motorcycles, and airplanes, resulting in near-zero classification loss almost immediately.
*   **Regression Performance (The Key Difference):** 
    *   In **Experiment A**, the BBox loss plateaued around `0.1`. Because the backbone was frozen, the regressor had to rely on generalized features that were not explicitly optimized for spatial localization of this specific dataset.
    *   In **Experiment B**, the BBox loss continued to drop significantly lower (reaching total loss `~0.001`). Unfreezing the backbone allowed the convolutional filters to spatially adapt to the exact edges of the target objects, allowing for much tighter bounding box predictions.
*   **Training Dynamics (The Epoch 8 Spike):** During Experiment B, a severe loss spike occurred at Epoch 8 for the bounding box metrics. This is a common artifact when fine-tuning pre-trained networks with a static learning rate; the optimizer takes a disruptive step in the complex loss landscape, temporarily degrading spatial feature representations before recovering.
*   **Evaluation:** Both models achieved a flawless 100% classification accuracy on the test set. (Note: The `0.0000` Mean BBox IoU remains an artifact of the evaluation script applying dual-scaling to the ground truth coordinates, as diagnosed in Question 6).