from model import config
import sys
import os
import torch
import cv2
import numpy

# load our object detector, set it evaluation mode, and label

if len(sys.argv) < 2:
    print("Please enter the path to the model to be evaluated")
    sys.exit(1)

model_path = sys.argv[1]

print(f"**** loading object detector at {model_path}...")
model = torch.load(model_path, weights_only=False).to(config.DEVICE)
model.eval()
print(f"**** object detector loaded")

results_labels = dict()
results_iou = dict()           # NEW: track bounding box IoU

results_labels['all'] = []
results_iou['all'] = []        # NEW

for label_str in config.LABELS:
    results_labels[label_str] = []
    results_iou[label_str] = [] # NEW

for mode, csv_file in [['train', config.TRAIN_PATH],
                       ['validation', config.VAL_PATH],
                       ['test', config.TEST_PATH],]:
    data = []
    assert(csv_file.endswith('.csv'))

    print(f"Evaluating {mode} set...")
    # loop over CSV file rows (filename, startX, startY, endX, endY, label)
    for row in open(csv_file).read().strip().split("\n"):
        # TODO: read bounding box annotations
        filename, startX, startY, endX, endY, label = row.split(',')
        filename = os.path.join(config.IMAGES_PATH, label, filename)
        # TODO: add bounding box annotations here
        data.append((filename, startX, startY, endX, endY, label))

    print(f"Evaluating {len(data)} samples...")

    # Store all results as well as per class results
    results_labels[mode] = dict()
    results_labels[mode]['all'] = []
    for label_str in config.LABELS:
        results_labels[mode][label_str] = []

    # loop over the images that we'll be testing using our bounding box
    # regression model
    for filename, gt_start_x, gt_start_y, gt_end_x, gt_end_y, gt_label in data:
        # load the image, copy it, swap its colors channels, resize it, and
        # bring its channel dimension forward
        image = cv2.imread(filename)
        display = image.copy()
        h, w = display.shape[:2]

        # convert image to PyTorch tensor, normalize it, upload it to the
        # current device, and add a batch dimension
        image = config.TRANSFORMS(image).to(config.DEVICE)
        image = image.unsqueeze(0)

        # predict the bounding box of the object along with the class label
        # predict the bounding box of the object along with the class label
        predict = model(image)
        label_predictions = predict[0]
        bbox_predictions = predict[1]

        # determine the class label with the largest predicted probability
        label_predictions = torch.nn.Softmax(dim=-1)(label_predictions)
        most_likely_label = label_predictions.argmax(dim=-1).cpu()
        label = config.LABELS[most_likely_label]

        # denormalize predicted bounding box
        startX, startY, endX, endY = bbox_predictions.squeeze().detach().cpu().numpy()
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)

        # denormalize ground truth bounding box
        gt_startX = int(float(gt_start_x) * w)
        gt_startY = int(float(gt_start_y) * h)
        gt_endX = int(float(gt_end_x) * w)
        gt_endY = int(float(gt_end_y) * h)

        # Compare to gt data for classification
        results_labels['all'].append(label == gt_label)
        results_labels[gt_label].append(label == gt_label)

        # TODO: compute cumulated bounding box metrics (Intersection over Union - IoU)
        xA = max(startX, gt_startX)
        yA = max(startY, gt_startY)
        xB = min(endX, gt_endX)
        yB = min(endY, gt_endY)

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
        # compute the area of both the prediction and ground-truth rectangles
        boxAArea = max(0, endX - startX) * max(0, endY - startY)
        boxBArea = max(0, gt_endX - gt_startX) * max(0, gt_endY - gt_startY)
        # compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)

        results_iou['all'].append(iou)
        results_iou[gt_label].append(iou)

        if label != gt_label:
            print(f"\tFailure at {filename}")


# Compute per dataset accuracy
# Compute per dataset accuracy
for mode in ['train', 'validation', 'test']:
    print(f'\n*** {mode} set accuracy')
    print(f"\tMean accuracy for all labels: "
          f"{numpy.mean(numpy.array(results_labels['all'])):.4f}")
    
    # TODO: display bounding box metrics
    print(f"\tMean BBox IoU for all labels: "
          f"{numpy.mean(numpy.array(results_iou['all'])):.4f}")

    for label_str in config.LABELS:
        print(f'\n\tMean accuracy for label {label_str}: '
              f'{numpy.mean(numpy.array(results_labels[label_str])):.4f}')
        print(f'\t\t {numpy.sum(results_labels[label_str])} over '
              f'{len(results_labels[label_str])} samples')
        
        # TODO: display bounding box metrics
        print(f'\t\t Mean BBox IoU: '
              f'{numpy.mean(numpy.array(results_iou[label_str])):.4f}')

