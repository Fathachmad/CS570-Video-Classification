from collections import Counter
import csv
import os
from pprint import pprint
import torch
from tqdm import tqdm


TEST_DIR = "datasets/04_Test_Datasets/"

FULL_KEYS = {0: "plane", 24: "plane", 9: "bird"}
BINARY_KEYS = {
    0: "plane",
    1: "bird",
}

CLASSES = {"plane": ["plane", "aircraft", "jet"], "bird": ["bird"]}


def _classify_video(model, video_path, label_dict, frame_conf=1e-4, class_conf=0.7):

    results = model(video_path, stream=True, conf=frame_conf, verbose=False)

    def box_helper(result):
        class_conf = torch.stack((result.boxes.cls, result.boxes.conf), dim=1)
        if class_conf.numel() == 0:
            return -1

        def filter_boxes(box):
            return int(box[0]) in label_dict

        filtered_boxes = filter(filter_boxes, class_conf)
        filtered_boxes = list(filtered_boxes)
        if len(filtered_boxes) == 0:
            return -1
        ret = max(filtered_boxes, key=lambda x: x[1])
        # print(ret)
        return int(ret[0].item())

    predictions = [box_helper(r) for r in results]

    frame_preds = list(Counter(predictions).items())
    frames = sum([num for _, num in frame_preds])
    ratios = sorted(
        [(val, num / frames) for val, num in frame_preds],
        reverse=True,
        key=lambda x: x[1],
    )
    if ratios[0][0] != -1 or (
        ratios[0][0] == 1 and ratios[0][1] > class_conf or len(ratios) == 1
    ):
        return ratios[0][0]
    else:
        return ratios[1][0]


def classify_video(model, video_path, frame_conf, class_conf, label_dict):

    prediction = _classify_video(model, video_path, label_dict, frame_conf, class_conf)

    aligned_prediction = label_dict.get(prediction, "None")
    return aligned_prediction


def get_label(fname):
    for key, values in CLASSES.items():
        if any([value in fname for value in values]):
            return key
    return "None"


def get_test_files(test_dir):
    subdirs = [
        os.path.join(test_dir, dir)
        for dir in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, dir))
    ]
    test_files = []
    for subdir in subdirs:
        label = get_label(subdir.lower())
        subdir = os.path.join(subdir, "Test")
        test_files.extend(
            [(os.path.join(subdir, file), label) for file in os.listdir(subdir)]
        )
    return test_files


def eval_video(model, test_files, frame_conf, class_conf, label_dict):
    files = [t[0] for t in test_files]
    labels = [t[1] for t in test_files]
    labels_dist = {
        label: len([l for l in labels if l == label]) for label in set(labels)
    }
    print("starting eval...")
    predictions = []
    for file in tqdm(files):
        predictions.append(
            classify_video(
                model,
                file,
                frame_conf=frame_conf,
                class_conf=class_conf,
                label_dict=label_dict,
            )
        )

    with open("raw_eval.csv", "w", newline="\n") as file:
        writer = csv.writer(file)
        writer.writerow(["Label", "Prediction"])
        writer.writerows(zip(labels, predictions))

    predictions_dist = {
        label: {
            "guess_rate": len([pred for pred in predictions if pred == label])
            / len(labels),
            "True_Pos": len(
                [
                    pred
                    for pred, gt in zip(predictions, labels)
                    if pred == label and gt == pred
                ]
            ),
            "False_Pos": (
                len(
                    [
                        pred
                        for pred, gt in zip(predictions, labels)
                        if pred == label and gt != pred
                    ]
                )
            ),
            "True_Neg": len(
                [
                    pred
                    for pred, gt in zip(predictions, labels)
                    if pred != label and gt != label
                ]
            ),
            "False_Neg": (
                len(
                    [
                        pred
                        for pred, gt in zip(predictions, labels)
                        if pred != label and gt == label
                    ]
                )
            ),
        }
        for label in set(predictions)
    }

    # for f, l, p in zip(files, labels, predictions):
    #     print(f"{f.split('/')[-1]}: label={l}, prediction={p}")
    correct = sum(l == p for l, p in zip(labels, predictions))
    acc = correct / len(labels)
    print(f"NUM CORRECT: {correct} / {len(labels)} ({acc:.2f})")
    print(f"LABELS DISTRIBUTION:")
    pprint(labels_dist)
    print(f"PREDICTIONS DISTRIBUTION:")
    pprint(predictions_dist)
    return {
        "acc": acc,
        "labels_dist": labels_dist,
        "predictions_dist": predictions_dist,
    }
