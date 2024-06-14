# CS570-Project
In this project, we experiment utilize the Yolov8 model from Ultralytics, and pretrain them on object detection task. We also develop LSTM model that feature the object tracking from Yolo to see if it improves the performance. Finally, we implemented data augmentation method from VideoMix that we apply on privately owned dataset.


# Environment Requirements:
* Torch
* OpenCV
* SciPy
* ultralytics
* Pillow
* matplotlib
* scikit-learn

# File Category:
- **Data Augmentation**
  - VideoMix.py
  - VideoTransform.py
  - augment.py
  - dataset.py

- **Object Tracking for YOLO**
  - object_tracking/custom_yolo/customops.py
  - object_tracking/custom_yolo/custompredictor.py
  - object_tracking/custom_yolo/customresults.py
  - object_tracking/custom_yolo/customtracker.py
  - object_tracking/custom_yolo/customyolo.py
  - object_tracking/score_calc.py
  - object_tracking/track_checker.py
  - object_tracking/track_extraction.py
  - object_tracking/yololstm.py
  - model.py

- **Finetuning**
  - YOLO_training.ipynb
  - convert_baseline.ipynb
  - custom_yolo.py
  - inference-video.ipynb

- **Bias Testing and Data Distribution**
  - bias_testing.ipynb
  - custom_eval.py
  - plotter.ipynb
```
