# CS570-Project

Environment Requirements:
* Torch
* OpenCV
* SciPy
* ultralytics
* Pillow
* matplotlib
* scikit-learn

File Category:
* Data Augmentation
  -VideoMix.py
  -VideoTransform.py
  -augment.py
  -dataset.py

* Object Tracking for YOLO
  -object_tracking/custom_yolo/customops.py
  -object_tracking/custom_yolo/custompredictor.py
  -object_tracking/custom_yolo/customresults.py
  -object_tracking/custom_yolo/customtracker.py
  -object_tracking/custom_yolo/customyolo.py
  -object_tracking/score_calc.py
  -object_tracking/track_checker.py
  -object_tracking/track_extraction.py
  -object_tracking/yololstm.py
  -model.py

* Finetuning
  -YOLO_training.ipynb
  -convert_baseline.ipynb
  -custom_yolo.py
  -inference-video.ipynb

* Bias Testing and Data Distribution
  -bias_testing.ipynb
  -custom_eval.py
  -plotter.ipynb
