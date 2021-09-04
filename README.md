

# A Low-Complexity Radar Detector Outperforming OS-CFAR for Indoor Drone Obstacle Avoidance
This repository contains an example code comparing our proposed detector to OS-CFAR, a video showcasing the reduced computational complexity of our system compared to standard detection schemes and a video showing our proposed detector working on-line on a drone.

## Example code

Please check-out the example code and dataset provided in this repository. It provides a Python implementation of our proposed detector and of OS-CFAR. By running the code, the Receiver Operating Curves (ROC) of our proposed detector and OS-CFAR showing the probability of detection vs. the probability of false alarm are computed and shown. 

## Computational complexity

Compared to standard 2D CA-CFAR detection, our proposed detector significantly reduces the detection complexity, enabling higher detection frames.

https://user-images.githubusercontent.com/10224818/119790963-ad040c80-bed4-11eb-9ab0-b4f77b945d84.mp4

## Our proposed detector in action

We implemented our detector in the ARM Cortex R4 MCU of the AWR1443 radar used throughout this work. The video below shows our detector working in real-time on-line at 30 FPS. The structure of the room (shelves, wall,...) can be easily seen in the radar detections which qualitatively shows the effectiveness of our proposed method.

https://user-images.githubusercontent.com/10224818/119706514-5eb22780-be5a-11eb-98c4-1f7c4e97c7e1.mp4

# How to cite

If you use this work, please cite:

**A. Safa et al., "A Low-Complexity Radar Detector Outperforming OS-CFAR for Indoor Drone Obstacle Avoidance," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, doi: 10.1109/JSTARS.2021.3107686.**










