# Driver-Assistance-and-Safety-System
Minor/Major project B.Tech

### What we are trying to Achieve

  - Detection of Potholes and asphalt deformalities to alert the driver before hand
  - Porject these visual alerts on the windscreen to the driver knows the exact position of the poholes
  - Track the facial expressions of the driver and predict whether the driver is alert or not
  




### Technologies Used

We are using the following libraries for our project

* [Tensor FLow API](https://github.com/tensorflow/models/tree/master/research/object_detection)- To Train the models on the potholes dataset
* [OpenCv](https://github.com/opencv/opencv) - To capture the Video feed and do the image processing
* [Dllib](https://github.com/davisking/dlib) - Facial landmarking

### References
* [Tensor FLow API Documentation] (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)

### Approximate Roadmap/Workflow
1. Working of the system
    1. Geting the Video feed from the camera or the Mobile phone.
    2. Use Opencv to perform the preprocessing on the input Video feed frame by frame.
    3. Feeding the Input to a tensorflow image classifier that predicts and gives and location of the poholes in the image.
    4. Plotting the Bounding box using Opencv and projecting them on the screen.
    5. Parallely Using another camera or phone monitoring the driver's facial expressing and alerting the driver when he/she is not paying attention to the driving and the road is busy.
2. Making of the system
    1. Making a dataset of Pothole images and resizing them using OpenCV.
    2. Lbaleing these images to train our Deep learning model using tranfer learning.
    3. Training our CNN to recognise these potholes in the given photos.
    4. comparing different models to know what gives the best speed to accuracy ratio
    5. using this model , implementing a predictor function to identify the potholes in the give video feed
    6. For facial expressions repeating the above steps for detecting the distracted driver
    7. Making a decission script to relate the road condition and driver alertness to decide whether to alert the driver or not.

