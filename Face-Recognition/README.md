Face Recognition System


This project implements a simple face recognition system using OpenCV. The system captures images, trains a face recognition model using LBPH (Local Binary Patterns Histograms), and recognizes faces in real-time through a webcam.

Features
Capture images of faces and store them for training.
Train a face recognition model using labeled face data.
Recognize and label faces in real-time with confidence scores.


Step-by-Step Explanation
1. Setting up the Environment
Import necessary libraries such as cv2, os, and numpy.
Initialize a face detector using the Haar Cascade Classifier (haarcascade_frontalface_default.xml).
2. Capturing Face Images
Function: capture_images(User)

Description:

Opens the webcam to capture images of the user's face.
Converts each frame to grayscale for easier face detection.
Detects faces using the Haar Cascade Classifier and saves cropped face regions in a Faces/ directory.
Captures up to 100 face images for each user or exits if 'q' is pressed.
Output: Cropped face images stored as <UserName>_<Count>.jpg in the Faces directory.

3. Training the Face Recognition Model
Function: train_model(label)

Description:

Reads all images from the Faces directory.
Maps the file name to user labels provided in the label dictionary.
Detects faces in each image and crops the face region.
Trains an LBPH (Local Binary Patterns Histogram) face recognizer using the cropped face images and their corresponding labels.
Saves the trained model to a file named trained_model.xml.
Output: A trained LBPH model ready for real-time recognition.

4. Real-time Face Recognition
Function: recognize_faces(recognizer, label)

Description:

Opens the webcam and continuously reads frames.
Detects faces in the frame and uses the trained model to predict the label and confidence for each face.
Displays the user's name (if recognized) or "Unrecognized" (if the confidence is low).
Draws a rectangle around detected faces and overlays the recognized label on the video feed.
Stops recognition when 'q' is pressed.
Output: Real-time labeled video feed with recognized users' names displayed.

5. Directory Structure

bash
Copy code
Project/
│
├── Faces/                 # Directory where captured face images are stored
│   ├── User1_0.jpg
│   ├── User1_1.jpg
│   └── ...
│
├── trained_model.xml      # Trained face recognition model
├── main.py                # Python script containing the project code
└── README.md              # Explanation of the project


7. How to Run the Project
Install Requirements: Ensure OpenCV and NumPy are installed:

bash
Copy code
pip install opencv-python numpy
Capture Images: Uncomment the capture_images function call and pass the user's name:

python
Copy code
capture_images('Harshini')
Run the script to capture images.

Train the Model: Ensure the Faces folder contains images and call train_model:

python
Copy code
recognizer = train_model(label)
Recognize Faces: Run the recognize_faces function with the trained recognizer:

python
Copy code
recognize_faces(recognizer, label)
Press 'q' to quit the video feed during any stage.

Improvements
Add GUI for better user interaction.
Use a deep learning-based model for improved accuracy.
Extend functionality to recognize multiple faces in one frame simultaneously.
