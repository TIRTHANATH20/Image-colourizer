Image-colorisation
This project is a Python-based application that allows users to colorize black and white images using machine learning models. The application features a GUI built with Tkinter and uses OpenCV's deep learning framework to perform image colorization. The model is pre-trained and loaded from the disk using a Caffe-based architecture.

Features Upload Image: Users can upload a black and white image from their computer. Colorize Image: Once an image is uploaded, the app will colorize it and display both the original and colorized images. User Interface: The application has a graphical interface that displays the uploaded and colorized images side-by-side. Prerequisites Python 3.x opencv-python (for OpenCV functions) numpy (for numerical operations) tkinter (for GUI) Pillow (for image handling in Tkinter) To install the necessary dependencies, run the following command:

Model Files You will need to download the following files and place them in the specified directories:

colorization_deploy_v2.prototxt: Defines the structure of the neural network. colorization_release_v2.caffemodel: Pre-trained weights for the colorization model. pts_in_hull.npy: Quantization centers for the ab channels used in LAB color space. Place these files in the model folder in the project directory.

Example Directory Structure: css Copy code Image Colorizer │ ├── main.py ├── model/ │ ├── colorization_deploy_v2.prototxt │ ├── colorization_release_v2.caffemodel │ └── pts_in_hull.npy How to Run Download or clone this repository to your local machine. Ensure that the necessary dependencies are installed using the command mentioned above. Place the model files (.prototxt, .caffemodel, .npy) in the correct folder structure.

Code Breakdown

Model Loading The model is loaded using OpenCV’s dnn module:
python Copy code net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL) The quantization centers for the ab channels are loaded from pts_in_hull.npy and used in layers of the model:

python Copy code pts = np.load(POINTS) 2. Image Colorization The colorize_image() function takes the input black and white image, converts it to LAB color space, uses the model to predict the ab color channels, and then combines them with the L channel to form the final colorized image.

Tkinter GUI The GUI is built using Tkinter, allowing users to upload images and display both the original and colorized versions. It features:
A title An upload button for image selection A colorize button to trigger the colorization process 4. Gradient Background A custom gradient background is generated for the GUI to enhance the user experience:

python Copy code def create_gradient_image(): # Create a gradient image Acknowledgments This project uses a pre-trained model for image colorization based on Caffe and OpenCV. Special thanks to the OpenCV community for providing the pre-trained models and the deep learning framework.
