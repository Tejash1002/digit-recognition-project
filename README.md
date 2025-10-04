# ✨ Handwritten Digit Recognition using CNN

This project uses a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9).  
Built with **TensorFlow/Keras**, it learns from the **MNIST dataset** and can predict digits from your own handwritten images.

---

## 📘 Project Overview

The goal of this project is to:

- Train a CNN to recognize handwritten digits.
- Test the model’s accuracy on unseen data.
- Use your own digit images for prediction

---

## 🧠 Tech Stack

-  **Python 3.9+**
- **TensorFlow / Keras** — for building and training the CNN  
- **OpenCV** — for image processing  
- **NumPy** — for data handling  
- **Matplotlib** — for plotting results  
- **Scikit-learn** — for evaluation metrics  
- **Jupyter Notebook** — for interactive coding and visualization

---

## 📁 Folder Structure
 digit-recognition-project/
│
├── digit_recognition.ipynb          # Main Jupyter Notebook
├── digit_model.h5                   # Trained model (auto-saved after training)
├── requirements.txt                 # All required dependencies
├── custom_digits/                   # (Optional) Folder for your own digit images
└── README.md                        # Project documentation

## ⚙️ Setup Instructions

### 1️⃣ Clone this Repository
git clone https://github.com/Tejash1002/digit-recognition-project.git

cd digit-recognition-project

2️⃣ Create and Activate Environment

-Using Anaconda

conda create -n digit-env python=3.9 -y

conda activate digit-env

3️⃣ Install Required Packages

pip install tensorflow opencv-python numpy matplotlib scikit-learn jupyter

pip install -r requirements.txt

4️⃣ Run the Notebook

##1️⃣ Start Jupyter Notebook

jupyter notebook

2️⃣ Open and Run the Notebook

Open digit_recognition.ipynb and run all the cells one by one.

The notebook will:

-Load and preprocess MNIST data

-Build and train the CNN model

-Plot accuracy/loss curves

-Evaluate model performance

-Save the model (digit_model.h5)

-Test the model on your own image

🧠 Predict Your Own Digit
To test your own handwritten digit:

-Write a digit (0–9) on white paper with a dark pen.

-Take a clear photo and crop it tightly around the digit.

-Save it as my_digit.png in the project folder.

-Run the prediction cell (Cell 8) in the notebook.

The model will preprocess the image and predict the digit with confidence.

📊 Model Summary

Dataset: MNIST (70,000 images)

Accuracy: ~98% on test data

💡 Tips
-If training is slow, set USE_SMALL_SUBSET = True (faster demo).

-For better accuracy, set it to False to train on the full dataset.

-If your digit image is inverted (black background), the code auto-fixes it.

-You can also train using your own dataset (folder structure explained in the notebook).

👤 Author
Tejash
📎 GitHub Profile

⭐ If you found this project helpful, don’t forget to star the repository!













