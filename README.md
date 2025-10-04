 # ✨ Handwritten Digit Recognition using CNN

This project uses a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9).
Built with **TensorFlow/Keras**, it learns from the **MNIST dataset** and can predict digits from your own handwritten images.

---

## 📘 Project Overview

The goal of this project is to:

* Train a CNN to recognize handwritten digits.
* Test the model’s accuracy on unseen data.
* Use your own digit images for prediction.

---

## 🧠 Tech Stack

* **Python 3.9+**
* **TensorFlow / Keras** — For building and training the CNN
* **OpenCV** — For image processing
* **NumPy** — For data handling
* **Matplotlib** — For plotting results
* **Scikit-learn** — For evaluation metrics
* **Jupyter Notebook** — For interactive coding and visualization

---

## 📁 Folder Structure

```
digit-recognition-project/
│
├── digit_recognition.ipynb      # Main Jupyter Notebook
├── digit_model.h5               # Trained model (auto-saved after training)
├── requirements.txt             # All required dependencies
├── custom_digits/               # (Optional) Folder for your own digit images
└── README.md                    # Project documentation
```

---

## ⚙️ Setup and Instructions

### 1️⃣ Clone this Repository

```bash
git clone [https://github.com/Tejash1002/digit-recognition-project.git](https://github.com/Tejash1002/digit-recognition-project.git)
cd digit-recognition-project
```

### 2️⃣ Create and Activate a Virtual Environment

*Using Anaconda:*

```bash
conda create -n digit-env python=3.9 -y
conda activate digit-env
```

### 3️⃣ Install Required Packages

The recommended way is to use the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

*Alternatively, you can install the main packages manually:*
```bash
pip install tensorflow opencv-python numpy matplotlib scikit-learn jupyter
```

### 4️⃣ Run the Notebook

**Start the Jupyter Notebook server:**

```bash
jupyter notebook
```

**Open and run `digit_recognition.ipynb`:**
Once the server is running, open the notebook file in your browser and run all the cells one by one.

The notebook will perform the following steps:
* Load and preprocess the MNIST data.
* Build and train the CNN model.
* Plot the accuracy and loss curves.
* Evaluate the model's performance.
* Save the trained model as `digit_model.h5`.
* Provide a section to test the model on your own custom image.

---

## 🧠 Predict Your Own Digit

To test your own handwritten digit:

1.  Write a digit (0–9) on a white paper with a dark pen.
2.  Take a clear photo and crop it tightly around the digit.
3.  Save it as `my_digit.png` in the project folder (or inside `custom_digits/`).
4.  Update the file path in the prediction cell (the last cell) in the notebook and run it.

The model will preprocess your image and predict the digit.

---

## 📊 Model Summary

* **Dataset:** MNIST (70,000 images)
* **Accuracy:** Achieves ~98% accuracy on the test data.

---

## 💡 Tips

* If training is too slow on your computer, you can set `USE_SMALL_SUBSET = True` in the notebook for a faster demonstration. For the best results, set it to `False` to train on the full dataset.
* The prediction code automatically handles both white-on-black and black-on-white digit images.
* You can also train the model on your own dataset. The notebook explains the required folder structure.

---

## 👤 Author

**Tejash**

* [**GitHub Profile**](https://github.com/Tejash1002)

⭐ *If you found this project helpful, don’t forget to star the repository!*
