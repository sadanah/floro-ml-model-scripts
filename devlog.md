## 2025-07-02  

Here's a **summary of everything we did today** to get your orchid disease detection model trained and ready for use in your app:

---

### **1. Dataset Preparation**

* You decided to **start with orchids only**, not anthuriums yet.
* Your dataset has 5 classes:
  `anthracnose`, `erwinia`, `fusarium`, `soft rot`, `healthy`
* Structured dataset in folders:

  ```
  orchid-dataset/
    train/
      anthracnose/
      ...
    valid/
    test/
  ```

---

### **2. Environment Setup**

* Created a new virtual environment: `floro310-env`
* Installed dependencies including:

  * `tensorflow`
  * `tensorflow-hub`
  * `tensorflow-model-optimization`
  * `Pillow`
  * `scipy`
* Encountered and fixed errors related to `PIL.Image` and `scipy` missing

---

### **3. Model Setup (Transfer Learning)**

* Chose **EfficientNet Lite (EfficientNetB0 from Keras)** for transfer learning
* Built a classification model with `GlobalAveragePooling2D` and `Dense` layers
* Set up a training script `train_orchid_model.py`

---

### **4. Model Training**

* Trained the model with:

  * `10 epochs`
  * `ImageDataGenerator` (no augmentation yet)
  * Achieved \~**49% accuracy** on test set

---

### **5. Model Export**

* Successfully exported the trained model to TFLite:

  * File: `orchid_disease_model.tflite`

---

### **6. Next Steps Discussed**

* Improve accuracy later (e.g., 60%+) by:

  * Adding **data augmentation**
  * Training longer (e.g., 30–50 epochs)
  * Trying more EfficientNet versions (B1, B2)
* Start building the **mobile app** now and integrate the model

---

### **7. Tracking Progress**

* Discussed how to **track experiments** for project reporting:

  * Maintain a `devlog.md` or Google Doc
  * Save model versions with names like `model_v1_49acc.tflite`
  * Organize project folders clearly

---

Would you like me to now generate a clean `devlog.md` template or the updated `train_orchid_model.py` with augmentation added?

- Ran EfficientNet Lite model on orchid dataset  
- Epochs: 10  
- Accuracy: ~49%  
- Added scipy and pillow due to runtime errors  


Next step: Add augmentation + increase to 30 epochs


## Problem with overfitting

Overfitting was a problem because the dataset was imbalanced 
 - soft-rot and anthacnose had over 500-600 data 
 - other two had about 10-60 each 
 - healthy had about 60


 ## !Need to address healthy plant under representation problem

 ### Trained model again for two diseases

 Removed the lesser classes and kept the major ones.


 ## New model accuracy 

 After initial training of 10 epochs: loss: 0.8791 - accuracy: 0.5205
 After another augmentation and 10 epochs: loss: 0.8709 - accuracy: 0.5205