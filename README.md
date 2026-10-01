\# 🛡️ Hinglish/Kannada-English Scam Message Detector



A Machine Learning and Natural Language Processing project that classifies messages as \*\*SCAM\*\* or \*\*NORMAL\*\*, with a focus on English, Hinglish, and Kannada-English code-mixed messages.



\## 📌 Project Overview



Scam messages often use informal language, spelling variations, short forms, and code-mixed text to appear convincing.



This project uses \*\*Word TF-IDF + Character TF-IDF features with Logistic Regression\*\* to identify suspicious message patterns and classify incoming messages.



The trained model is integrated into an interactive \*\*Streamlit web application\*\* for real-time message classification.



\## 🎯 Objectives



\* Detect potentially suspicious scam messages.

\* Preprocess informal and code-mixed text.

\* Extract Word and Character TF-IDF features.

\* Train a Logistic Regression classification model.

\* Compare different feature representations.

\* Evaluate the model using standard classification metrics.

\* Provide an easy-to-use Streamlit interface.



\## 📊 Dataset



The final dataset contains \*\*3,887 messages\*\*.



| Category | Count |

| -------- | ----: |

| Normal   | 3,090 |

| Scam     |   797 |

| Total    | 3,887 |



The dataset combines a public labeled dataset with a small set of synthetic examples created for the project.



The data was divided into:



\* \*\*80% Training:\*\* 3,109 messages

\* \*\*20% Testing:\*\* 778 messages



\## ⚙️ Methodology



```text

Dataset

&#x20;  ↓

Data Preprocessing

&#x20;  ↓

Word TF-IDF + Character TF-IDF

&#x20;  ↓

Hybrid Feature Combination

&#x20;  ↓

Logistic Regression

&#x20;  ↓

SCAM / NORMAL Prediction

&#x20;  ↓

Streamlit Web Application

```



\## 🧹 Data Preprocessing



The preprocessing pipeline includes:



\* Converting text to lowercase

\* Removing URLs

\* Removing unnecessary special characters

\* Preserving Kannada Unicode characters

\* Normalizing whitespace



\## 🔤 Feature Extraction



\### Word TF-IDF



Captures important word-level patterns and terms occurring in messages.



\### Character TF-IDF



Captures character-level patterns and helps handle:



\* Spelling variations

\* Informal writing

\* Short forms

\* Code-mixed text



\### Hybrid Features



The project combines both Word and Character TF-IDF representations before classification.



\## 🤖 Machine Learning Model



The main classifier used is:



\*\*Logistic Regression\*\*



The hybrid model combines:



```text

Word TF-IDF features

&#x20;       +

Character TF-IDF features

&#x20;       ↓

Logistic Regression

&#x20;       ↓

SCAM / NORMAL

```



\## 📈 Model Comparison



| Model                                  | Test Accuracy |

| -------------------------------------- | ------------: |

| Word TF-IDF + Logistic Regression      |        96.53% |

| Character TF-IDF + Logistic Regression |        95.24% |

| Hybrid Word + Character TF-IDF         |    \*\*97.43%\*\* |



The hybrid representation achieved the highest measured test accuracy in this experiment.



\## 📊 Model Performance



The hybrid model achieved:



| Metric         |     Result |

| -------------- | ---------: |

| Accuracy       | \*\*97.43%\*\* |

| Scam Precision |   \*\*100%\*\* |

| Scam Recall    |  \*\*87.5%\*\* |

| Scam F1-Score  |    \*\*93%\*\* |



\### Confusion Matrix



```text

&#x20;                Predicted

&#x20;             Normal    Scam

Actual Normal    618       0

Actual Scam       20     140

```



This means:



\* 618 normal messages were correctly classified.

\* 140 scam messages were correctly classified.

\* 20 scam messages were missed.

\* 0 normal messages were incorrectly classified as scam.



\## 🌐 Streamlit Web Application



The project includes a Streamlit interface where users can enter a message and receive:



\* SCAM or NORMAL prediction

\* Prediction confidence

\* Model information

\* Performance information

\* Confusion matrix



Example:



```text

Input:

Nimma bank account block agutte, KYC update madoke ee link click madi



Output:

SCAM MESSAGE DETECTED

```



\## 🛠️ Technologies Used



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* SciPy

\* Matplotlib

\* Seaborn

\* Joblib

\* Streamlit

\* Logistic Regression

\* TF-IDF

\* Git \& GitHub

\* VS Code



\## 📁 Project Structure



```text

Hinglish\_Kannada\_Scam\_Detector/

│

├── app.py

├── README.md

├── project\_documentation.txt

│

├── data/

│   ├── messages.csv

│   ├── data.json

│   ├── hinglish\_dataset.csv

│   └── final\_dataset.csv

│

├── models/

│   ├── hybrid\_model.pkl

│   ├── word\_vectorizer.pkl

│   └── char\_vectorizer.pkl

│

├── results/

│   ├── model\_comparison.py

│   ├── model\_comparison.csv

│   ├── model\_accuracy\_comparison.png

│   └── confusion\_matrix.png

│

├── src/

│   ├── preprocessing.py

│   ├── hybrid\_model.py

│   ├── character\_model.py

│   ├── predict.py

│   ├── confusion\_matrix.py

│   ├── combine\_dataset.py

│   ├── convert\_dataset.py

│   ├── check\_final\_dataset.py

│   ├── load\_data.py

│   └── train.py

│

└── .gitignore

```



\## ▶️ How to Run



\### 1. Clone the repository



```bash

git clone https://github.com/shreyamokshanatha-hue/Hinglish-Kannada-English-Scam-Message-Detector.git

```



\### 2. Open the project



```bash

cd Hinglish-Kannada-English-Scam-Message-Detector

```



\### 3. Create and activate a virtual environment



Windows PowerShell:



```powershell

python -m venv venv

venv\\Scripts\\activate

```



\### 4. Install dependencies



```bash

pip install pandas numpy scikit-learn matplotlib seaborn scipy joblib streamlit

```



\### 5. Run the Streamlit application



```bash

streamlit run app.py

```



The application will open in your browser.



\## 🔮 Future Enhancements



\* Increase the size and diversity of Kannada-English training data.

\* Add more real-world scam categories.

\* Support additional Indian languages.

\* Experiment with advanced machine learning and deep learning models.

\* Add continuous model retraining for new scam patterns.

\* Deploy the application online.



\## ⚠️ Limitations



\* The dataset may not represent every real-world scam pattern.

\* Kannada-English/Hinglish examples are limited compared with the overall dataset.

\* New scam patterns may not be detected accurately.

\* The model prediction is not a guaranteed confirmation that a message is a scam.

\* Human verification is still important for suspicious messages.



\## 📚 Project Type



\*\*Academic Data Mining / Machine Learning Project\*\*



\*\*Domain:\*\* Natural Language Processing, Text Classification, Cybersecurity



\## 👩‍💻 Author



\*\*Shreya G M\*\*



Computer Science \& Engineering



\---



⭐ If you find this project useful, consider giving the repository a star.



