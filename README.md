# Vehicle Insurance Claim Fraud Detection

## 📌 Project Overview
This project is an end-to-end Machine Learning solution aimed at detecting fraudulent vehicle insurance claims. A clean, flat-file structure is used for simplicity and ease of deployment. The core of the system is a Streamlit application (`app.py`) that uses a pre-trained Machine Learning pipeline (`pipeline.pkl`) to predict whether a claim is legitimate or fraudulent in real-time.

## 🎯 Problem Statement
Insurance fraud is a significant issue leading to massive financial losses for insurance companies. Correctly identifying fraudulent claims helps in:
*   Reducing financial loss.
*   Speeding up the processing of legitimate claims.
*   Automating the initial screening process.

## ⚙️ ML Approach
The project follows a rigorous Machine Learning workflow:
1.  **Data Processing**: 
    *   Dataset analysis (`Vehicle_Insurance_Claim.ipynb`).
    *   Cleaning invalid values (e.g., '0' in dates).
    *   Dropping irrelevant IDs.
2.  **Pipeline Construction**:
    *   **Categorical Features**: Handled via `OneHotEncoder`.
    *   **Numerical Features**: Scaled using `StandardScaler`.
    *   **Imputation**: Simple imputation strategies for robustness.
3.  **Model Selection**:
    *   Evaluated `Logistic Regression`, `Random Forest`, and `Gradient Boosting`.
    *   Selected the best performing model based on F1-Score (to handle class imbalance) and Accuracy.
    *   The final model and preprocessing steps are saved in a single `pipeline.pkl` file.

## 🚀 How to Run Locally

### Prerequisites
*   Python 3.8+ installed.

### Steps
1.  **Clone/Download** this repository.
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the App**:
    ```bash
    streamlit run app.py
    ```
4.  **Access**: Open your browser at `http://localhost:8501`.

## ☁️ How to Deploy (Streamlit Cloud)
This project is "One-Click Deploy" ready for Streamlit Cloud.

1.  Push this code to a GitHub repository.
2.  Log in to [Streamlit Cloud](https://streamlit.io/cloud).
3.  Click **"New App"**.
4.  Select your repository and branch.
5.  Set "Main file path" to `app.py`.
6.  Click **"Deploy"**.

The application will automatically install dependencies from `requirements.txt` and launch.

## 📂 File Structure
*   `app.py`: Main Streamlit application file.
*   `pipeline.pkl`: Serialized ML pipeline (Preprocessor + Model).
*   `requirements.txt`: List of Python dependencies.
*   `analysis_and_training.py`: Script used to train and generate the pipeline.
*   `fraud_oracle.csv`: Dataset used for training and input validation.
