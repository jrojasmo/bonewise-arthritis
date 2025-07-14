# Knee Arthritis Detection using CNNs

This project uses a deep learning approach to classify the severity of knee arthritis from X-ray images. The model is built using a Convolutional Neural Network (CNN).

The model classifies knee X-rays into five distinct stages according to the Kellgren-Lawrence (KL) grading system:

  * Grade 0: Normal
  * Grade 1: Doubtful
  * Grade 2: Mild
  * Grade 3: Moderate
  * Grade 4: Severe

## Dataset

This project utilizes the **Annotated Dataset for Knee Arthritis Detection** available on Kaggle. You can download it from [this link](https://www.kaggle.com/datasets/hafiznouman786/annotated-dataset-for-knee-arthritis-detection).

The training dataset contains X-ray images organized into folders corresponding to the five arthritis grades.

## How to run
First ensure that you have your **kaggle API key** visible to the Kaggle CLI (inside of `~/.kaggle`)

Then, setup a python virtual env and run `pip install -r requirements.txt`

To run the web_app locally you can execute `streamlit run web_app/streamlit_app.py`
