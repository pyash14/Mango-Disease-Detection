# Mango Disease Detection 🥭

A simple Streamlit app that looks at a photo of a mango leaf and tells you which disease it has (or if it's healthy). It runs on a MobileNet model trained on labeled mango leaf images.

## Award

This project won "Most Innovative Use of Technology" at the 2025 LA Geospatial Summit ArcGIS StoryMaps Competition. Awarded to Yash Kishorbhai Pansheriya.

- Award announcement: https://dornsife.usc.edu/spatial/2025/03/28/celebrating-innovation-highlights-from-the-2025-la-geospatial-summit-arcgis-storymaps-competition/
- StoryMap for this project: http://storymaps.arcgis.com/stories/b2ae676f4dfd4d42a9c437b162c3e44e

## What it does

Mango crops deal with a handful of common leaf diseases that can hurt yield if they go unnoticed. This app takes an uploaded leaf image and classifies it into one of eight categories:

- Anthracnose
- Bacterial Canker
- Cutting Weevil
- Die Back
- Gall Midge
- Healthy
- Powdery Mildew
- Sooty Mould

You upload a photo, and it gives you back the predicted label plus a confidence score.

## How it works

1. On startup, the app loads the trained model (`MoblieNet_mango.h5`). If it's not already on disk, it downloads it from the GitHub repo automatically.
2. The uploaded image gets resized to 224x224, turned into an array, and run through MobileNet's preprocessing.
3. The model predicts probabilities for each of the 8 classes.
4. The app shows the image back to you along with the predicted disease and confidence score.

## Built with

- Streamlit for the UI
- TensorFlow / Keras (MobileNet) for the model
- NumPy for array handling
- Pillow for image loading
- gdown to fetch the model file

## Running it locally

You'll need Python 3.8 or newer.

```bash
git clone https://github.com/pyash14/Mango-Disease-Detection.git
cd Mango-Disease-Detection
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints out (usually `http://localhost:8501`), upload a mango leaf image, and check the result.

## Project structure

```
.
├── app.py                 # the Streamlit app
├── MoblieNet_mango.h5     # trained model
├── requirements.txt       # dependencies
└── README.md
```

## License

Provided as-is for educational and research use.
