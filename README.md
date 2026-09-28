# ♻️ Waste Classification AI — MobileNetV3Large

## Classes
Glass, Paper, Metal, Plastic

## Structure
```text
waste_classification_mobilenetv3/
├── app.py
├── train_model.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── class_names.json
│   └── waste_mobilenetv3.keras   # generated after training
└── dataset/
    ├── Glass/
    ├── Paper/
    ├── Metal/
    └── Plastic/
```

## Install
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset
Put images into the four class folders under `dataset/`.

## Train
Open `train_model.ipynb` and run the cells. The notebook saves:
`models/waste_mobilenetv3.keras` and `models/class_names.json`.

## Run the web app
```bash
streamlit run app.py
```

**Note:** The trained `.keras` model is not included in this ZIP because it must be generated from your dataset (and can be large).
