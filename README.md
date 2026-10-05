# Air Quality Analytics and Pollution Prediction

## Project
A Data Analytics and Data Visualization mini project that analyzes environmental air-quality patterns and compares two machine-learning regression models for predicting CO concentration.

## Important dataset note
The included CSV is a **synthetic academic dataset with 3,000 records**, generated with a fixed random seed to resemble the structure of the UCI Air Quality dataset. It is included so the project runs immediately without a dataset download.

If your guide specifically requires a real-world public dataset, replace `data/air_quality_data.csv` with the official UCI Air Quality CSV and adjust the preprocessing if needed.

## Features
- Data preprocessing and missing-value handling
- Exploratory data analysis
- Interactive visualizations
- Monthly and hourly pollution analysis
- Environmental condition analysis
- Linear Regression
- Random Forest Regression
- MAE, RMSE and R² comparison
- Interactive CO prediction interface

## Run in 3 steps

### Windows
1. Extract the ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal in VS Code and run:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The browser should open automatically. If it does not, open the local URL shown in the terminal, usually:
`http://localhost:8501`

## Regenerate the dataset
If needed:

```bash
python generate_data.py
```

## GitHub
Create a new GitHub repository, for example:
`air-quality-analytics`

Then upload all project files except `venv/`.

Suggested repository structure:

```text
Air_Quality_Analytics/
├── data/
│   └── air_quality_data.csv
├── images/
├── models/
├── app.py
├── generate_data.py
├── requirements.txt
└── README.md
```

## Academic honesty
Do not claim the included synthetic dataset is measured real-world data. If your mentor requires a public real dataset, use the UCI Air Quality dataset and cite it in the final report.

## Suggested project explanation
Problem: Environmental air pollution is difficult to understand from raw measurements alone.

Solution: Analyze pollutant and environmental variables through a dashboard and compare two ML models for CO prediction.

Models:
1. Linear Regression
2. Random Forest Regression

Evaluation:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
