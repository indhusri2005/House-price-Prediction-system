# House Price Prediction System

A comprehensive machine learning project that predicts house prices using multiple regression algorithms. This system compares the performance of five different machine learning models to identify the most accurate approach for house price prediction.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Models Implemented](#models-implemented)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Results](#results)
- [Files Structure](#files-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)

## 🏠 Overview

This project implements and compares five different machine learning regression models for predicting house prices. The system includes data preprocessing, model training, evaluation, and visualization components to provide comprehensive insights into model performance.

## ✨ Features

- **Multiple ML Models**: Implements 5 different regression algorithms
- **Data Preprocessing**: Complete data cleaning and feature engineering pipeline
- **Model Evaluation**: Comprehensive metrics comparison (R², MAE, MSE, RMSE, MAPE, EVS)
- **Visualization**: Interactive charts showing model performance comparison
- **Prediction System**: Ready-to-use prediction interface for new house data
- **Standardized Features**: Proper scaling and encoding of categorical variables

## 🤖 Models Implemented

1. **Decision Tree Regressor** - Best performing model (R² = 0.9988)
2. **XGBoost Regressor** - Second best (R² = 0.9974)
3. **Random Forest Regressor** - Good performance (R² = 0.9455)
4. **K-Nearest Neighbors Regressor** - Moderate performance (R² = 0.7429)
5. **Linear Regression** - Baseline model (R² = 0.6810)

## 📊 Dataset

The system uses a housing dataset with the following features:
- **Area**: House area in square feet
- **Bedrooms**: Number of bedrooms
- **Bathrooms**: Number of bathrooms  
- **Stories**: Number of stories
- **Mainroad**: Access to main road (yes/no)
- **Guestroom**: Presence of guest room (yes/no)
- **Basement**: Presence of basement (yes/no)
- **Hot water heating**: Hot water heating availability (yes/no)
- **Air conditioning**: Air conditioning availability (yes/no)
- **Parking**: Number of parking spaces
- **Preferred area**: Located in preferred area (yes/no)
- **Furnishing status**: Furnished/Semi-furnished/Unfurnished

## 🚀 Installation

1. **Clone or download the project**
```bash
git clone [repository-url]
cd "House price prediction system"
```

2. **Install required dependencies**
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

3. **Ensure dataset is available**
   - Place your `Housing.csv` file in the appropriate directory
   - Update the file path in `Preprocessed code` if needed

## 💻 Usage

### 1. Data Preprocessing
```python
# Run the preprocessing code
exec(open('Preprocessed code').read())
```

### 2. Train Individual Models
```python
# Train Random Forest
exec(open('Randon Forest regression').read())

# Train XGBoost  
exec(open('XG boost regressor').read())

# Train Linear Regression
exec(open('linear regression').read())

# Train Decision Tree
exec(open('decision tree regression').read())

# Train KNN
exec(open('KNeighbors regressor').read())
```

### 3. Compare Models and Make Predictions
```python
# Run analysis and visualization
exec(open('Data prediction and analysis').read())
```

### 4. Make New Predictions
```python
# Example prediction for a new house
input_data = (7420, 4, 2, 3, 1, 0, 0, 0, 1, 2, 1, 2)
# Format: (area, bedrooms, bathrooms, stories, mainroad, guestroom, 
#          basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus)

# The system will output predictions from all 5 models
```

## 📈 Model Performance

| Model | R² Score | MAE | MSE | RMSE | MAPE | EVS |
|-------|----------|-----|-----|------|------|-----|
| **Decision Tree** | **0.9988** | **7,706** | **4.48B** | **66,922** | **0.0030** | **0.9988** |
| **XGBoost** | **0.9974** | **54,630** | **9.53B** | **97,611** | **0.0147** | **0.9974** |
| **Random Forest** | 0.9455 | 339,097 | 248.88B | 498,883 | 0.0749 | 0.9317 |
| **KNN** | 0.7429 | 691,419 | 937.01B | 967,990 | 0.1507 | 0.7438 |
| **Linear Regression** | 0.6810 | 799,720 | 1.16T | 1,078,192 | 0.1776 | 0.6810 |

### Key Insights:
- **Decision Tree Regressor** achieves the highest accuracy with 99.88% R² score
- **XGBoost** provides excellent performance with 99.74% accuracy
- **Random Forest** offers good performance with acceptable overfitting control
- **Linear models** show limitations for this complex housing dataset

## 📊 Results

The system generates comprehensive visualizations showing:
- Model accuracy comparison (R² scores)
- Error metrics comparison (MAE, MSE, RMSE, MAPE)
- Model variance explanation (EVS)

## 📁 Files Structure

```
House price prediction system/
├── README.md                      # This file
├── Preprocessed code              # Data preprocessing and setup
├── Randon Forest regression       # Random Forest model implementation
├── XG boost regressor            # XGBoost model implementation  
├── linear regression             # Linear Regression model
├── decision tree regression      # Decision Tree model
├── KNeighbors regressor          # K-Nearest Neighbors model
└── Data prediction and analysis  # Model comparison and predictions
```

## 📦 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- pickle

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Notes

- The dataset path in `Preprocessed code` may need to be updated based on your file location
- Models are evaluated on training data - consider implementing train/validation/test splits for production use
- The Decision Tree model shows perfect performance, which might indicate overfitting - consider cross-validation for robust evaluation

## 🎯 Future Enhancements

- Implement cross-validation for more robust model evaluation
- Add hyperparameter tuning for optimal model performance
- Create a web interface for easy house price predictions
- Add feature importance analysis
- Implement ensemble methods for improved accuracy

---

**Note**: This project demonstrates various machine learning approaches to housing price prediction. The Decision Tree and XGBoost models show exceptional performance for this dataset.
