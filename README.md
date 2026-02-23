# WearSafe: Activity Recognition & Heart Rate Forecasting System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tableau](https://img.shields.io/badge/Tableau-Public-blue.svg)](https://public.tableau.com/app/profile/faisal.gaffoor/viz/FinalProjectv3_17718083570410/Dashboard1?publish=yes)

An intelligent IoT wearable system that combines deep learning and machine learning to provide real-time activity recognition and proactive heart rate forecasting for enhanced safety and health monitoring.

## 🎯 Project Overview

WearSafe is a comprehensive IoT solution designed for real-time physical activity monitoring and cardiovascular health prediction. The system leverages wearable sensor data to classify 12 different physical activities and forecast heart rate trends, enabling proactive health interventions and safety alerts.

### Problem Statement

Traditional fitness trackers provide reactive insights—they tell you what happened after the fact. WearSafe takes a proactive approach by predicting future heart rate trends and providing real-time activity classification, enabling users and healthcare providers to intervene before potential health issues arise.

### Solution

Our system combines two powerful machine learning models:
- **1D-CNN for Activity Recognition**: Classifies physical activities in real-time from multi-sensor data
- **XGBoost for Heart Rate Forecasting**: Predicts future heart rate trends with high accuracy

## 🚀 Key Features

- **Real-time Activity Recognition**: 1D-CNN model achieving 77.9% accuracy across 12 activity types
- **Heart Rate Forecasting**: XGBoost model with MAE of 0.426 BPM and R² of 0.997
- **Interactive Dashboard**: Tableau Public visualization with 4 comprehensive charts
- **End-to-End Pipeline**: Data preprocessing, model training, evaluation, and deployment
- **Reproducible Research**: Complete notebooks and scripts for full reproducibility

## 📊 Dataset

This project uses the **PAMAP2 Physical Activity Monitoring** dataset from the UCI Machine Learning Repository.

**Dataset Details:**
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)
- **Subjects**: 9 participants
- **Activities**: 18 different physical activities (12 used in this project)
- **Sensors**: 3 Inertial Measurement Units (IMU) + Heart Rate Monitor
- **Sampling Rate**: 100 Hz
- **Features**: Accelerometer, gyroscope, magnetometer readings, temperature, heart rate

**Activities Monitored:**
1. Lying
2. Sitting
3. Standing
4. Walking
5. Running
6. Cycling
7. Nordic Walking
8. Ascending Stairs
9. Descending Stairs
10. Vacuum Cleaning
11. Ironing
12. Rope Jumping

## 🏗️ System Architecture

```
┌─────────────────┐
│  Wearable       │
│  Sensors        │
│  (IMU + HR)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data           │
│  Preprocessing  │
└────────┬────────┘
         │
         ├──────────────────┬──────────────────┐
         ▼                  ▼                  ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1D-CNN     │    │  XGBoost    │    │  Feature    │
│  Activity   │    │  Heart Rate │    │  Engineering│
│  Classifier │    │  Forecaster │    │             │
└──────┬──────┘    └──────┬──────┘    └─────────────┘
       │                  │
       └────────┬─────────┘
                ▼
        ┌───────────────┐
        │   Tableau     │
        │   Dashboard   │
        └───────────────┘
```

## 📈 Model Performance

### Activity Recognition (1D-CNN)

| Metric | Value |
|--------|-------|
| **Accuracy** | 77.9% |
| **Macro-F1 Score** | 0.646 |
| **Architecture** | 3 Conv1D layers + MaxPooling + Dense layers |
| **Input Shape** | (100, 52) - 100 timesteps, 52 features |

### Heart Rate Forecasting (XGBoost)

| Metric | Value |
|--------|-------|
| **Mean Absolute Error (MAE)** | 0.426 BPM |
| **R² Score** | 0.997 |
| **Root Mean Squared Error (RMSE)** | ~0.5 BPM |
| **Features** | Activity type, sensor readings, temporal features |

## 🎨 Dashboard Visualizations

Explore the live interactive dashboard on [Tableau Public](https://public.tableau.com/app/profile/faisal.gaffoor/viz/FinalProjectv3_17718083570410/Dashboard1?publish=yes)

### Visualization 1: Activity Distribution (Donut Chart)
Shows the proportion of time spent in each activity during a monitoring session. Provides an at-a-glance understanding of workout composition.

### Visualization 2: Model Performance (Bar Chart)
Displays key evaluation metrics for both the 1D-CNN and XGBoost models, providing transparency about system reliability.

### Visualization 3: HR Forecasting (Dual-Axis Line Chart)
Compares actual heart rate (blue) with predicted heart rate (orange) over time, demonstrating forecasting accuracy.

### Visualization 4: Heart Rate Over Time (Area Chart)
Shows the complete cardiovascular profile throughout the session, highlighting intensity variations and recovery periods.

## 🛠️ Technologies Used

### Machine Learning & Data Science
- **TensorFlow/Keras** - Deep learning framework for 1D-CNN
- **XGBoost** - Gradient boosting for heart rate forecasting
- **scikit-learn** - Model evaluation and preprocessing
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Visualization
- **Tableau Public** - Interactive dashboard creation
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical data visualization

### Development Environment
- **Python 3.8+** - Primary programming language
- **Jupyter Notebook** - Interactive development
- **Google Colab** - Cloud-based training environment

## 📦 Repository Structure

```
wearsafe-activity-hr-forecasting/
├── data/                          # Dataset and preprocessed data
│   ├── raw/                       # Original PAMAP2 dataset
│   └── processed/                 # Cleaned and preprocessed data
├── notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_activity_recognition.ipynb
│   └── 04_hr_forecasting.ipynb
├── src/                          # Source code
│   ├── preprocessing/            # Data preprocessing scripts
│   │   ├── data_loader.py
│   │   └── feature_engineering.py
│   ├── models/                   # Model training and evaluation
│   │   ├── cnn_model.py
│   │   └── xgboost_model.py
│   └── visualization/            # Visualization scripts
│       └── plot_utils.py
├── models/                       # Saved trained models
│   ├── cnn_activity_classifier.h5
│   └── xgboost_hr_forecaster.pkl
├── dashboard/                    # Tableau workbook and data sources
│   ├── tableau_activity_distribution.csv
│   ├── tableau_model_performance.csv
│   └── tableau_hr_forecasting.csv
├── docs/                         # Documentation and reports
│   └── WearSafe_Final_Report.pdf
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/wearsafe-activity-hr-forecasting.git
cd wearsafe-activity-hr-forecasting
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the PAMAP2 dataset**
```bash
# Download from UCI ML Repository
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00231/PAMAP2_Dataset.zip
unzip PAMAP2_Dataset.zip -d data/raw/
```

### Usage

#### 1. Data Preprocessing
```bash
python src/preprocessing/data_loader.py
python src/preprocessing/feature_engineering.py
```

#### 2. Train Activity Recognition Model
```bash
python src/models/cnn_model.py
```

#### 3. Train Heart Rate Forecasting Model
```bash
python src/models/xgboost_model.py
```

#### 4. Generate Visualizations
```bash
python src/visualization/plot_utils.py
```

### Running Notebooks

Launch Jupyter Notebook to explore the interactive analysis:
```bash
jupyter notebook notebooks/
```

## 📊 Results

### Activity Recognition Confusion Matrix

The 1D-CNN model shows strong performance across most activity classes, with particularly high accuracy for stationary activities (lying, sitting, standing) and distinct movement patterns (running, cycling).

### Heart Rate Forecasting Accuracy

The XGBoost model achieves near-perfect R² score (0.997) with an MAE of only 0.426 BPM, demonstrating exceptional predictive capability for short-term heart rate trends.

## 🔬 Methodology

### Data Preprocessing
1. **Missing Value Handling**: Removed rows with missing heart rate data
2. **Feature Selection**: Selected 52 most relevant sensor features
3. **Normalization**: Applied StandardScaler to sensor readings
4. **Windowing**: Created sliding windows of 100 timesteps (1 second at 100Hz)
5. **Train-Test Split**: 80-20 split with stratification

### Model Training

**1D-CNN Architecture:**
- Input Layer: (100, 52)
- Conv1D Layer 1: 64 filters, kernel size 3, ReLU activation
- MaxPooling1D: pool size 2
- Conv1D Layer 2: 128 filters, kernel size 3, ReLU activation
- MaxPooling1D: pool size 2
- Conv1D Layer 3: 64 filters, kernel size 3, ReLU activation
- GlobalAveragePooling1D
- Dense Layer: 64 units, ReLU activation
- Dropout: 0.5
- Output Layer: 12 units, Softmax activation

**XGBoost Configuration:**
- Objective: reg:squarederror
- Max Depth: 6
- Learning Rate: 0.1
- N Estimators: 100
- Features: Activity class, sensor statistics, temporal features

## 📝 Academic Context

This project was developed as the final project for **AAI-530: IoT Analytics** at the **University of San Diego** as part of the Applied Artificial Intelligence program. It demonstrates the application of machine learning and deep learning techniques to real-world IoT sensor data for health monitoring and safety applications.

**Course**: AAI-530 - IoT Analytics  
**Institution**: University of San Diego  
**Program**: Master of Science in Applied Artificial Intelligence  
**Team**: Group 3  
**Date**: February 2026

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PAMAP2 Dataset Creators**: Reiss, A. and Stricker, D. for creating and sharing the dataset
- **UCI Machine Learning Repository**: For hosting and maintaining the dataset
- **University of San Diego**: Applied Artificial Intelligence Program
- **AAI-530 Course Instructors**: For guidance and support throughout the project
- **TensorFlow & XGBoost Communities**: For excellent documentation and resources

## 📚 References

1. Reiss, A. and Stricker, D. (2012). "Introducing a New Benchmarked Dataset for Activity Monitoring." The 16th IEEE International Symposium on Wearable Computers (ISWC), 2012.
2. Goodfellow, I., Bengio, Y., and Courville, A. (2016). "Deep Learning." MIT Press.
3. Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." KDD '16.

## 📧 Contact

For questions, suggestions, or collaboration opportunities:

- **Project Repository**: [GitHub](https://github.com/yourusername/wearsafe-activity-hr-forecasting)
- **Tableau Dashboard**: [View Live Dashboard](https://public.tableau.com/app/profile/faisal.gaffoor/viz/FinalProjectv3_17718083570410/Dashboard1?publish=yes)
- **Issues**: [GitHub Issues](https://github.com/yourusername/wearsafe-activity-hr-forecasting/issues)

---

**Keywords:** IoT, Wearable Sensors, Activity Recognition, Heart Rate Forecasting, Deep Learning, 1D-CNN, XGBoost, Health Monitoring, PAMAP2, Machine Learning, TensorFlow, Tableau, Real-time Analytics

---

⭐ **If you find this project useful, please consider giving it a star!** ⭐
