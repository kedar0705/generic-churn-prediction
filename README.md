# Customer Churn Analysis

This project aims to analyze customer churn data from various sources and build predictive models to identify customers who are likely to churn. The project includes data preprocessing, exploratory data analysis, model training, and evaluation.

### Directories

- **churn_data/**: Contains various datasets related to customer churn.
- **models/**: Contains trained model files.
- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis and model development.
- **src/**: Contains source code for data processing, model training, and explanations.

## Setup

### Prerequisites

- Python 3.11.9
- pip (Python package installer)

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd generic-churn-prediction
```

### Usage

- Start  by installing the required packages using pip: ```pip install -r requirements.txt```

- Start the FastAPI server: ```uvicorn src.main:app --reload```

Note: modify line 54: 
df = load_data(f'/home/user/Kedar/generic-churn-prediction/churn_data/{dataset}.csv') 
