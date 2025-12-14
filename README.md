# Credit-Risk-Probability-Model-for-Alternative-Data

An end-to-end machine learning system that transforms eCommerce behavioral data into actionable credit risk signals using RFM analysis, proxy target engineering, MLflow tracking, FastAPI deployment, and CI/CD automation.

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Credit Scoring Business Understanding](#credit-scoring-business-understanding)
- [Development](#development)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **RFM Analysis**: Recency, Frequency, Monetary value analysis of eCommerce transaction data
- **Proxy Target Engineering**: Creates credit risk proxies from behavioral patterns when direct default labels are unavailable
- **MLflow Integration**: Comprehensive model tracking, versioning, and experiment management
- **FastAPI Deployment**: RESTful API for real-time credit risk predictions
- **Interpretable Models**: Logistic Regression with Weight of Evidence (WoE) transformations for regulatory compliance
- **CI/CD Pipeline**: Automated testing and deployment workflows
- **Docker Support**: Containerized deployment for consistent environments

## Project Structure

```
credit-risk-model/
├── data/
│   ├── raw/              # Raw input data
│   └── processed/        # Processed and feature-engineered data
├── notebooks/
│   └── eda.ipynb         # Exploratory data analysis
├── src/
│   ├── api/
│   │   ├── main.py       # FastAPI application
│   │   └── pydantic_models.py  # API request/response models
│   ├── data_processing.py  # Data preprocessing and feature engineering
│   ├── train.py          # Model training pipeline
│   └── predict.py        # Prediction utilities
├── tests/
│   └── tests_data_processing.py  # Unit tests
├── .github/
│   └── workflows/
│       └── ci.yml        # CI/CD pipeline configuration
├── Dockerfile            # Docker container configuration
├── docker-compose.yml    # Docker Compose configuration
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker and Docker Compose for containerized deployment

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Danielmituku/credit-risk-model.git
   cd credit-risk-model
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables** (if needed)
   ```bash
   cp .env.example .env  # Create .env file with your configuration
   ```

## Usage

### Training the Model

```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Perform RFM analysis and feature engineering
- Train the credit risk model
- Log experiments and models to MLflow

### Making Predictions

```bash
python src/predict.py
```

### Running the API Server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run Docker container directly
docker build -t credit-risk-model .
docker run -p 8000:8000 credit-risk-model
```

## API Documentation

Once the FastAPI server is running, interactive API documentation is available at:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Example API Request

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "customer_id": "12345",
    "recency": 30,
    "frequency": 5,
    "monetary": 1000.0,
    # ... other features
}

response = requests.post(url, json=data)
print(response.json())
```

## Credit Scoring Business Understanding

### Basel II Accord and Model Interpretability

The Basel II Capital Accord's emphasis on risk measurement fundamentally shapes our modeling approach. Under Basel II's **Internal Ratings-Based (IRB) approaches**, financial institutions must use internal models to calculate capital requirements for credit risk. This regulatory framework mandates:

- **Supervisory Review (Pillar 2)**: Regulators must validate and approve internal risk models, requiring comprehensive documentation of methodology, assumptions, and validation results.
- **Model Governance**: Banks must demonstrate that their models are well-understood, properly validated, and consistently applied.
- **Transparency Requirements**: The framework emphasizes explainability so that stakeholders (regulators, auditors, senior management) can understand how risk is quantified.

**Implications for our model**: We need an interpretable and well-documented model because:
1. **Regulatory Compliance**: Regulators must be able to validate our risk assessment methodology and understand how capital requirements are derived.
2. **Audit Trail**: Clear documentation enables traceability of decisions, which is critical for regulatory examinations and internal audits.
3. **Stakeholder Trust**: Interpretable models build confidence among business users, risk managers, and executives who need to understand and justify credit decisions.
4. **Model Validation**: Interpretability facilitates ongoing model monitoring, validation, and recalibration, which are required under Basel II's supervisory review process.

### Proxy Variable Necessity and Business Risks

Since we lack a direct "default" label in our alternative data context, creating a proxy variable is **necessary** because:

1. **Data Availability**: Traditional credit data (payment histories, defaults) may not be available for all borrowers, especially in underserved markets or for new-to-credit customers.
2. **Model Training Requirement**: Supervised learning algorithms require labeled data to learn patterns. Without a proxy, we cannot train a predictive model.
3. **Behavioral Inference**: Alternative data (e.g., eCommerce transaction patterns) can serve as indicators of financial behavior and creditworthiness, even if not directly measuring default.

**Potential Business Risks** of using a proxy variable:

1. **Proxy Drift**: The proxy may not perfectly correlate with actual default risk. For example, low transaction frequency might indicate financial distress, but could also reflect seasonal patterns or lifestyle choices.
2. **Model Misalignment**: A model optimized on proxy performance may not generalize to actual default outcomes, leading to suboptimal credit decisions.
3. **Regulatory Scrutiny**: Regulators may question the validity of proxy-based models, especially if the proxy's relationship to actual credit risk cannot be empirically demonstrated.
4. **Business Impact**: Incorrect credit decisions based on proxy predictions can result in:
   - **False Positives**: Rejecting creditworthy customers (lost revenue, customer dissatisfaction)
   - **False Negatives**: Approving risky customers (credit losses, capital erosion)
5. **Model Validation Challenges**: Validating proxy-based models requires establishing the proxy's predictive power for actual defaults, which may require additional data or longer observation periods.

**Mitigation Strategies**: Regular validation against actual outcomes (when available), continuous monitoring of proxy performance, and clear documentation of proxy selection rationale and limitations.

### Model Complexity Trade-offs in Regulated Financial Context

The choice between simple, interpretable models (e.g., **Logistic Regression with Weight of Evidence (WoE)**) and complex, high-performance models (e.g., **Gradient Boosting**) involves critical trade-offs in a regulated financial environment:

#### Simple, Interpretable Models (Logistic Regression with WoE)

**Advantages:**
- **Regulatory Compliance**: Easier to explain to regulators, auditors, and business stakeholders
- **Transparency**: Each feature's contribution is clear and quantifiable (coefficients, WoE transformations)
- **Validation Simplicity**: Easier to validate assumptions, test stability, and perform sensitivity analysis
- **Documentation**: Straightforward to document methodology and decision logic
- **Debugging**: Issues can be traced to specific features or transformations
- **Basel II Alignment**: Meets supervisory review requirements for model explainability

**Disadvantages:**
- **Performance Limitations**: May achieve lower predictive accuracy, especially with complex non-linear relationships
- **Feature Engineering Dependency**: Requires careful feature engineering (WoE binning) to capture non-linearities
- **Limited Complexity**: May miss subtle interactions between features

#### Complex, High-Performance Models (Gradient Boosting)

**Advantages:**
- **Superior Performance**: Often achieves higher accuracy, better AUC, and improved discrimination
- **Automatic Feature Interactions**: Captures complex non-linear relationships and feature interactions automatically
- **Robustness**: Better handling of missing values and outliers
- **Competitive Advantage**: Higher accuracy can translate to better risk-adjusted returns

**Disadvantages:**
- **Black-Box Nature**: Difficult to explain individual predictions or feature contributions
- **Regulatory Challenges**: Regulators may require extensive justification and may impose restrictions
- **Validation Complexity**: Harder to validate assumptions, test stability, and perform sensitivity analysis
- **Documentation Burden**: Requires sophisticated techniques (SHAP, LIME) to provide interpretability
- **Operational Risk**: Unexplained decisions can lead to regulatory issues, customer disputes, or model governance failures

#### Recommended Approach for Regulated Context

**Hybrid Strategy**: 
1. **Primary Model**: Use Logistic Regression with WoE for regulatory compliance and interpretability
2. **Validation Benchmark**: Use Gradient Boosting as a benchmark to assess potential performance gains
3. **Ensemble Consideration**: If performance gap is significant, consider ensemble approaches with clear documentation of interpretability techniques
4. **Documentation**: Regardless of model choice, maintain comprehensive documentation of model development, validation, and interpretability methods

**Decision Framework**: In a regulated financial context, **interpretability often outweighs marginal performance gains** unless the performance improvement is substantial and can be justified with proper documentation and validation. The Basel II emphasis on supervisory review and transparency makes interpretability a regulatory requirement, not just a best practice.

## Development

### Setting Up Development Environment

1. Follow the [Installation](#installation) steps
2. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Code Style

This project follows PEP 8 style guidelines. Code formatting is enforced using:

- **Black**: Automatic code formatting
- **Flake8**: Linting and style checking
- **MyPy**: Static type checking

Run code quality checks:
```bash
black src/ tests/
flake8 src/ tests/
mypy src/
```

### MLflow Tracking

The project uses MLflow for experiment tracking. To view experiments:

```bash
mlflow ui
```

This will start the MLflow UI at `http://localhost:5000`

## Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/tests_data_processing.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and add tests
4. Ensure all tests pass (`pytest`)
5. Run code quality checks (`black`, `flake8`, `mypy`)
6. Commit your changes (`git commit -m 'Add some amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Pull Request Guidelines

- Ensure CI checks pass
- Update documentation if needed
- Add tests for new features
- Follow the existing code style

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built for credit risk assessment using alternative data sources
- Designed with Basel II regulatory compliance in mind
- Implements best practices for interpretable machine learning in financial services
