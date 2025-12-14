# Interim Progress Report
## Credit Risk Probability Model for Alternative Data

**Project**: Credit-Risk-Probability-Model-for-Alternative-Data  
**Date**: December 2024  
**Status**: In Progress

---

## Executive Summary

This interim report documents the progress on developing a credit risk probability model using alternative data sources (eCommerce transaction data) for **Bati Bank**, a leading financial service provider with over 10 years of experience. The project is part of a partnership with an eCommerce company to enable a **buy-now-pay-later (BNPL) service**, allowing customers to purchase products on credit if they qualify.

The project aims to transform behavioral transaction data into actionable credit risk signals while maintaining regulatory compliance with Basel II requirements. The key innovation lies in using **Recency, Frequency, and Monetary (RFM) analysis** to engineer a proxy for credit risk from eCommerce transaction patterns.

---

## Task 1: Project Understanding ✅

### 1.1 Business Context and Objectives

**Business Context**:
- **Client**: Bati Bank (leading financial service provider, 10+ years experience)
- **Partnership**: eCommerce company (Xente platform)
- **Service**: Buy-Now-Pay-Later (BNPL) credit facility
- **Challenge**: Enable credit decisions for eCommerce customers using transaction behavioral data

**Project Goal**: Build an end-to-end credit scoring system that:
1. Transforms eCommerce transaction behavioral data into credit risk signals
2. Provides interpretable risk assessments for regulatory compliance
3. Enables automated credit decision-making for BNPL service

**Key Business Drivers**:
- Enable credit access for eCommerce customers without traditional credit history
- Leverage alternative data (transaction patterns) for credit assessment
- Regulatory compliance with Basel II Capital Accord requirements
- Need for interpretable models that can be validated by regulators
- Support automated credit approval process for BNPL service

### 1.2 Project Deliverables

The project requires building a comprehensive product with **5 core components**:

#### Deliverable 1: Proxy Variable Definition
- **Objective**: Define a proxy variable to categorize users as **high risk (bad)** or **low risk (good)**
- **Approach**: Use RFM analysis and behavioral patterns from transaction data
- **Key Challenge**: No direct default labels available - must engineer meaningful proxy from behavioral indicators
- **Considerations**: 
  - Proxy must correlate with actual credit risk
  - Must be defensible to regulators
  - Should capture customer payment behavior patterns

#### Deliverable 2: Feature Selection
- **Objective**: Select observable features with **high correlation** to the proxy default variable
- **Approach**: 
  - RFM features (Recency, Frequency, Monetary)
  - Transaction patterns (ProductCategory, ChannelId, temporal features)
  - Behavioral indicators (FraudResult, transaction consistency)
- **Key Challenge**: Identify features that are both predictive and interpretable

#### Deliverable 3: Risk Probability Model
- **Objective**: Develop a model that assigns **risk probability** for a new customer
- **Output**: Probability score (0-1) indicating likelihood of default/high risk
- **Requirements**: 
  - Interpretable model (Logistic Regression with WoE recommended)
  - Regulatory compliance (Basel II)
  - Well-documented methodology

#### Deliverable 4: Credit Score Model
- **Objective**: Develop a model that assigns a **credit score** from risk probability estimates
- **Output**: Credit score (typically 300-850 scale or similar)
- **Approach**: Transform risk probability into standardized credit score
- **Considerations**: Score must be interpretable and aligned with business risk appetite

#### Deliverable 5: Loan Optimization Model
- **Objective**: Develop a model that predicts the **optimal loan amount and duration**
- **Output**: 
  - Recommended loan amount
  - Recommended loan duration/tenure
- **Approach**: Use risk probability and customer transaction patterns to determine appropriate credit limits
- **Considerations**: Balance risk with business opportunity

### 1.3 Regulatory Framework Understanding

#### Basel II Capital Accord Requirements

The project must align with Basel II's **Internal Ratings-Based (IRB) approaches**, which mandate:

1. **Supervisory Review (Pillar 2)**:
   - Regulators must validate and approve internal risk models
   - Comprehensive documentation of methodology, assumptions, and validation results required
   - Models must be well-understood, properly validated, and consistently applied

2. **Model Governance**:
   - Clear audit trail for all credit decisions
   - Ongoing model monitoring, validation, and recalibration
   - Transparency for stakeholders (regulators, auditors, senior management)

3. **Transparency Requirements**:
   - Models must be explainable
   - Feature contributions must be quantifiable
   - Decision logic must be traceable

**Implications for Our Model**:
- **Interpretability is mandatory**, not optional
- Model documentation must be comprehensive and clear
- Validation processes must be rigorous and well-documented
- Model choices must balance performance with regulatory compliance

### 1.4 Data Context and Challenges

#### Dataset Overview
- **Source**: Xente eCommerce platform transaction data
- **Type**: Alternative data (transaction behavioral patterns)
- **Data Location**: Available on Kaggle (Xente Challenge) or provided dataset
- **Key Challenge**: No direct "default" labels available - must create proxy variable

#### Complete Data Structure (15 Fields)

**Identifiers**:
- `TransactionId`: Unique transaction identifier on the platform
- `BatchId`: Unique number assigned to a batch of transactions for processing
- `AccountId`: Unique number identifying the customer on the platform
- `SubscriptionId`: Unique number identifying the customer subscription
- `CustomerId`: Unique identifier attached to Account

**Geographic and Provider Information**:
- `CurrencyCode`: Country currency (e.g., UGX)
- `CountryCode`: Numerical geographical code of the country (e.g., 256)
- `ProviderId`: Source provider of the item bought
- `ProductId`: Item name being bought
- `ProductCategory`: ProductIds organized into broader categories (airtime, financial_services, utility_bill, etc.)

**Transaction Details**:
- `ChannelId`: Identifies if customer used web, Android, iOS, pay later, or checkout
- `Amount`: Value of transaction (positive for debits from customer account, negative for credits into customer account)
- `Value`: Absolute value of the amount
- `TransactionStartTime`: Transaction start time (timestamp)
- `PricingStrategy`: Category of Xente's pricing structure for merchants
- `FraudResult`: Fraud status of transaction (1 = yes, 0 = No) - **potential proxy indicator**

#### Key Data Challenges Identified

1. **No Direct Default Labels**: 
   - Traditional credit data (payment histories, defaults) unavailable
   - Must create proxy variables from behavioral patterns

2. **Proxy Variable Necessity**:
   - Supervised learning requires labeled data
   - Behavioral patterns (RFM analysis) can infer creditworthiness
   - Need to engineer meaningful proxies from transaction patterns

3. **Data Quality Considerations**:
   - Transaction data may have missing values
   - Temporal patterns need to be captured
   - Aggregation from transaction-level to customer-level required

### 1.5 Modeling Approach and Trade-offs

#### Model Selection Strategy

**Primary Approach**: Logistic Regression with Weight of Evidence (WoE) transformations

**Rationale**:
- ✅ **Regulatory Compliance**: Easier to explain to regulators and auditors
- ✅ **Transparency**: Each feature's contribution is clear and quantifiable
- ✅ **Validation Simplicity**: Easier to validate assumptions and perform sensitivity analysis
- ✅ **Basel II Alignment**: Meets supervisory review requirements for model explainability
- ⚠️ **Trade-off**: May achieve lower predictive accuracy compared to complex models

**Alternative Consideration**: Gradient Boosting (as benchmark)
- Higher accuracy potential
- Black-box nature requires sophisticated interpretability techniques (SHAP, LIME)
- Regulatory challenges may arise

**Decision**: Start with interpretable model (Logistic Regression + WoE) for compliance, use Gradient Boosting as performance benchmark.

### 1.6 Risk Management Considerations

#### Proxy Variable Risks

1. **Proxy Drift**: Proxy may not perfectly correlate with actual default risk
2. **Model Misalignment**: Model optimized on proxy may not generalize to actual defaults
3. **Regulatory Scrutiny**: Need to empirically demonstrate proxy's relationship to credit risk
4. **Business Impact**: 
   - False Positives: Rejecting creditworthy customers (lost revenue)
   - False Negatives: Approving risky customers (credit losses)

#### Mitigation Strategies

- Regular validation against actual outcomes (when available)
- Continuous monitoring of proxy performance
- Clear documentation of proxy selection rationale and limitations
- Robust feature engineering (RFM analysis) to capture meaningful behavioral patterns

### 1.7 Feature Engineering Strategy

#### RFM Analysis Framework

**Recency (R)**: How recently did the customer make a transaction?
- Time since last transaction
- Indicates customer engagement and activity level

**Frequency (F)**: How often does the customer transact?
- Number of transactions in a given period
- Indicates customer loyalty and engagement

**Monetary (M)**: How much does the customer spend?
- Total transaction value
- Average transaction amount
- Indicates customer value and financial capacity

**Extended RFM (RFMS)**: 
- Standard deviation of transaction amounts
- Provides additional insight into spending behavior consistency

#### Additional Features to Consider

- Transaction patterns by ProductCategory
- Channel usage patterns (web vs mobile)
- Temporal patterns (day of week, time of day)
- Fraud indicators (FraudResult)
- Geographic patterns (CountryCode)
- Provider diversity

---

## Detailed Task Breakdown

### Task 1: Understanding Credit Risk ✅

**Status**: ✅ Complete

**Objective**: Understand the fundamentals of credit risk and how it applies to this project.

**Completed Activities**:
- ✅ Read and analyzed key references on Credit Risk and Basel II Capital Accord
- ✅ Created "Credit Scoring Business Understanding" section in README.md
- ✅ Documented Basel II implications for model interpretability
- ✅ Documented proxy variable necessity and business risks
- ✅ Documented model complexity trade-offs

**Deliverables**:
- ✅ Updated README.md with "Credit Scoring Business Understanding" section

---

### Task 2: Exploratory Data Analysis (EDA) ⏳

**Status**: ⏳ Not Yet Started

**Objective**: Explore the dataset to uncover patterns, identify data quality issues, and form hypotheses that will guide feature engineering.

**Instructions**:
- Use Jupyter Notebook (`notebooks/eda.ipynb`) for all exploratory work
- This notebook is for exploration only; not for production code

**Required Activities**:

#### 2.1 Overview of the Data
- [ ] Understand the structure of the dataset
- [ ] Number of rows, columns, and data types
- [ ] Initial data inspection

#### 2.2 Summary Statistics
- [ ] Understand central tendency, dispersion, and shape of distributions
- [ ] Calculate descriptive statistics for numerical features
- [ ] Analyze categorical feature distributions

#### 2.3 Distribution of Numerical Features
- [ ] Visualize distributions to identify patterns
- [ ] Identify skewness and potential outliers
- [ ] Create histograms, box plots for numerical features

#### 2.4 Distribution of Categorical Features
- [ ] Analyze frequency and variability of categories
- [ ] Create bar charts, pie charts for categorical features
- [ ] Identify dominant categories and rare categories

#### 2.5 Correlation Analysis
- [ ] Understand relationships between numerical features
- [ ] Create correlation matrix and heatmap
- [ ] Identify highly correlated features

#### 2.6 Identifying Missing Values
- [ ] Identify missing values across all columns
- [ ] Determine missing data patterns
- [ ] Decide on appropriate imputation strategies

#### 2.7 Outlier Detection
- [ ] Use box plots to identify outliers
- [ ] Analyze outlier patterns
- [ ] Determine outlier treatment strategy

**Deliverables**:
- [ ] Complete EDA notebook (`notebooks/eda.ipynb`)
- [ ] Summarize top 3-5 most important insights in the notebook

---

### Task 3: Feature Engineering ⏳

**Status**: ⏳ Not Yet Started

**Objective**: Build a robust, automated, and reproducible data processing script that transforms raw data into a model-ready format.

**Key Requirement**: Use `sklearn.pipeline.Pipeline` to chain together all transformation steps.

**Required Activities**:

#### 3.1 Create Aggregate Features
- [ ] **Total Transaction Amount**: Sum of all transaction amounts per customer
- [ ] **Average Transaction Amount**: Average transaction amount per customer
- [ ] **Transaction Count**: Number of transactions per customer
- [ ] **Standard Deviation of Transaction Amounts**: Variability of transaction amounts per customer

#### 3.2 Extract Temporal Features
- [ ] **Transaction Hour**: The hour of the day when the transaction occurred
- [ ] **Transaction Day**: The day of the month when the transaction occurred
- [ ] **Transaction Month**: The month when the transaction occurred
- [ ] **Transaction Year**: The year when the transaction occurred

#### 3.3 Encode Categorical Variables
- [ ] **One-Hot Encoding**: Convert categorical values into binary vectors
- [ ] **Label Encoding**: Assign unique integers to each category
- [ ] Choose appropriate encoding method for each categorical feature

#### 3.4 Handle Missing Values
- [ ] **Imputation Options**:
  - Mean, median, mode imputation
  - KNN imputation
- [ ] **Removal**: Remove rows/columns with missing values if appropriate
- [ ] Document strategy for each feature

#### 3.5 Normalize/Standardize Numerical Features
- [ ] **Normalization**: Scale data to range [0, 1]
- [ ] **Standardization**: Scale data to mean=0, std=1
- [ ] Choose appropriate scaling method

#### 3.6 Feature Engineering with WoE and IV
- [ ] Read about Weight of Evidence (WoE) and Information Value (IV)
- [ ] Implement WoE transformation using libraries like `xverse` or `woe`
- [ ] Calculate Information Value for feature selection

**Deliverables**:
- [ ] Complete `src/data_processing.py` script
- [ ] Implement sklearn Pipeline for all transformations
- [ ] Document feature engineering pipeline

---

### Task 4: Proxy Target Variable Engineering ⏳

**Status**: ⏳ Not Yet Started

**Objective**: Create a credit risk target variable since there is no pre-existing "credit risk" column in the data.

**Approach**: Programmatically identify "disengaged" customers and label them as high-risk proxies.

**Required Activities**:

#### 4.1 Calculate RFM Metrics
- [ ] For each `CustomerId`, calculate:
  - **Recency**: Time since last transaction
  - **Frequency**: Number of transactions
  - **Monetary**: Total transaction value
- [ ] Define a snapshot date to calculate Recency consistently
- [ ] Ensure RFM values are calculated correctly

#### 4.2 Cluster Customers
- [ ] Use **K-Means clustering** algorithm to segment customers into **3 distinct groups** based on RFM profiles
- [ ] Pre-process (scale) the RFM features appropriately before clustering
- [ ] Set `random_state` during clustering to ensure reproducibility
- [ ] Analyze cluster characteristics

#### 4.3 Define and Assign the "High-Risk" Label
- [ ] Analyze resulting clusters to determine which represents the least engaged (highest-risk) segment
- [ ] High-risk typically characterized by:
  - Low frequency
  - Low monetary value
  - High recency (inactive)
- [ ] Create new binary target column named `is_high_risk`
- [ ] Assign value of 1 to customers in high-risk cluster
- [ ] Assign value of 0 to all others

#### 4.4 Integrate the Target Variable
- [ ] Merge `is_high_risk` column back into main processed dataset
- [ ] Ensure target variable is ready for model training
- [ ] Validate target variable distribution

**Deliverables**:
- [ ] Proxy target variable (`is_high_risk`) created
- [ ] Target variable integrated into processed dataset
- [ ] Documentation of proxy variable definition rationale

---

### Task 5: Model Training and Tracking ⏳

**Status**: ⏳ Not Yet Started

**Objective**: Develop a structured model training process that includes experiment tracking, model versioning, and unit testing.

**Required Activities**:

#### 5.1 Setup
- [ ] Add `mlflow` and `pytest` to `requirements.txt`
- [ ] Install and configure MLflow
- [ ] Set up testing framework

#### 5.2 Data Preparation
- [ ] Split data into training and testing sets
- [ ] Set `random_state` for reproducibility
- [ ] Validate data splits

#### 5.3 Model Selection and Training
- [ ] Choose and train at least **two models** from:
  - [ ] Logistic Regression
  - [ ] Decision Tree
  - [ ] Random Forest
  - [ ] Gradient Boosting (XGBoost, LightGBM)

#### 5.4 Hyperparameter Tuning
- [ ] Use hyperparameter tuning techniques:
  - [ ] Grid Search
  - [ ] Random Search
- [ ] Optimize model performance
- [ ] Document best hyperparameters

#### 5.5 Experiment Tracking with MLflow
- [ ] Log all experiments to MLflow, including:
  - [ ] Model parameters
  - [ ] Evaluation metrics
  - [ ] Model artifacts
- [ ] Compare model runs in MLflow UI
- [ ] Identify best model
- [ ] Register best model in MLflow Model Registry

#### 5.6 Model Evaluation
- [ ] Assess model performance using metrics:
  - [ ] **Accuracy**: Ratio of correctly predicted observations
  - [ ] **Precision**: Ratio of correctly predicted positives
  - [ ] **Recall (Sensitivity)**: Ratio of correctly predicted positives to all actual positives
  - [ ] **F1 Score**: Weighted average of Precision and Recall
  - [ ] **ROC-AUC**: Area Under the ROC Curve

#### 5.7 Write Unit Tests
- [ ] In `tests/test_data_processing.py`, write at least two unit tests
- [ ] Test helper functions within scripts
- [ ] Example: Test that feature engineering function returns expected columns

**Deliverables**:
- [ ] Complete `src/train.py` script
- [ ] MLflow experiments logged and compared
- [ ] Best model registered in MLflow Model Registry
- [ ] Unit tests written and passing
- [ ] Model evaluation metrics documented

---

### Task 6: Model Deployment and Continuous Integration ⏳

**Status**: ⏳ Not Yet Started

**Objective**: Package the trained model into a containerized API and set up a CI/CD pipeline to automate testing and ensure code quality.

**Required Activities**:

#### 6.1 Setup
- [ ] Add `fastapi`, `uvicorn`, and linter (`flake8` or `black`) to `requirements.txt`
- [ ] Install and configure FastAPI
- [ ] Set up linting tools

#### 6.2 Create the API
- [ ] In `src/api/main.py`, build REST API using FastAPI
- [ ] API should load best model from MLflow registry
- [ ] Create `/predict` endpoint that:
  - [ ] Accepts new customer data (matching model's features)
  - [ ] Returns risk probability
- [ ] Use Pydantic models in `src/api/pydantic_models.py` for:
  - [ ] Request data validation
  - [ ] Response data validation

#### 6.3 Containerize the Service
- [ ] Write `Dockerfile` that:
  - [ ] Sets up the environment
  - [ ] Runs FastAPI application using uvicorn
- [ ] Write `docker-compose.yml` to easily build and run service
- [ ] Test containerized service

#### 6.4 Configure CI/CD
- [ ] In `.github/workflows/ci.yml`, create GitHub Actions workflow
- [ ] Workflow triggers on every push to main branch
- [ ] Workflow includes:
  - [ ] Step to run code linter (flake8) to check code style
  - [ ] Step to run pytest to execute unit tests
  - [ ] Build fails if linter or tests fail

**Deliverables**:
- [ ] Complete `src/api/main.py` with FastAPI application
- [ ] Complete `src/api/pydantic_models.py` with request/response models
- [ ] Complete `Dockerfile` for containerization
- [ ] Complete `docker-compose.yml` for service orchestration
- [ ] Complete `.github/workflows/ci.yml` with CI/CD pipeline
- [ ] API tested and working
- [ ] CI/CD pipeline tested and working

---

## Task 2: EDA Findings ⏳ (Detailed Checklist)

### 2.1 Planned EDA Activities

#### Data Quality Assessment
- [ ] Check for missing values across all columns
- [ ] Identify duplicate transactions
- [ ] Analyze data completeness by time period
- [ ] Check for outliers in Amount and Value fields
- [ ] Validate data types and formats

#### Univariate Analysis
- [ ] Distribution of transaction amounts
- [ ] Distribution of transaction frequencies per customer
- [ ] Analysis of ProductCategory distribution
- [ ] ChannelId usage patterns
- [ ] Temporal patterns (transaction timing)
- [ ] Geographic distribution (CountryCode)

#### Bivariate and Multivariate Analysis
- [ ] Relationship between Amount and ProductCategory
- [ ] Channel preferences by customer segments
- [ ] Temporal trends over time
- [ ] Correlation analysis between features
- [ ] Customer segmentation based on RFM features

#### RFM Feature Engineering and Analysis
- [ ] Calculate Recency, Frequency, Monetary metrics
- [ ] Analyze RFM score distributions
- [ ] Identify customer segments (Champions, At Risk, etc.)
- [ ] Explore RFM patterns by ProductCategory
- [ ] Temporal analysis of RFM metrics

#### Proxy Target Variable Exploration
- [ ] Define proxy target variable(s) based on behavioral patterns
- [ ] Analyze proxy target distribution
- [ ] Explore relationships between RFM features and proxy target
- [ ] Assess proxy target validity and business rationale

#### Data Preprocessing Requirements
- [ ] Identify features requiring transformation
- [ ] Determine handling strategy for missing values
- [ ] Plan for outlier treatment
- [ ] Define feature engineering pipeline
- [ ] Plan for temporal aggregation (transaction → customer level)

### 2.2 Expected Deliverables

Once Task 2 is completed, this section will include:

1. **Data Summary Statistics**
   - Dataset dimensions
   - Data quality metrics
   - Key statistics for numerical features

2. **Key Findings**
   - Data quality issues identified
   - Patterns discovered in transaction behavior
   - Insights from RFM analysis
   - Proxy target variable definition and rationale

3. **Visualizations**
   - Distribution plots
   - Correlation heatmaps
   - Temporal trend analysis
   - Customer segmentation visualizations

4. **Preprocessing Recommendations**
   - Data cleaning steps required
   - Feature engineering approach
   - Handling of missing values and outliers

5. **Top 3-5 Most Important Insights**
   - Documented in the EDA notebook
   - Key patterns that will guide feature engineering
   - Important data quality considerations

---

## Project Progress Summary

| Task | Status | Completion % | Notes |
|------|--------|--------------|-------|
| **Task 1: Understanding Credit Risk** | ✅ Complete | 100% | README.md updated with Credit Scoring Business Understanding section |
| **Task 2: Exploratory Data Analysis (EDA)** | ⏳ Pending | 0% | EDA notebook not yet started |
| **Task 3: Feature Engineering** | ⏳ Pending | 0% | Awaiting EDA completion |
| **Task 4: Proxy Target Variable Engineering** | ⏳ Pending | 0% | RFM clustering and proxy definition pending |
| **Task 5: Model Training and Tracking** | ⏳ Pending | 0% | Awaiting feature engineering and proxy variable |
| **Task 6: Model Deployment and CI/CD** | ⏳ Pending | 0% | Awaiting model training completion |

### Deliverables Status

| Deliverable | Status | Completion % | Notes |
|-------------|--------|--------------|-------|
| **Deliverable 1: Proxy Variable Definition** | ⏳ Pending | 0% | Part of Task 4 - RFM clustering approach |
| **Deliverable 2: Feature Selection** | ⏳ Pending | 0% | Part of Task 3 - Feature engineering |
| **Deliverable 3: Risk Probability Model** | ⏳ Pending | 0% | Part of Task 5 - Model training |
| **Deliverable 4: Credit Score Model** | ⏳ Pending | 0% | Part of Task 5 - Model training (transformation) |
| **Deliverable 5: Loan Optimization Model** | ⏳ Pending | 0% | Part of Task 5 - Model training (amount/duration) |

---

## Key Insights and Learnings

### From Task 1

1. **Business Context is Clear**: Bati Bank's BNPL service requires a comprehensive credit scoring system with 5 distinct deliverables, not just a single risk model.

2. **Regulatory Compliance is Paramount**: Basel II requirements make interpretability a regulatory necessity, not just a best practice. This significantly influences model selection.

3. **Proxy Variables are the Foundation**: The first and most critical deliverable is defining a meaningful proxy variable. Without this, all subsequent models cannot be built. This must be done carefully with strong business rationale.

4. **Multi-Model System Required**: The project requires 3 interconnected models:
   - Risk Probability Model (core)
   - Credit Score Model (transformation of probability)
   - Loan Optimization Model (business application)

5. **RFM Analysis is Core Innovation**: The RFM (Recency, Frequency, Monetary) framework is the key innovation that transforms behavioral data into credit risk signals.

6. **Feature Selection is Critical**: Deliverable 2 requires identifying features with high correlation to the proxy - this directly impacts model performance.

7. **End-to-End Product**: This is not just a model, but a complete product that enables automated credit decisions for BNPL service.

8. **Documentation is Critical**: Comprehensive documentation of methodology, assumptions, and validation is required for regulatory approval.

---

## Next Steps

### Immediate Priority: Task 2 - Exploratory Data Analysis

1. **Start EDA Notebook** (`notebooks/eda.ipynb`):
   - Load and explore the Xente transaction dataset
   - Perform comprehensive data quality assessment
   - Complete all 7 required EDA activities:
     - Overview of data structure
     - Summary statistics
     - Distribution analysis (numerical and categorical)
     - Correlation analysis
     - Missing value identification
     - Outlier detection
   - Document top 3-5 most important insights

### Short-term: Feature Engineering and Proxy Variable

2. **Task 3: Feature Engineering**:
   - Implement `src/data_processing.py` with sklearn Pipeline
   - Create aggregate features (Total, Average, Count, Std Dev)
   - Extract temporal features (Hour, Day, Month, Year)
   - Encode categorical variables
   - Handle missing values
   - Normalize/standardize features
   - Implement WoE and IV transformations

3. **Task 4: Proxy Target Variable Engineering**:
   - Calculate RFM metrics for each customer
   - Use K-Means clustering (3 clusters) on RFM features
   - Identify high-risk cluster (least engaged customers)
   - Create `is_high_risk` binary target variable
   - Integrate target variable into processed dataset

### Medium-term: Model Development

4. **Task 5: Model Training and Tracking**:
   - Set up MLflow for experiment tracking
   - Train at least 2 models (Logistic Regression + one other)
   - Perform hyperparameter tuning (Grid Search/Random Search)
   - Evaluate models using all required metrics
   - Register best model in MLflow Model Registry
   - Write unit tests for data processing functions

### Long-term: Deployment and CI/CD

5. **Task 6: Model Deployment and CI/CD**:
   - Build FastAPI application with `/predict` endpoint
   - Create Pydantic models for request/response validation
   - Containerize service with Dockerfile and docker-compose.yml
   - Set up GitHub Actions CI/CD pipeline
   - Test end-to-end deployment

---

## Appendix

### References

- Basel II Capital Accord Framework
- Credit Risk Management Best Practices
- RFM Analysis Methodology
- Alternative Credit Scoring Approaches (World Bank, HKMA)

### Data Dictionary

**Complete Variable List** (15 fields from Xente dataset):

| Variable | Type | Description | Potential Use |
|----------|------|-------------|---------------|
| `TransactionId` | Identifier | Unique transaction identifier | Join key, deduplication |
| `BatchId` | Identifier | Batch processing identifier | Transaction grouping |
| `AccountId` | Identifier | Customer account identifier | Customer aggregation |
| `SubscriptionId` | Identifier | Subscription identifier | Customer segmentation |
| `CustomerId` | Identifier | Unique customer identifier | **Primary aggregation key** |
| `CurrencyCode` | Categorical | Country currency (UGX, etc.) | Geographic analysis |
| `CountryCode` | Categorical | Numerical country code (256, etc.) | Geographic segmentation |
| `ProviderId` | Categorical | Source provider of item | Provider diversity analysis |
| `ProductId` | Categorical | Item name being bought | Product-level analysis |
| `ProductCategory` | Categorical | Product category (airtime, financial_services, utility_bill, etc.) | **Key feature for RFM** |
| `ChannelId` | Categorical | Channel (web, Android, iOS, pay later, checkout) | **Behavioral feature** |
| `Amount` | Numerical | Transaction value (positive=debit, negative=credit) | **Core RFM feature** |
| `Value` | Numerical | Absolute value of amount | **Core RFM feature** |
| `TransactionStartTime` | DateTime | Transaction timestamp | **Temporal analysis, Recency** |
| `PricingStrategy` | Categorical | Xente's pricing structure category | Pricing analysis |
| `FraudResult` | Binary | Fraud status (1=yes, 0=no) | **Potential proxy component** |

**Key Variables for RFM Analysis**:
- **Recency**: Derived from `TransactionStartTime`
- **Frequency**: Count of transactions per `CustomerId`
- **Monetary**: Aggregated `Value` or `Amount` per `CustomerId`

**Potential Proxy Indicators**:
- `FraudResult`: Direct risk indicator
- Transaction patterns: Inconsistency, negative amounts (credits)
- Channel usage: Pay later behavior
- ProductCategory patterns: Financial services usage

---

**Report Last Updated**: December 2024  
**Next Review Date**: After Task 2 Completion

