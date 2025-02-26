# FIN305 Risk Analysis and Portfolio Management

**A comprehensive risk management study using financial econometrics and portfolio analysis techniques.**

## Table of Contents

- [Introduction](#introduction)
- [Dataset and Methodology](#dataset-and-methodology)
- [Question 1: Portfolio Construction](#question-1-portfolio-construction)
  - [Summary Statistics of Stocks](#summary-statistics-of-stocks)
  - [Efficient Frontier and Optimal Portfolio](#efficient-frontier-and-optimal-portfolio)
- [Question 2: Value-at-Risk (VaR) Analysis](#question-2-value-at-risk-var-analysis)
  - [VaR Estimation Methods](#var-estimation-methods)
  - [Rolling Window VaR and Backtesting](#rolling-window-var-and-backtesting)
  - [VaR Model Validation](#var-model-validation)
- [Question 3: Volatility Modeling Using GARCH](#question-3-volatility-modeling-using-garch)
  - [GARCH(1,1) Model Estimation](#garch11-model-estimation)
  - [Volatility Behavior During Crises](#volatility-behavior-during-crises)
  - [GARCH-Based VaR Enhancement](#garch-based-var-enhancement)
- [Findings and Conclusions](#findings-and-conclusions)

---

## Introduction

This project analyzes financial risk through portfolio optimization, Value-at-Risk (VaR) estimation, and volatility modeling. Using historical stock price data from three major Chinese manufacturing companies, we evaluate risk and returns, estimate VaR, and apply GARCH models to enhance risk measurement.

The study follows these steps:
1. Construct an efficient portfolio using Modern Portfolio Theory.
2. Compute and compare VaR using different methodologies.
3. Model volatility dynamics and improve risk estimation with GARCH.

## Dataset and Methodology

The dataset includes historical stock prices for the following three companies:
- **XCMG Machinery (000425.SZ)** - Leading construction equipment manufacturer.
- **LiuGong (000528.SZ)** - Specializes in automation and intelligent manufacturing.
- **Anhui HeLi (600761.SH)** - Prominent forklift manufacturer.

Stock returns are calculated using **log returns**, and risk-free rates are converted to a **daily rate**.

## Question 1: Portfolio Construction

### Summary Statistics of Stocks

| Stock  | Mean (%) | Std.Dev (%) | Max (%) | Min (%) | Skewness | Kurtosis | Obs  |
|--------|---------|------------|---------|---------|----------|----------|------|
| 000425 | 0.042   | 2.87       | 29.23   | -10.60  | 0.316    | 7.67     | 5194 |
| 000528 | 0.036   | 2.81       | 13.30   | -10.58  | -0.025   | 5.25     | 5259 |
| 600761 | 0.037   | 2.65       | 9.59    | -12.09  | -0.121   | 5.59     | 5282 |

### Efficient Frontier and Optimal Portfolio

The **efficient frontier** is constructed by simulating 10,000 random portfolios. The **Sharpe ratio** is used to determine the optimal portfolio allocation:

| Stock  | Weight (%) |
|--------|-----------|
| 000425 | 46.86    |
| 000528 | 11.88    |
| 600761 | 41.26    |

This allocation maximizes the risk-adjusted return.

<img width="371" alt="image" src="https://github.com/user-attachments/assets/e8245500-1fa0-4fc7-85ac-9bea3e96a5bc" />


## Question 2: Value-at-Risk (VaR) Analysis

### VaR Estimation Methods

Two approaches are used to estimate **1-day VaR at 90% confidence**:
1. **Historical Simulation:** Based on past return percentiles.
2. **Variance-Covariance:** Assumes normally distributed returns.

| Method | VaR Estimate (%) |
|--------|------------------|
| Historical Simulation | 2.16 |
| Variance-Covariance | 2.49 |

<img width="382" alt="image" src="https://github.com/user-attachments/assets/5a5b97eb-ee00-4189-a98f-67ac3219c261" />


Results indicate that **Historical Simulation better captures market behavior**, while Variance-Covariance tends to overestimate risk due to normality assumptions.

### Rolling Window VaR and Backtesting

To capture **time-varying risk**, a **rolling-window VaR** is computed. The **Kupiec test** is applied to validate model performance.       

<img width="374" alt="image" src="https://github.com/user-attachments/assets/7af42798-a7d9-43a8-a7f3-25a34a772e96" />


| Method | Observed Breaches | Expected Breaches | P-Value | Result |
|--------|------------------|------------------|--------|--------|
| Historical Simulation | 442 | 408.1 | 0.080 | Fail to Reject |
| Variance-Covariance | 358 | 408.1 | 0.008 | Reject |

Findings:
- **Historical Simulation provides more accurate estimates**, though slightly conservative.
- **Variance-Covariance underestimates risk**, particularly during crises.

### VaR Model Validation

| Asset | Historical Simulation (%) | Variance-Covariance (%) |
|-------|---------------------------|-------------------------|
| XCMG (000425) | 3.05% | 3.63% |
| Liugong (000528) | 3.09% | 3.57% |
| Anhui Heli (600761) | 2.86% | 3.36% |
| **Portfolio** | **2.58%** | **2.99%** |

The portfolio shows **lower risk** than individual assets, demonstrating diversification benefits.

## Question 3: Volatility Modeling Using GARCH

### GARCH(1,1) Model Estimation

A **GARCH(1,1) model** is estimated for each stock to capture volatility clustering:

| Stock  | ω (Long-Term Volatility) | α (Shock Impact) | β (Volatility Persistence) |
|--------|-------------------------|------------------|---------------------------|
| 000425 | 1.23e-6 | 0.089 | 0.901 |
| 000528 | 1.45e-6 | 0.078 | 0.912 |
| 600761 | 1.67e-6 | 0.092 | 0.897 |

Findings:
- **High β values (~0.9) indicate strong volatility persistence.**
- **Volatility spikes occur during crises.**
  
<img width="550" alt="Screenshot 2025-02-26 at 23 06 53" src="https://github.com/user-attachments/assets/4befe195-cb69-478a-8ea4-13cfbdd20076" />

### Volatility Behavior During Crises

Volatility patterns are analyzed across **2008 Financial Crisis, 2015 Stock Crash, and COVID-19 Pandemic**.

| Stock  | Crisis Response |
|--------|----------------|
| 000425 | Highest volatility spikes in crises. |
| 000528 | Moderate volatility increases. |
| 600761 | More stable with lower volatility peaks. |

<img width="361" alt="image" src="https://github.com/user-attachments/assets/1814768a-d1f7-46ed-b9ea-526d25e76d41" />

### GARCH-Based VaR Enhancement

A **GARCH-enhanced rolling-window VaR** model is compared to previous methods.

| Method | Observed Breaches | Expected Breaches | P-Value | Breach Rate (%) |
|--------|------------------|------------------|--------|----------------|
| Historical Simulation | 442 | 408.1 | 0.080 | 10.83 |
| Variance-Covariance | 358 | 408.1 | 0.008 | 8.77 |
| **GARCH-Based** | **354** | **408.1** | **0.004** | **8.67** |

Findings:
- **GARCH-based VaR improves accuracy** by dynamically adjusting to market conditions.
- **Lower breach rates during crises**, making it **more robust** than traditional methods.

<img width="368" alt="image" src="https://github.com/user-attachments/assets/17e5ccad-defa-43cd-aee8-3e8abd4e338f" />


## Findings and Conclusions

1. **Portfolio Optimization:**
   - The **efficient frontier** confirms optimal risk-return trade-offs.
   - **Diversification reduces risk**, as portfolio VaR is lower than individual assets.

2. **VaR Analysis:**
   - **Historical Simulation is more accurate**, but slightly conservative.
   - **Variance-Covariance method underestimates risk**, especially in crises.

3. **Volatility and GARCH Modeling:**
   - **GARCH(1,1) captures volatility clustering effectively.**
   - **Extreme market conditions lead to volatility spikes**, especially in 000425.
   - **GARCH-based VaR provides the most accurate risk estimation.**

This study highlights the importance of **dynamic risk management techniques** and suggests that **GARCH-enhanced VaR models** should be incorporated into risk assessment frameworks.
