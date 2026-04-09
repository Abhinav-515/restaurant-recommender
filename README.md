# 🍽️ Predictive Restaurant Recommendation System

## 📌 Objective
Build a recommendation engine to predict which restaurants customers are most likely to order from based on customer data, vendor information, and order history.

---

## 🧠 Approach

### Data Processing
- Merged customer, location, order, and vendor datasets
- Handled missing values and cleaned data

### Feature Engineering
- Customer order frequency
- Vendor popularity
- Customer–vendor interaction history
- Delivery and behavioral features

### Model
- XGBoost Classifier used for prediction

---

## 📊 Output
- Generated submission file with probability scores for each (Customer, Location, Vendor) combination

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python restaurant_recommender.py
