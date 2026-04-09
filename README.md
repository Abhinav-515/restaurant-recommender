# Predictive Restaurant Recommender
### Soulpage IT Solutions – Data Scientist Intern Assignment

---

## 📁 Project Structure

```
restaurant_project/
├── restaurant_recommender.py   ← main script
├── requirements.txt            ← dependencies
├── README.md                   ← this file
├── data/
│   ├── Train/
│   │   ├── orders.csv
│   │   ├── train_customers.csv
│   │   ├── train_locations.csv
│   │   └── vendors.csv
│   └── Test/
│       ├── test_customers.csv
│       └── test_locations.csv
└── output/
    └── submission.csv          ← generated after running
```

---

## ⚙️ Setup & Run (Any Device)

### 1. Install Python
Make sure Python 3.8 or above is installed.
Download from: https://www.python.org/downloads/

### 2. Install dependencies
Open terminal / command prompt in the project folder and run:
```bash
pip install -r requirements.txt
```

### 3. Run the script
```bash
python restaurant_recommender.py
```

The submission file will be saved at `output/submission.csv`.

---

## 🧠 Approach

| Step | Description |
|------|-------------|
| Pre-processing | Fix data types, encode gender, remove GPS outliers |
| Feature Engineering | Customer history, vendor popularity, distance features |
| Model | Random Forest Classifier (100 trees) |
| Cold-start | New customers → recommend top 20% popular vendors |
| Validation | ROC-AUC: **0.9959** |

---

## 📊 Key Features Used

- **Customer × Vendor order history** — strongest signal (prior orders)
- **Vendor reach %** — what fraction of customers ordered from each vendor
- **Distance** — Euclidean distance from customer location to vendor
- **Customer demographics** — gender, age, status
- **Vendor attributes** — rating, delivery charge, category, prep time
