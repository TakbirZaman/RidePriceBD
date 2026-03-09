# 🛺 RidePrice BD

> Know your fare before you ride.

A ride fare estimator for Dhaka, Bangladesh. Instantly calculates expected costs for Rickshaw, Bike, Car, CNG, and Bus across 76 locations — with a full ML-powered price breakdown.

## ✨ Features

- **Instant calculation** — runs entirely in the browser, no server delay
- **5 vehicle types** — Rickshaw, Bike (Pathao/Uber Moto), Car, CNG, Bus
- **76 Dhaka locations** — sorted by distance from Jagannath University
- **Real Dhaka rates** — base fare + per km + per minute traffic charge
- **ML Price Breakdown** — shows exactly how distance, weather, time, and demand affect your fare
- **Smart conditions** — rush hour, late night, Friday discount, rain/storm surge
- **Price confidence range** — ±8% model uncertainty window

## 🚀 Live Demo

👉 [ridepricebd.netlify.app](https://ridepricebd.netlify.app)

## 🧠 How It Works

The fare engine is a trained `GradientBoostingRegressor` model (scikit-learn) with features:

| Feature | Description |
|---|---|
| Distance | Haversine × 1.35 road factor |
| Vehicle | Rickshaw / Bike / Car / Bus / CNG |
| Weather | Sunny / Regular / Rainy / Stormy |
| Time of day | Hour + rush/night detection |
| Day of week | Friday discount applied |
| Demand | Regular / Busy Hour / Emergency |

## 🛠 Tech Stack

| Layer | Tech |
|---|---|
| ML Model | Python, scikit-learn, pandas |
| API | Flask + Gunicorn |
| Frontend | Vanilla HTML/CSS/JS |
| Deployment | Render (API) + Netlify (frontend) |

`

## 👤 Author

**Takbir Zaman Bhuiyan**  
[LinkedIn](https://www.linkedin.com/in/takbir-zaman-bhuiyan)
```
