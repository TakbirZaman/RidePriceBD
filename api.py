# api.py — python api.py

from flask import Flask, request, jsonify, make_response
import joblib, pandas as pd, math
from datetime import datetime

app      = Flask(__name__)
model    = joblib.load("model.pkl")
LOCS     = joblib.load("locations.pkl")
VEHICLES = joblib.load("vehicles.pkl")

def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return r

@app.after_request
def after(r): return cors(r)

@app.before_request
def options():
    if request.method == "OPTIONS":
        return cors(make_response()), 200

def calc_distance(loc1, loc2):
    lat1, lon1 = LOCS[loc1]
    lat2, lon2 = LOCS[loc2]
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return round(math.sqrt(a) * R * 2 * 1.35, 1)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/locations")
def locations():
    return jsonify({"locations": list(LOCS.keys())})

@app.route("/predict", methods=["POST"])
def predict():
    d = request.json

    from_loc = d["from_loc"]
    to_loc   = d["to_loc"]
    vehicle  = d["vehicle"]
    weather  = d["weather"]
    hour     = int(d["hour"])
    demand   = float(d["demand"])
    dow      = datetime.now().weekday()
    dist     = calc_distance(from_loc, to_loc)

    row = pd.DataFrame([{
        "from_loc":    from_loc,
        "to_loc":      to_loc,
        "vehicle":     vehicle,
        "weather":     weather,
        "hour":        hour,
        "dow":         dow,
        "demand":      demand,
        "distance_km": dist,
    }])

    price = round(float(model.predict(row)[0]), 0)
    price = max(VEHICLES[vehicle]["base"], price)

    # Surge tiers differ by vehicle
    thresholds = {
        "Rickshaw": [80,  120, 160],
        "Bike":     [100, 160, 220],
        "Car":      [180, 300, 420],
        "Bus":      [40,  70,  100],
    }
    t = thresholds[vehicle]
    if   price < t[0]: surge = ("Normal",   "green")
    elif price < t[1]: surge = ("Moderate", "yellow")
    elif price < t[2]: surge = ("High",     "orange")
    else:              surge = ("Peak",     "red")

    return jsonify({
        "price":       price,
        "distance_km": dist,
        "surge_label": surge[0],
        "surge_color": surge[1],
    })

if __name__ == "__main__":
    print("\n🚗 RidePrice BD API running → http://localhost:5000\n")
    app.run(port=5000, debug=False)
