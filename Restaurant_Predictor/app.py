from flask import Flask, render_template, request
import pandas as pd
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter

# ✅ NEW Gemini SDK
from google import genai

# 🔴 ADD YOUR KEY HERE
client = genai.Client(api_key="AIzaSyAh5wdUqpHcW-fbJqwWU9ZO6zr2D1rgQmg")


class MultiLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=30):
        self.mlb = MultiLabelBinarizer()
        self.min_freq = min_freq

    def clean_split(self, x):
        if isinstance(x, str):
            return [i.strip().lower() for i in x.split(",") if i.strip()]
        elif isinstance(x, list):
            return [str(i).strip().lower() for i in x if str(i).strip()]
        return []

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif isinstance(X, list):
            X = pd.Series(X)
        elif isinstance(X, str):
            X = pd.Series([X])

        X_clean = X.apply(self.clean_split)

        all_items = [item for sublist in X_clean for item in sublist]
        counts = Counter(all_items)

        self.frequent_classes = {k for k, v in counts.items() if v >= self.min_freq}

        X_filtered = X_clean.apply(
            lambda lst: [i if i in self.frequent_classes else "other" for i in lst]
        )

        self.mlb.fit(X_filtered)
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0]
        elif isinstance(X, list):
            X = pd.Series(X)
        elif isinstance(X, str):
            X = pd.Series([X])

        X_clean = X.apply(self.clean_split)

        X_filtered = X_clean.apply(
            lambda lst: [i if i in self.frequent_classes else "other" for i in lst]
        )

        return self.mlb.transform(X_filtered)

    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_


app = Flask(__name__)
model = joblib.load("model.pkl")


def safe_float(value, default=0):
    try:
        return float(value)
    except:
        return default


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        cuisine_str = data.get("cuisine", "")
        cuisine_list = [c.strip() for c in cuisine_str.split(",") if c.strip()]
        cuisine_count = len(cuisine_list)

        df = pd.DataFrame([{
            "city": data.get("city"),
            "price_category": data.get("price_category"),
            "cost_log": safe_float(data.get("cost")),
            "cuisine_count": cuisine_count,
            "city_population": safe_float(data.get("city_population")),
            "city_area_km": safe_float(data.get("city_area_km")),
            "order_online": int(data.get("order_online", 0)),
            "book_table": int(data.get("book_table", 0)),
            "delivery_available": int(data.get("delivery_available", 0)),
            "dinein_available": int(data.get("dinein_available", 0)),
            "tourism_flag": int(data.get("tourism_flag", 0)),
            "gdp_flag": int(data.get("gdp_flag", 0)),
            "cuisine": cuisine_str
        }])

        rating = model.predict(df)[0]
        rating_str = f"{rating:.2f}"
        rating_val = float(rating_str)

        # ✅ Suggestion logic
        if rating_val >= 4.2:
            suggestion_text = "✅ Great! Your restaurant is already performing well."
        else:
            suggestion_text = generate_suggestions(data, rating_val)

        return render_template(
            "index.html",
            prediction=f"Rating: {rating_str}",
            suggestions=suggestion_text
        )

    except Exception as e:
        return render_template("index.html", error=str(e))


# ✅ Gemini function (FIXED MODEL)
def generate_suggestions(data, rating):
    try:
        prompt = f"""
        Restaurant details:
        City: {data.get('city')}
        Price Category: {data.get('price_category')}
        Cost: {data.get('cost')}
        Cuisines: {data.get('cuisine')}
        Online Order: {data.get('order_online')}
        Table Booking: {data.get('book_table')}
        Delivery: {data.get('delivery_available')}
        Dine-in: {data.get('dinein_available')}

        Current rating: {rating}

        Give 4 short practical suggestions to improve rating.
        Use bullet points.
        """

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )

        text = response.text.strip()
        return "\n".join([line.strip() for line in text.split("\n") if line.strip()])

    except Exception as e:
        print("Gemini Error:", e)

        # ✅ SMART FALLBACK (AI-like logic)
        suggestions = []

        if int(data.get("order_online", 0)) == 0:
            suggestions.append("Enable online ordering to increase accessibility")

        if int(data.get("delivery_available", 0)) == 0:
            suggestions.append("Start delivery service to reach more customers")

        if int(data.get("book_table", 0)) == 0:
            suggestions.append("Allow table reservations for better planning")

        if int(data.get("dinein_available", 0)) == 0:
            suggestions.append("Improve dine-in experience and seating comfort")

        if data.get("price_category") == "low":
            suggestions.append("Enhance food quality while maintaining pricing")

        if not data.get("cuisine"):
            suggestions.append("Add diverse cuisine options to attract customers")

        # ensure 4 suggestions
        if len(suggestions) < 4:
            suggestions += [
                "Improve service speed and staff responsiveness",
                "Enhance ambience, lighting, and cleanliness",
                "Offer discounts, combos, and loyalty programs"
            ]

        return "• " + "\n• ".join(suggestions[:4])


if __name__ == "__main__":
    app.run(debug=True)