from flask import Flask, request, jsonify,render_template 
import numpy as np
from scipy.stats import norm
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import tensorflow as tf
from pandas_datareader import data as pdr


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)


def get_stock_price(symbol):
    try:
        # sym = yf.Ticker(symbol)
        # hist = sym.history(period='3mo')
        # if hist.empty:
        #     raise ValueError("No historical stock price data found for this symbol.")
        # S = hist['Close'][-1]
        # return S,hist['Close']
        yf.pdr_override() # <== that's all it takes :-)

        # download dataframe
        data = pdr.get_data_yahoo(symbol,start = dt.datetime.today() - dt.timedelta(days = 60))
        return data.iloc[:,3][-1],data.index.values,data.iloc[:,3]
        return data
    except (ValueError, KeyError, IndexError) as e:
        raise ValueError("Error fetching stock price data: " + str(e))

#print(get_stock_price('AMZN')[1])
    
def black_scholes_analytic(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Use 'call' or 'put'.")

    return option_price

def ai_model(S, K, T, sigma, r=0.01):
    T = T * 365
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    model.load_weights('my_model_weights.h5')
    
    k = (K - 115.886382) / 380.716775
    t = (T - 187.674280) / 234.793258
    s = (S - 115.886382) / 135.569885
    i_v = (sigma - 0.165873) / 0.697085
    
    v = model.predict(np.array([[k, t, s, i_v]]))
    V = (v * 100.04787534116524) + 33.96399613737509
    
    return V[0, 0]

#print(ai_model(100, 150, 1, 1.5))

@app.route("/")
def home():
    labels = []
 
    results = []
 
    # Return the components to the HTML template 
    return render_template(
        template_name_or_list='app.html',
        results=results,
        labels=labels,
    )

@app.route('/api/call_option_price', methods=['POST'])
def get_call_option_price():
    data = request.get_json()

    Symbol = data.get('Symbol')  # Current stock price
    K = data.get('K')  # Strike price
    T = data.get('T')  # Time to expiration
    r = data.get('r')  # Interest rate
    sigma = data.get('sigma')  # Volatility

    option_type = 'call'  # Default to call option
    if 'option_type' in data:
        option_type = data['option_type']
    try:
        Symbol = str(Symbol)
        S = get_stock_price(Symbol)[0]
        if S is None:
            raise ValueError("Stock price data is not available for this symbol.")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    
    inital_date = dt.datetime.now()
    expiry_date = dt.datetime.strptime(T, "%Y-%m-%d")
    date_diff = expiry_date - inital_date
    TT = round((date_diff.days/365),2)
    
    # Validate input data
    if None in [S, K, T, r, sigma]:
        return jsonify({"error": "Missing input data"}), 400

    try:
        # Convert inputs to appropriate types
        S = float(S)
        K = float(K)
        TT = float(TT)
        r = float(r)
        sigma = float(sigma)
    except ValueError:
        return jsonify({"error": "Invalid input data"}), 400

    # Calculate the call option price using the Black-Scholes function
    
    try:
        call_option_price = black_scholes_analytic(S, K, TT, r, sigma, option_type='call')
        ai_option_price = ai_model(S, K, TT, sigma)
        print(call_option_price)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    # labels_num = np.linspace(0,TT,5)
    # labels = [(inital_date.today() + dt.timedelta(days = year*365)).strftime("%d-%m-%Y")for year in labels_num]
    # results = black_scholes_analytic(S, K, labels_num, r, sigma, option_type='call')
    labels = get_stock_price(Symbol)[1].tolist
    results = get_call_option_price(Symbol)[2]
    return jsonify({"call_option_price_bs": round(call_option_price,2),
                    "results":list(results),
                    "labels":list(labels),
                    "Stock_Price":round(S),
                    "call_option_price_ai":ai_option_price})

if __name__ == '__main__':
    app.run(debug=True)


# # # Example usage for a call option
# # S0 = 100  # Current stock price
# # K = 100   # Strike price
# # T = 1     # Time to expiration (in years)
# # r = 0.05  # Risk-free interest rate
# # sigma = 0.2  # Volatility

# # call_price_analytic = black_scholes_analytic(S0, K, T, r, sigma, option_type='call')
