import json
import urllib.request
import urllib.parse
import yfinance as yf

# ==========================================
# 1. PYTHON TOOL FUNCTIONS
# ==========================================

def get_crypto_price(coin_id="bitcoin", vs_currency="usd"):
    """
    Fetches real-time crypto prices using CoinGecko's free keyless API.
    Example coin_ids: bitcoin, ethereum, dogecoin, solana, cardano.
    """
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id.lower()}&vs_currencies={vs_currency.lower()}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        if coin_id.lower() in data:
            price = data[coin_id.lower()][vs_currency.lower()]
            return json.dumps({
                "coin": coin_id,
                "currency": vs_currency.upper(),
                "price": price,
                "status": "success"
            })
        else:
            return json.dumps({
                "error": f"Coin '{coin_id}' not found. Ensure full coin IDs are passed (e.g., 'bitcoin', 'ethereum').",
                "status": "failed"
            })
    except Exception as e:
        return json.dumps({"error": str(e), "status": "failed"})


def get_weather(city="Lahore"):
    """
    Fetches live weather using Open-Meteo's free geocoding and forecast API.
    """
    try:
        # Step A: Convert City Name to Lat/Lon via Geocoding
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1&language=en&format=json"
        req_geo = urllib.request.Request(geo_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req_geo) as response:
            geo_data = json.loads(response.read().decode('utf-8'))
            
        if not geo_data.get("results"):
            return json.dumps({"error": f"City '{city}' not found.", "status": "failed"})
            
        location = geo_data["results"][0]
        lat = location["latitude"]
        lon = location["longitude"]
        city_name = location["name"]
        country = location.get("country", "")

        # Step B: Fetch Current Weather
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        req_weather = urllib.request.Request(weather_url, headers={'User-Agent': 'Mozilla/5.0'})
        
        with urllib.request.urlopen(req_weather) as response:
            w_data = json.loads(response.read().decode('utf-8'))
            
        current = w_data.get("current_weather", {})
        temp = current.get("temperature")
        windspeed = current.get("windspeed")
        
        return json.dumps({
            "city": city_name,
            "country": country,
            "temperature_celsius": temp,
            "windspeed_kmh": windspeed,
            "status": "success"
        })
    except Exception as e:
        return json.dumps({"error": str(e), "status": "failed"})


def get_stock_price(ticker="AAPL"):
    """
    Fetches real-time stock price data using Yahoo Finance (yfinance).
    Examples: AAPL, TSLA, NVDA, GOOGL, MSFT, AMZN.
    """
    try:
        stock = yf.Ticker(ticker.upper())
        # Fast 1-day history check to pull the latest available closing price
        hist = stock.history(period="1d")
        
        if not hist.empty:
            latest_price = float(hist["Close"].iloc[-1])
            currency = stock.info.get("currency", "USD")
            company_name = stock.info.get("shortName") or ticker.upper()

            return json.dumps({
                "ticker": ticker.upper(),
                "company_name": company_name,
                "price": round(latest_price, 2),
                "currency": currency,
                "status": "success"
            })
        else:
            return json.dumps({"error": f"Ticker symbol '{ticker}' not found or has no market data.", "status": "failed"})
    except Exception as e:
        return json.dumps({"error": str(e), "status": "failed"})


# ==========================================
# 2. MAP FUNCTIONS TO DICTIONARY FOR EXECUTION
# ==========================================

AVAILABLE_TOOLS = {
    "get_crypto_price": get_crypto_price,
    "get_weather": get_weather,
    "get_stock_price": get_stock_price
}


# ==========================================
# 3. GROQ / OPENAI COMPATIBLE TOOL SCHEMAS
# ==========================================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_crypto_price",
            "description": "Fetch current live price of any cryptocurrency (e.g. bitcoin, ethereum, solana, dogecoin).",
            "parameters": {
                "type": "object",
                "properties": {
                    "coin_id": {
                        "type": "string",
                        "description": "The full CoinGecko ID of the cryptocurrency in lowercase (e.g., 'bitcoin', 'ethereum', 'solana', 'dogecoin')."
                    },
                    "vs_currency": {
                        "type": "string",
                        "description": "The target currency to compare against (e.g., 'usd', 'eur', 'gbp'). Default is 'usd'."
                    }
                },
                "required": ["coin_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Fetch real-time current weather information for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city (e.g., 'Lahore', 'London', 'Tokyo', 'New York')."
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Fetch current live stock market price for a company using its stock ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "The stock market ticker symbol (e.g., 'AAPL' for Apple, 'TSLA' for Tesla, 'NVDA' for Nvidia, 'GOOGL' for Google)."
                    }
                },
                "required": ["ticker"]
            }
        }
    }
]