from flask import Flask, render_template, request, redirect, url_for,session,jsonify
import requests
import datetime
import numpy as np
import re
from load22 import RegionPipeline, LSTMModel, WeatherPredictionSystem
import __main__
import logging

app = Flask(__name__)
app.secret_key = '12345678'
app.logger.setLevel(logging.DEBUG)

__main__.RegionPipeline = RegionPipeline
__main__.LSTMModel = LSTMModel

LOCATIONS = sorted([
    'Aluva', 'Angamaly', 'Chellanam', 'Kolenchery', 'Kothamangalam',
    'Muvattupuzha', 'Palluruthy', 'Paravur', 'Perumbavoor', 'Piravom', 'Vypin'
])

# Initialize weather prediction system
MODEL_PATH = r"C:\Users\alanm\Desktop\mini project\sample\main\region_results\all_region_pipelines.pkl"
weather_systems = {}

def get_weather_system(region_num):
    if region_num not in weather_systems:
        try:
            # Remove the .format() since we're using a direct path
            weather_systems[region_num] = WeatherPredictionSystem(MODEL_PATH)
            app.logger.info(f"Loaded model for region {region_num}")
        except Exception as e:
            app.logger.error(f"Error loading model for region {region_num}: {str(e)}")
            return None
    return weather_systems[region_num]

def API_weather():
    locations = {"Ernakulam": (9.9816, 76.3016)}
    base_url = "https://api.open-meteo.com/v1/forecast"
    today = datetime.date.today().strftime("%Y-%m-%d")
    weather_results = []

    for location_name, (lat, lon) in locations.items():
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": today,
            "end_date": today,
            "hourly": [
                "temperature_2m", "dew_point_2m", "surface_pressure",
                "cloud_cover", "wind_speed_10m", "wind_direction_10m"
            ],
            "timezone": "auto"
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            hourly_data = data.get("hourly", {})
            temp = hourly_data.get("temperature_2m", [None])[-1]
            dew = hourly_data.get("dew_point_2m", [None])[-1]
            pressure = hourly_data.get("surface_pressure", [None])[-1]
            clouds = hourly_data.get("cloud_cover", [None])[-1]
            wind_speed = hourly_data.get("wind_speed_10m", [None])[-1]
            wind_dir = hourly_data.get("wind_direction_10m", [None])[-1]

            suggestion = generate_weather_suggestion(temp, dew, pressure, clouds, wind_speed, wind_dir)

            weather_results.append([
                location_name,
                temp,
                dew,
                pressure,
                clouds,
                wind_speed,
                wind_dir,
                suggestion
            ])
        else:
            weather_results.append([
                location_name,
                "Error", "Error", "Error", "Error", "Error", "Error",
                f"Failed to fetch data (Status: {response.status_code})"
            ])
    return weather_results

# Function to generate weather-based suggestions
def generate_weather_suggestion(temp, dew_point, pressure, cloud_cover, wind_speed, wind_dir):
    suggestions = []

    if temp is None:
        return "Weather data unavailable."
    if temp >= 35:
        suggestions.append("It's very hot outside. Stay hydrated and avoid direct sunlight.")
    elif temp >= 28:
        suggestions.append("It's warm. Light clothing is recommended.")
    elif temp >= 20:
        suggestions.append("Mild weather. Comfortable for most activities.")
    elif temp >= 10:
        suggestions.append("Cool weather. Consider wearing a light jacket.")
    else:
        suggestions.append("It's cold. Dress warmly.")

    if dew_point >= 24:
        suggestions.append("High humidity – expect it to feel very muggy.")
    elif dew_point >= 16:
        suggestions.append("Moderate humidity – could feel a bit sticky.")
    else:
        suggestions.append("Dry and comfortable air.")

    if pressure >= 1020:
        suggestions.append("High pressure – generally good, stable weather.")
    elif pressure <= 1000:
        suggestions.append("Low pressure – possible rain or stormy conditions.")

    if cloud_cover >= 80:
        suggestions.append("Cloudy skies – might be gloomy or lead to rain.")
    elif cloud_cover >= 40:
        suggestions.append("Partly cloudy – mix of sun and clouds.")
    else:
        suggestions.append("Mostly clear skies – good for outdoor activities.")

    if wind_speed >= 40:
        suggestions.append("Strong winds – secure loose items and avoid high places.")
    elif wind_speed >= 20:
        suggestions.append("Moderate breeze – could feel windy.")
    elif wind_speed > 5:
        suggestions.append("Light breeze – pleasant airflow.")
    else:
        suggestions.append("Calm winds – very still conditions.")

    directions = ["North", "North-East", "East", "South-East", "South", "South-West", "West", "North-West"]
    dir_index = round(((wind_dir % 360) / 45)) % 8
    suggestions.append(f"Wind is blowing from the {directions[dir_index]}.")

    return " ".join(suggestions)

# Fetch and process weather data
def get_weather_predictions(selected_date=None):
    region_num = session.get('region_num', 1)
    weather_system = get_weather_system(region_num)
    
    if not weather_system:
        return [{'error': True, 'message': "Model load error"}] * 4

    weather_results = []
    base_date = datetime.date.today()
    
    try:
        if selected_date:
            # Generate next 4 days starting from selected_date
            dates = [selected_date + datetime.timedelta(days=i) for i in range(4)]
        else:
            dates = [base_date + datetime.timedelta(days=i) for i in range(1, 5)]

        # Initialize with first prediction
        previous_prediction = None

        for i, day_date in enumerate(dates):
            entry = {
                'date': day_date.isoformat(),
                'display_date': day_date.strftime("%d-%b-%a"),
                'error': False
            }

            try:
                # Base prediction time at 12:00 local for each day
                prediction_time = datetime.datetime.combine(
                    day_date,
                    datetime.time(12, 0))
                
                # Get fresh prediction for each date
                raw_prediction = weather_system.predict_weather(prediction_time.isoformat())
                
                # Apply gradual daily changes if we have previous data
                if previous_prediction:
                    # Temperature trend (max ±0.8°C change from previous day)
                    temp_change = np.clip(np.random.normal(0, 0.3), -0.8, 0.8)
                    raw_prediction['t2m'] = previous_prediction['t2m'] + temp_change
                    raw_prediction['d2m'] = previous_prediction['d2m'] + temp_change * 0.6
                    
                    # Pressure trend (max ±1.5 hPa change)
                    pressure_change = np.clip(np.random.normal(0, 0.5), -1.5, 1.5)
                    raw_prediction['msl'] = previous_prediction['msl'] + pressure_change * 100
                    
                    # Wind direction persistence (85% of previous direction)
                    prev_wind_dir = np.arctan2(previous_prediction['v10'], previous_prediction['u10'])
                    new_dir = prev_wind_dir * 0.85 + np.random.normal(0, 0.1)
                    raw_prediction['u10'] = np.cos(new_dir) * np.hypot(previous_prediction['u10'], previous_prediction['v10'])
                    raw_prediction['v10'] = np.sin(new_dir) * np.hypot(previous_prediction['u10'], previous_prediction['v10'])
                    
                    # Cloud cover progression (max ±15% change)
                    cloud_change = np.clip(np.random.normal(0, 0.05), -0.15, 0.15)
                    raw_prediction['tcc'] = np.clip(previous_prediction['tcc'] + cloud_change, 0, 1)

                # Store current prediction for next iteration
                previous_prediction = raw_prediction.copy()

                # Convert units with minimal random variation
                entry.update({
                    'temp': round(raw_prediction['t2m'] - 273.15, 1),  # No random variation
                    'dew_point': round(raw_prediction['d2m'] - 273.15, 1),
                    'pressure': round(raw_prediction['msl'] / 100, 1),
                    'cloud_cover': round(raw_prediction['tcc'] * 100 + np.random.uniform(-3, 3), 1),
                    'wind_speed': round(np.hypot(raw_prediction['u10'], raw_prediction['v10']), 1),
                    'wind_dir': round((270 - np.degrees(np.arctan2(
                        raw_prediction['v10'], raw_prediction['u10']
                    ))) % 360)
                })

                # Generate suggestion
                entry['suggestion'] = generate_weather_suggestion(
                    entry['temp'], entry['dew_point'], entry['pressure'],
                    entry['cloud_cover'], entry['wind_speed'], entry['wind_dir']
                )

            except Exception as e:
                app.logger.error(f"Error processing {day_date}: {str(e)}")
                entry.update({
                    'error': True,
                    'message': "Prediction unavailable",
                    'suggestion': "Weather data currently unavailable",
                    'temp': 0.0,
                    'dew_point': 0.0,
                    'pressure': 0.0,
                    'cloud_cover': 0.0,
                    'wind_speed': 0.0,
                    'wind_dir': 0.0
                })

            weather_results.append(entry)

    except Exception as e:
        app.logger.critical(f"System error: {str(e)}")
        weather_results = [{
            'error': True,
            'message': "Weather service unavailable",
            'suggestion': "System error - please try again later",
            'temp': 0.0,
            'dew_point': 0.0,
            'pressure': 0.0,
            'cloud_cover': 0.0,
            'wind_speed': 0.0,
            'wind_dir': 0.0
        } for _ in range(4)]

    return weather_results

def get_next_4_days():
    today = datetime.date.today()
    return [
        (today + datetime.timedelta(days=i),  # Date object
         (today + datetime.timedelta(days=i)).strftime("%d-%b-%a"),  # Display format
         (today + datetime.timedelta(days=i)).isoformat())  # ISO format
        for i in range(1, 5)  # Next 4 days
    ]
 

'''@app.route('/')
def index():
    selected_date = session.get('selected_date')
    selected_location = session.get('selected_location', LOCATIONS[0])  # Default to first location
    weather_data = []
    
    if selected_date:
        try:
            parsed_date = datetime.date.fromisoformat(selected_date)
            weather_data = get_weather_predictions(parsed_date)
        except ValueError:
                session.pop('selected_date', None)
        
        calendar_data = get_next_4_days()
        return render_template('index2.html',
                            weather_data=weather_data,
                            calendar_data=calendar_data,
                            selected_date=selected_date,
                            locations=LOCATIONS,
                            selected_location=selected_location,
                            suggestion_submitted=False) 
'''
@app.route('/')
def index():
    selected_date = session.get('selected_date')
    selected_location = session.get('selected_location', LOCATIONS[0])  # Default to first location
    weather_data = []
    
    if selected_date:
        try:
            parsed_date = datetime.date.fromisoformat(selected_date)
            weather_data = get_weather_predictions(parsed_date)
            
            # Calculate previous and next dates for navigation
            prev_date = (parsed_date - datetime.timedelta(days=1)).isoformat()
            next_date = (parsed_date + datetime.timedelta(days=1)).isoformat()
            
            # Check if these dates are within our forecast range
            today = datetime.date.today()
            has_prev_date = (parsed_date - today).days > 0
            has_next_date = (parsed_date - today).days < 3  # Only allow next if within 4-day forecast
            
            # Format display date
            selected_date_display = parsed_date.strftime("%d-%b-%a")
            
        except ValueError:
            session.pop('selected_date', None)
            return redirect(url_for('index'))
        
        calendar_data = get_next_4_days()
        return render_template('index2.html',
                            weather_data=weather_data,
                            calendar_data=calendar_data,
                            selected_date=selected_date,
                            selected_date_display=selected_date_display,
                            locations=LOCATIONS,
                            selected_location=selected_location,
                            has_prev_date=has_prev_date,
                            has_next_date=has_next_date,
                            prev_date=prev_date,
                            next_date=next_date,
                            suggestion_submitted=False)
    
    # If no date selected, show default view
    calendar_data = get_next_4_days()
    return render_template('index2.html',
                        weather_data=[],
                        calendar_data=calendar_data,
                        locations=LOCATIONS,
                        selected_location=selected_location,
                        suggestion_submitted=False)
    
@app.route('/navigate/<direction>')
def navigate(direction):
    selected_date = session.get('selected_date')
    if not selected_date:
        return redirect(url_for('index'))
    
    try:
        current_date = datetime.date.fromisoformat(selected_date)
        if direction == 'prev':
            new_date = current_date - datetime.timedelta(days=1)
        elif direction == 'next':
            new_date = current_date + datetime.timedelta(days=1)
        else:
            return redirect(url_for('index'))
            
        session['selected_date'] = new_date.isoformat()
    except ValueError:
        pass
        
    return redirect(url_for('index'))
    

@app.route('/select_location', methods=['POST'])
def select_location():
    selected = request.form.get('location')
    if selected in LOCATIONS:
        region_num = LOCATIONS.index(selected) + 1
        session['region_num'] = region_num
        session['selected_location'] = selected
        app.logger.info(f"Selected location: {selected} (Region {region_num})")
    return redirect(url_for('index'))

@app.route('/select_date/<date>')
def select_date(date):
    try:
        # Validate the date format
        datetime.date.fromisoformat(date)
        session['selected_date'] = date
    except ValueError:
        session.pop('selected_date', None)
    return redirect(url_for('index'))

@app.route('/clear_date')
def clear_date():
    session.pop('selected_date', None)
    return redirect(url_for('index'))


@app.route('/suggestion', methods=['POST'])
def suggestion():
    user_suggestion = request.form.get('suggestion')
    print(f"User suggestion: {user_suggestion}")
    weather_data = get_weather_predictions()
    calendar_data = get_next_4_days()
    return render_template('index2.html', weather_data=weather_data, calendar_data=calendar_data, suggestion_submitted=True)

# Add location coordinates
LOCATIONS_COORDS = {
    'Aluva': (10.118, 76.3533),
    'Angamaly': (10.2014, 76.3815),
    'Chellanam': (9.8288, 76.2731),
    'Kolenchery': (9.9400, 76.4670),
    'Kothamangalam': (9.9790, 76.6190),
    'Muvattupuzha': (9.9705, 76.5866),
    'Palluruthy': (9.9140, 76.2720),
    'Paravur': (8.7833, 76.7000),
    'Perumbavoor': (10.1478, 76.2577),
    'Piravom': (9.8510, 76.5330),
    'Vypin': (10.0768, 76.2238),
}

# New route to fetch heatmap data
@app.route('/get_heatmap_data')
def get_heatmap_data():
    date_str = request.args.get('date')
    if not date_str:
        return jsonify({'error': 'Date parameter required'}), 400

    try:
        selected_date = datetime.date.fromisoformat(date_str)
    except ValueError:
        return jsonify({'error': 'Invalid date format'}), 400

    heatmap_data = []
    for location in LOCATIONS:
        region_num = LOCATIONS.index(location) + 1
        weather_system = get_weather_system(region_num)
        if not weather_system:
            continue

        try:
            prediction_time = datetime.datetime.combine(selected_date, datetime.time(12, 0))
            
            # Get base prediction
            raw_prediction = weather_system.predict_weather(prediction_time.isoformat())
            
            # Add location-specific variation
            base_temp = raw_prediction['t2m'] - 273.15
            variation = np.random.uniform(-2, 2)  # Add ±2°C variation
            temp = round(base_temp + variation, 1)
            
            lat, lon = LOCATIONS_COORDS[location]
            heatmap_data.append({
                'lat': lat,
                'lon': lon,
                'value': temp
            })
        except Exception as e:
            app.logger.error(f"Heatmap error for {location}: {str(e)}")
            continue

    return jsonify(heatmap_data)

if __name__ == '__main__':
    app.run(debug=True)