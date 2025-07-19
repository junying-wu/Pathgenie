import streamlit as st
import pandas as pd
from datetime import datetime, time, timedelta
import random
import requests
import os
import re
from PIL import Image
from datetime import datetime


st.set_page_config(page_title="AI Travel Agent", layout="centered")


@st.cache_data
def load_data():
    file_path = "current path with the location of the CSV file"
    
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            st.success(f"Successfully loaded data with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Error with {encoding} encoding: {str(e)}")
            continue
    else:
        # If all encodings fail, try with error handling
        try:
            df = pd.read_csv(file_path, encoding='utf-8', errors='ignore')
            st.warning("Loaded data with UTF-8 encoding, ignoring problematic characters")
        except Exception as e:
            st.error(f"Failed to load data with all attempted encodings: {str(e)}")
            return pd.DataFrame()
    
    # Check if required columns exist
    required_columns = ['name', 'address', 'city', 'categories', 'stars', 'Monday_hours', 'Tuesday_hours', 'Wednesday_hours', 'Thursday_hours','Friday_hours','Saturday_hours', 'Sunday_hours', 'tip']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"Data file missing required columns: {missing_columns}")
        return pd.DataFrame()
    
    # Display data info
    st.sidebar.success(f"Data loaded successfully: {len(df)} records")
    
    return df

# Initialize data
try:
    df = load_data()
    if df.empty:
        st.error("Failed to load data. Please check the file path and format.")
        st.stop()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Helper functions for text processing
def normalize_city_name(city_input):
    """Normalize city name capitalization"""
    if not city_input:
        return ""
    
    # Common city name mappings
    city_mapping = {
        'new york': 'New York',
        'los angeles': 'Los Angeles',
        'san francisco': 'San Francisco',
        'las vegas': 'Las Vegas',
        'washington dc': 'Washington DC',
        'boston': 'Boston',
        'chicago': 'Chicago',
        'miami': 'Miami',
        'seattle': 'Seattle',
        'philadelphia': 'Philadelphia'
    }
    
    city_lower = city_input.lower().strip()
    
    # Check if in mapping table
    if city_lower in city_mapping:
        return city_mapping[city_lower]
    
    # Capitalize each word for English city names
    return ' '.join(word.capitalize() for word in city_input.split())

def find_matching_places(must_visit_input, df, city):
    """Find if user input must-visit places exist in database"""
    if not must_visit_input.strip():
        return [], []
    
    must_visits = [place.strip() for place in must_visit_input.split(',') if place.strip()]
    found_places = []
    not_found_places = []
    
    # Filter by city first
    city_df = df[df['city'].str.lower() == city.lower()]
    
    for place in must_visits:
        pattern = r'\b' + re.escape(place.lower()) + r'\b'
        matches = city_df[city_df['name'].str.contains(pattern, case=False, na=False)]
        if not matches.empty:
            # Get the best match(highest rated)
            best_match = matches.loc[matches['stars'].idxmax()]
            found_places.append({
                'input': place,
                'matched_place': best_match.to_dict(),  
                'all_matches': matches[['name', 'categories', 'stars']].to_dict('records')
            })
        else:
            not_found_places.append(place)
    
    return found_places, not_found_places


def extract_time_preferences(preferences_input):
    """Extract comprehensive time preferences from user input"""
    if not preferences_input:
        return time(9, 0), time(21, 0)  # Default: 9 AM start, 9 PM end
    
    text = preferences_input.lower()
    
    # Initialize default times
    start_time = time(9, 0)
    end_time = time(21, 0)
    
    # Extract start time preferences
    start_patterns = {
        'early morning': time(8, 0),
        'morning': time(9, 0),
        'late morning': time(10, 0),
        'noon': time(12, 0),
        'midday': time(12, 0),
        'afternoon': time(14, 0),
        'early afternoon': time(13, 0),
        'late afternoon': time(16, 0),
        'start at noon': time(12, 0),
        'leave at noon': time(12, 0),
        'depart at noon': time(12, 0),
        'begin at noon': time(12, 0),
        'start at 10': time(10, 0),
        'start at 11': time(11, 0),
        'start at 9': time(9, 0),
        'start at 8': time(8, 0)
    }
    
    # Extract end time preferences
    end_patterns = {
        'end before 9 pm': time(21, 0),
        'end before 8 pm': time(20, 0),
        'end before 7 pm': time(19, 0),
        'end before 10 pm': time(22, 0),
        'finish before 9 pm': time(21, 0),
        'finish before 8 pm': time(20, 0),
        'finish before 7 pm': time(19, 0),
        'finish before 10 pm': time(22, 0),
        'done before 9 pm': time(21, 0),
        'done before 8 pm': time(20, 0),
        'done before 7 pm': time(19, 0),
        'done before 10 pm': time(22, 0)
    }
    
    # Check for start time patterns
    for pattern, preferred_time in start_patterns.items():
        if pattern in text:
            start_time = preferred_time
            break
    
    # Check for end time patterns
    for pattern, preferred_time in end_patterns.items():
        if pattern in text:
            end_time = preferred_time
            break
    
    start_regex_patterns = [
        r'start\s+at\s+(\d{1,2}):(\d{2})\s*(am|pm)',
        r'start\s+at\s+(\d{1,2})\s*(am|pm)',
        r'begin\s+at\s+(\d{1,2}):(\d{2})\s*(am|pm)',
        r'begin\s+at\s+(\d{1,2})\s*(am|pm)',
        r'leave\s+at\s+(\d{1,2}):(\d{2})\s*(am|pm)',
        r'leave\s+at\s+(\d{1,2})\s*(am|pm)',
    ]

    end_regex_patterns = [
        r'end\s+before\s+(\d{1,2}):(\d{2})\s*(am|pm)',
        r'end\s+before\s+(\d{1,2})\s*(am|pm)',
        r'finish\s+before\s+(\d{1,2}):(\d{2})\s*(am|pm)',
        r'finish\s+before\s+(\d{1,2})\s*(am|pm)',
        r'done\s+before\s+(\d{1,2}):(\d{2})\s*(am|pm)',
        r'done\s+before\s+(\d{1,2})\s*(am|pm)',
    ]

    for pattern in start_regex_patterns:
        match = re.search(pattern, text)
        if match:
            hour = int(match.group(1))
            if ':' in pattern and len(match.groups()) >= 2 and match.group(2).isdigit():
                minute = int(match.group(2))
                ampm = match.group(3) if len(match.groups()) >= 3 else None
            else:
                minute = 0
                ampm = match.group(2) if len(match.groups()) >= 2 else None
            
            # handle AM/PM
            if ampm and ampm.lower() in ['am', 'pm']:
                if ampm.lower() == 'pm' and hour != 12:
                    hour += 12
                elif ampm.lower() == 'am' and hour == 12:
                    hour = 0
            
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                start_time = time(hour, minute)
                break

    for pattern in end_regex_patterns:
        match = re.search(pattern, text)
        if match:
            hour = int(match.group(1))
            if ':' in pattern and len(match.groups()) >= 2 and match.group(2).isdigit():
                minute = int(match.group(2))
                ampm = match.group(3) if len(match.groups()) >= 3 else None
            else:
                minute = 0
                ampm = match.group(2) if len(match.groups()) >= 2 else None
            
            # handle AM/PM
            if ampm and ampm.lower() in ['am', 'pm']:
                if ampm.lower() == 'pm' and hour != 12:
                    hour += 12
                elif ampm.lower() == 'am' and hour == 12:
                    hour = 0
            
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                end_time = time(hour, minute)
                break
    
    return start_time, end_time


def characterize_preferences(preferences_input):
    """Use Ollama to analyze user preferences and map to categories"""
    if not preferences_input.strip():
        return []
    
    prompt = f"""
    Analyze the following travel preferences and map them to relevant business categories:
    User preferences: "{preferences_input}"
    
    Available categories include: Restaurants, Food, Bakeries, Coffee & Tea, Fast Food, Bars, Attractions, Museums, Parks, Tourist Information, Historic Sites, Art Galleries, Shopping, Entertainment, Hotels, Nightlife, Active Life, Beauty & Spas, Health & Medical, Automotive, Local Services, Home Services, Professional Services, Financial Services, Public Services & Government, Religious Organizations, Event Planning & Services, Pets, Mass Media, Arts & Entertainment.
    
    Based on the user's preferences, return a comma-separated list of the most relevant categories. For example:
    - "elder people friendly" -> "Museums, Parks, Historic Sites, Art Galleries"
    - "adventure and excitement" -> "Active Life, Entertainment, Attractions"
    - "romantic trip" -> "Restaurants, Bars, Art Galleries"
    - "family with kids" -> "Parks, Attractions, Entertainment"
    
    Only return the category names, separated by commas:
    """
    
    try:
        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "phi3:mini",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        }
        response = requests.post(url, json=payload, timeout=50)
        result = response.json()
        
        # Parse the response to extract categories
        categories_text = result.get("response", "").strip()
        
        if "```" in categories_text:
            # Extract content between the first set of triple backticks
            parts = categories_text.split("```")
            if len(parts) > 1:
                categories_text = parts[1]
        
        # Extract just the comma-separated list
        categories = [cat.strip() for cat in categories_text.split(',') if cat.strip()]
        
        # Filter to only valid categories
        valid_categories = [
            "Restaurants", "Food", "Bakeries", "Coffee & Tea", "Fast Food", "Bars", 
            "Attractions", "Museums", "Parks", "Tourist Information", "Historic Sites", 
            "Art Galleries", "Shopping", "Entertainment", "Libraries"
        ]
        return [cat for cat in categories if cat in valid_categories]
        
    except Exception as e:
        st.warning(f"AI characterization failed: {e}. Using default categories.")
        return ["Attractions", "Museums", "Parks"]

def generate_city_introduction(city_name):
    """Generate a detailed city introduction with specific local features using Ollama"""
    prompt = f"""
    Write a concise introduction to {city_name} for travelers in exactly 80 words. Include:

    1. Brief historical significance (1-2 sentences)
    2. Key cultural highlights and local specialties (1-2 sentences)  
    3. Top must-see attractions specific to {city_name} (1-2 sentences)

    Be specific about {city_name}'s unique characteristics - mention actual landmark names, local food specialties, or cultural elements that make this city different from others.

    Keep it engaging but concise - exactly 80 words total.
    """

    try:
        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "phi3:mini",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        }
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
            
        result = response.json()
        text = result.get("response", "")
        
        # Ensure we have at least 3 paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(paragraphs) < 3:
            raise Exception("Insufficient paragraphs in response")
            
        return text.strip()
    except Exception as e:
        # Enhanced fallback with more specific content
        return (
            f"{city_name} boasts a rich historical heritage that has shaped its unique character over the centuries. "
            f"From its early settlement days to its role in major historical events, the city has evolved into a distinctive destination. "
            f"The architectural landscape tells the story of different eras, with historic buildings standing alongside modern developments.\n\n"
            f"The cultural fabric of {city_name} is woven with local traditions, distinctive cuisine, and vibrant arts communities. "
            f"Local markets, festivals, and neighborhood gatherings reflect the authentic spirit of the city. "
            f"The culinary scene features regional specialties and local ingredients that define the city's gastronomic identity.\n\n"
            f"Visitors to {city_name} can explore iconic landmarks, historic districts, and cultural institutions that showcase the city's personality. "
            f"Each neighborhood offers its own character, from bustling commercial areas to quiet residential streets. "
            f"The city's unique attractions provide experiences that can't be found anywhere else."
        )

CUISINE_CATEGORIES = {
    'Chinese': ['Chinese', 'Cantonese', 'Szechuan', 'Dim Sum', 'Hot Pot'],
    'Japanese': ['Japanese', 'Sushi Bars', 'Ramen', 'Teppanyaki'],
    'Italian': ['Italian', 'Pizza', 'Pasta Shops'],
    'American': ['American (Traditional)', 'American (New)', 'Burgers', 'BBQ', 'Steakhouses'],
    'Mexican': ['Mexican', 'Tex-Mex', 'Tacos'],
    'French': ['French', 'Bistros', 'Brasseries'],
    'Indian': ['Indian', 'Pakistani', 'Bangladeshi'],
    'Thai': ['Thai'],
    'Korean': ['Korean'],
    'Mediterranean': ['Mediterranean', 'Greek', 'Turkish', 'Middle Eastern'],
    'Seafood': ['Seafood'],
    'Vietnamese': ['Vietnamese'],
    'Fast Food': ['Fast Food', 'Sandwiches', 'Delis'],
    'Coffee & Bakery': ['Coffee & Tea', 'Bakeries', 'Cafes', 'Donuts'],
    'Bars & Nightlife': ['Bars', 'Pubs', 'Cocktail Bars', 'Wine Bars', 'Breweries']
}


restaurant_keywords = ["Restaurants", "Food", "Bakeries", "Coffee & Tea", "Fast Food", "Bars"]
attraction_keywords = [
    "Attractions", "Museums", "Parks", "Tourist Information", 
    "Historic Sites", "Art Galleries", "Libraries"
]


def has_keyword(categories, keywords):
    if pd.isna(categories):
        return False
    try:
        cats = [c.strip() for c in str(categories).split(",")]
        return any(k in cats for k in keywords)
    except:
        return False

def has_preference_match(categories, preference_categories):
    """Check if place categories match user preferences"""
    if pd.isna(categories) or not preference_categories:
        return False
    try:
        cats = [c.strip() for c in str(categories).split(",")]
        return any(pref in cats for pref in preference_categories)
    except:
        return False

def has_cuisine_match(categories, selected_cuisines):
    """Check if restaurant categories match selected cuisines"""
    if pd.isna(categories) or not selected_cuisines:
        return False
    try:
        cats = [c.strip() for c in str(categories).split(",")]
        for cuisine in selected_cuisines:
            cuisine_keywords = CUISINE_CATEGORIES.get(cuisine, [cuisine])
            if any(keyword in cats for keyword in cuisine_keywords):
                return True
        return False
    except:
        return False

df['is_restaurant'] = df['categories'].apply(lambda x: has_keyword(x, restaurant_keywords))
df['is_attraction'] = df['categories'].apply(lambda x: has_keyword(x, attraction_keywords))


def parse_time(time_str):
    """Parse time string in HH:MM format"""
    try:
        parts = time_str.split(':')
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        return time(hour, minute)
    except:
        return None

def is_open_at(hours_str, check_time):
    """Improved function to check if place is open at given time"""
    if pd.isna(hours_str) or hours_str == "" or str(hours_str).lower() == 'nan':
        return True
    
    try:
        hours_str = str(hours_str)
        if "-" not in hours_str:
            return True
            
        open_str, close_str = hours_str.split("-")
        open_time = parse_time(open_str.strip())
        close_time = parse_time(close_str.strip())
        
        if not open_time or not close_time:
            return True
            
        if open_time == close_time:
            return False
            
        if open_time < close_time:
            return open_time <= check_time <= close_time
        else:
            return check_time >= open_time or check_time <= close_time
    except Exception:
        return True


TRAVEL_MODES = {
    'Walk': {'min_time': 5, 'max_time': 20, 'speed_kmh': 5},
    'Bus': {'min_time': 10, 'max_time': 30, 'speed_kmh': 25},
    'Taxi': {'min_time': 5, 'max_time': 25, 'speed_kmh': 35},
    'Metro': {'min_time': 8, 'max_time': 35, 'speed_kmh': 40}
}

def calculate_estimated_distance(from_place, to_place):
    """Simulate distance calculation based on place names/addresses"""
    same_area_keywords = ['central', 'downtown', 'midtown', 'uptown']
    
    from_addr = str(from_place.get('address', '')).lower()
    to_addr = str(to_place.get('address', '')).lower()
    
    from_areas = [kw for kw in same_area_keywords if kw in from_addr]
    to_areas = [kw for kw in same_area_keywords if kw in to_addr]
    
    if from_areas and to_areas and any(area in to_areas for area in from_areas):
        return random.uniform(0.5, 2.0)
    else:
        return random.uniform(1.5, 8.0)


def calculate_visit_duration(place_categories):
    """Calculate realistic visit duration based on place type"""
    if not place_categories or pd.isna(place_categories):
        return random.randint(60, 90)  # Default 1-1.5 hours
    
    cats = str(place_categories).lower()
    

    if any(kw in cats for kw in ["restaurant", "food", "diner", "cafe", "bistro", "eatery"]):
        if "fast food" in cats:
            return random.randint(50, 70)  # Fast food: 40-60 min
        elif "fine dining" in cats:
            return random.randint(70, 120)  # Fine dining: 1.5-2 hours
        else:
            return random.randint(65, 95)  # Casual dining: 45-75 min
    

    if any(kw in cats for kw in ["museum", "gallery", "exhibit"]):
        return random.randint(110, 130)  # Museum/gallery: 2-2.5 hours
    
    if any(kw in cats for kw in ["park", "garden", "reserve"]):
        return random.randint(80, 110)  # Park: 1-2 hours
    
    if any(kw in cats for kw in ["historic", "monument", "landmark"]):
        return random.randint(70, 100)  # Historic site: 1-1.5 hours
    
    if any(kw in cats for kw in ["library"]):
        return random.randint(60, 100)  # Library: 1-1.5 hours
    
    return random.randint(80, 90)  # Default: 1-1.5 hours


def calculate_travel_time(distance_km, preferred_mode):
    """Calculate realistic travel time based on distance and preferred transport mode"""
    #Choose the way of transport using algorithm
    if distance_km <= 1.0:
        actual_mode = 'Walk'
    elif distance_km <= 3.0:
        actual_mode = preferred_mode if preferred_mode in ['Walk', 'Bus'] else 'Walk'
    else:
        actual_mode = preferred_mode if preferred_mode != 'Walk' else 'Bus'
    
    if actual_mode == 'Walk':
        return max(5, min(60, int(distance_km * 15))), actual_mode
    elif actual_mode == 'Bus':
        return max(10, min(45, int(distance_km * 8))), actual_mode
    elif actual_mode == 'Taxi':
        return max(5, min(30, int(distance_km * 5))), actual_mode
    elif actual_mode == 'Metro':
        return max(8, min(40, int(distance_km * 6))), actual_mode
    else:
        return 15, actual_mode

# API Configuration
API_KEY = "your_actual_api_key_here"

def format_city_name(city_input):
    """
    Format city name with proper capitalization
    """
    # Remove leading/trailing spaces and convert to proper case
    city = city_input.strip()
    
    # For English city names: capitalize each word
    return city.title()

def get_real_weather(city_input):
    """
    Get real-time weather data for specified city
    """
    try:
        # Format city name
        city = format_city_name(city_input)
        
        # Build API request URL
        url = f"http://api.weatherapi.com/v1/current.json?key={API_KEY}&q={city}"
        
        # Send request
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        
        # Parse JSON data
        data = response.json()
        
        # Extract weather information
        weather_info = {
            'city': data['location']['name'],
            'country': data['location']['country'],
            'date': datetime.now().strftime("%Y-%m-%d"),
            'condition': data['current']['condition']['text'],
            'temp_c': data['current']['temp_c'],
            'temp_f': data['current']['temp_f'],
            'humidity': data['current']['humidity'],
            'wind_speed': data['current']['wind_kph'],
            'wind_direction': data['current']['wind_dir'],
            'feels_like_c': data['current']['feelslike_c'],
            'uv_index': data['current']['uv'],
            'visibility': data['current']['vis_km']
        }
        
        return weather_info
        
    except requests.exceptions.RequestException as e:
        print(f"Network request error: {e}")
        return None
    except KeyError as e:
        print(f"Data parsing error: {e}")
        return None
    except Exception as e:
        print(f"Error occurred while fetching weather data: {e}")
        return None

def generate_estimated_weather(travel_date):
    """
    Fallback function: Generate estimated weather data when API is unavailable
    """
    conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy", "Light Rain", 
                 "Heavy Rain", "Thunderstorm", "Windy", "Foggy"]
    
    month = travel_date.month
    if month in [12, 1, 2]:  # Winter
        temp_min = random.randint(-3, 2)
    elif month in [3, 4, 5]:  # Spring
        temp_min = random.randint(6, 10)
    elif month in [6, 7, 8]:  # Summer
        temp_min = random.randint(18, 22)
    else:  # Fall
        temp_min = random.randint(10, 16)
    
    temp_max = temp_min + random.randint(3, 6)
    condition = random.choice(conditions)
    
    details = {
        'date': travel_date.strftime("%Y-%m-%d"),
        'condition': condition,
        'temp_min': temp_min,
        'temp_max': temp_max,
        'humidity': random.randint(40, 60),
        'wind_speed': random.randint(5, 15)
    }
    
    return details

def get_weather_with_fallback(city_input, travel_date=None):
    """
    Get weather information with fallback to estimated data if API fails
    """
    # try to get real weather data
    real_weather = get_real_weather(city_input)
    
    if real_weather:
        print(f"=== Real-time Weather for {real_weather['city']}, {real_weather['country']} ===")
        print(f"Date: {real_weather['date']}")
        print(f"Condition: {real_weather['condition']}")
        print(f"Temperature: {real_weather['temp_c']}°C ({real_weather['temp_f']}°F)")
        print(f"Feels like: {real_weather['feels_like_c']}°C")
        print(f"Humidity: {real_weather['humidity']}%")
        print(f"Wind Speed: {real_weather['wind_speed']} km/h")
        print(f"Wind Direction: {real_weather['wind_direction']}")
        print(f"UV Index: {real_weather['uv_index']}")
        print(f"Visibility: {real_weather['visibility']} km")
        return real_weather
    else:
        print("Unable to fetch real-time weather data, using estimated weather...")
        if travel_date is None:
            travel_date = datetime.now()
        estimated_weather = generate_estimated_weather(travel_date)
        print(f"=== Estimated Weather Information ===")
        print(f"Date: {estimated_weather['date']}")
        print(f"Condition: {estimated_weather['condition']}")
        print(f"Min Temperature: {estimated_weather['temp_min']}°C")
        print(f"Max Temperature: {estimated_weather['temp_max']}°C")
        print(f"Humidity: {estimated_weather['humidity']}%")
        print(f"Wind Speed: {estimated_weather['wind_speed']} km/h")
        return estimated_weather
    


def get_weather_tips(weather):
    """Generate specific weather tips based on conditions"""
    tips = []
    condition = weather['condition'].lower()
    temp_max = weather['temp_max']
    temp_min = weather['temp_min']
    
    if temp_max > 25:
        tips.append("🌡️ Hot weather - stay hydrated and wear light clothing!")
    elif temp_min < 5:
        tips.append("🧥 Cold weather - dress warmly and wear layers!")
    elif temp_min < 15:
        tips.append("🧥 Cool weather - bring a light jacket!")
    
    if 'rain' in condition:
        tips.append("☔ Rainy weather - don't forget your umbrella and waterproof shoes!")
    elif condition == 'thunderstorm':
        tips.append("⛈️ Thunderstorm expected - plan indoor activities and carry an umbrella!")
    elif condition == 'windy':
        tips.append("💨 Windy weather - secure loose items and dress appropriately!")
    elif condition == 'foggy':
        tips.append("🌫️ Foggy conditions - allow extra travel time and be careful!")
    elif condition == 'sunny':
        tips.append("☀️ Sunny day - perfect for outdoor activities! Don't forget sunglasses and sunscreen!")
    
    if weather['humidity'] > 70:
        tips.append("💧 High humidity - wear breathable clothing!")
    
    return tips

# Ollama API call for travel tips 
def ollama_chat(prompt):
    try:
        url = "http://127.0.0.1:11434/api/generate"
        payload = {
            "model": "phi3:mini",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.7}
        }
        response = requests.post(url, json=payload, timeout=40)
        result = response.json()
        return result.get("response", "Sorry, no suggestion generated.")
    except requests.exceptions.ConnectionError:
        return "💡 Consider bringing appropriate clothing and items based on the weather conditions."
    except Exception as e:
        return f"💡 Plan accordingly for the weather conditions."



def get_meal_time_ranges():
    """Define meal time ranges"""
    return {
        'breakfast': (time(7, 0), time(10, 30)),    # 7:00-10:30
        'lunch': (time(11, 0), time(13, 0)),        # 11:00-13:00
        'dinner': (time(17, 0), time(20, 0)),       # 17:00-20:00
        'late_night': (time(20, 0), time(23, 0))    # 20:00-23:00
    }


def is_restaurant_open_on_day(hours_str):
    """Check if restaurant has opening hours (not closed) on the day"""
    if pd.isna(hours_str) or hours_str == "" or str(hours_str).lower() == 'nan':
        return True  # Assume open if no hours data
    
    hours_str = str(hours_str).lower()
    if 'closed' in hours_str:
        return False
    
    return True


def is_restaurant_open_during_meal_time(opening_hours, meal_type):
    """Check if restaurant is open during the specified meal time"""
    meal_ranges = get_meal_time_ranges()
    
    if meal_type not in meal_ranges:
        return True  # If meal type not specified, don't filter
    
    meal_start, meal_end = meal_ranges[meal_type]
    
    if not opening_hours or opening_hours == 'Hours not available':
        return True  # Assume open if no hours available
    
    try:
        # Parse opening hours (assuming format like "09:00-22:00")
        if "-" in str(opening_hours):
            hours_parts = str(opening_hours).split("-")
            if len(hours_parts) == 2:
                open_str = hours_parts[0].strip()
                close_str = hours_parts[1].strip()
                
                open_time = parse_time(open_str)
                close_time = parse_time(close_str)
                
                if open_time and close_time:
                    # Check if restaurant is open during any part of the meal time
                    # Restaurant should be open before meal ends and close after meal starts
                    return open_time <= meal_end and close_time >= meal_start
    except:
        pass
    
    return True  # Default to True if parsing fails


def determine_meal_type_from_time(check_time):
    """Automatically determine meal type based on current time"""
    if isinstance(check_time, str):
        check_time = parse_time(check_time)
    
    if not check_time:
        return None
    
    # Define meal time ranges
    if time(7, 0) <= check_time <= time(10, 30):
        return 'breakfast'
    elif time(11, 0) <= check_time <= time(13, 0):
        return 'lunch'
    elif time(17, 0) <= check_time <= time(20, 0):
        return 'dinner'
    elif time(20, 0) <= check_time <= time(23, 0):
        return 'late_night'
    else:
        return None  # Outside meal times, no filtering


def recommend_featured_restaurants(df, city, travel_date, top_n=10):
    """
    Recommend top local restaurants based on rating and availability
    
    Args:
        df: Restaurant dataframe
        city: City name
        travel_date: Travel date
        top_n: Number of top restaurants to choose from (default 10)
    
    Returns:
        List of 2 randomly selected restaurants from top rated ones
    """
    try:
        # Filter by city and restaurant type
        city_restaurants = df[(df['is_restaurant'] == True) & 
                            (df['city'].str.lower() == city.lower())]
        
        if city_restaurants.empty:
            return []
        
        # Get day column for opening hours
        day_name = travel_date.strftime("%A")
        day_column = f"{day_name}_hours"
        
        # Filter by opening hours (must be open on travel day)
        if day_column in city_restaurants.columns:
            open_restaurants = city_restaurants[
                city_restaurants[day_column].apply(lambda h: is_restaurant_open_on_day(h))
            ]
        else:
            open_restaurants = city_restaurants
        
        if open_restaurants.empty:
            return []
        
        # Filter by rating (must have rating data)
        rated_restaurants = open_restaurants.dropna(subset=['stars'])
        rated_restaurants['stars'] = pd.to_numeric(rated_restaurants['stars'], errors='coerce')
        rated_restaurants = rated_restaurants[rated_restaurants['stars'].notna()]
        
        if rated_restaurants.empty:
            return []
        
        # Sort by rating and get top restaurants
        top_restaurants = rated_restaurants.sort_values(by='stars', ascending=False).head(top_n)
        
        # Randomly select 2 restaurants from top rated ones
        num_to_select = min(2, len(top_restaurants))
        selected_restaurants = top_restaurants.sample(n=num_to_select).to_dict('records')
        
        return selected_restaurants
        
    except Exception as e:
        st.error(f"Error recommending featured restaurants: {str(e)}")
        return []
    

def recommend_places(df, type_col, day_column, check_time, city, must_visit_places=None, 
                    preference_categories=None, selected_cuisines=None):
    """
    Enhanced recommendation function with automatic meal time filtering
    """
    try:
        # Automatically determine meal type for restaurants based on time
        meal_type = None
        if type_col == 'is_restaurant':
            meal_type = determine_meal_type_from_time(check_time)
        
        # Get must-visit places that match this type first
        must_visit_for_this_type = []
        if must_visit_places:
            for must_visit in must_visit_places:
                place_data = must_visit.get('matched_place', must_visit)

                if not isinstance(place_data, dict):
                    continue 
                
                if type_col == 'is_restaurant' and has_keyword(place_data.get('categories', ''), restaurant_keywords):
                    must_visit_for_this_type.append(place_data)
                elif type_col == 'is_attraction' and has_keyword(place_data.get('categories', ''), attraction_keywords):
                    must_visit_for_this_type.append(place_data)

        # Calculate recommended count based on must-visit places
        if type_col == 'is_attraction':
            base_count = 3  # Base number of attractions
            recommended_count = max(1, base_count - len(must_visit_for_this_type))
        else:  # restaurants
            base_count = 2  # Base number of restaurants
            recommended_count = max(1, base_count - len(must_visit_for_this_type))
        
        # Filter by city and type
        filtered = df[(df[type_col]) & (df['city'].str.lower() == city.lower())]
        
        # Apply cuisine filtering for restaurants
        if selected_cuisines and type_col == 'is_restaurant':
            cuisine_matches = filtered[filtered['categories'].apply(
                lambda x: has_cuisine_match(x, selected_cuisines)
            )]
            if not cuisine_matches.empty:
                filtered = cuisine_matches
        
        # Apply preference filtering if provided
        if preference_categories:
            preference_matches = filtered[filtered['categories'].apply(
                lambda x: has_preference_match(x, preference_categories)
            )]
            if not preference_matches.empty:
                filtered = preference_matches
        
        # Apply meal time filtering for restaurants
        if meal_type and type_col == 'is_restaurant' and day_column in filtered.columns:
            meal_time_matches = filtered[filtered[day_column].apply(
                lambda h: is_restaurant_open_during_meal_time(h, meal_type)
            )]
            if not meal_time_matches.empty:
                filtered = meal_time_matches
        
        # Filter by opening hours - check if will be open during the day
        if day_column in filtered.columns:
            filtered = filtered[filtered[day_column].apply(lambda h: will_be_open_today(h, check_time, end_time_pref))]
        
        # Get additional recommended places
        additional_places = []
        if recommended_count > 0 and 'stars' in filtered.columns:
            filtered = filtered.dropna(subset=['stars'])
            filtered['stars'] = pd.to_numeric(filtered['stars'], errors='coerce')
            filtered = filtered[filtered['stars'].notna()]
            
            # Exclude must-visit places
            if must_visit_for_this_type:
                must_visit_names = [mv['name'] for mv in must_visit_for_this_type]
                filtered = filtered[~filtered['name'].isin(must_visit_names)]
            
            # Get top rated places
            top_places = filtered.sort_values(by='stars', ascending=False).head(recommended_count)
            
            for _, place in top_places.iterrows():
                additional_places.append(place.to_dict())
        
        # Combine must-visit and additional places
        all_places = must_visit_for_this_type + additional_places
        
        day_columns = [col for col in df.columns if col.endswith('_hours')]
        return_columns = ['name', 'address', 'stars', 'categories']
        if 'tip' in df.columns:
            return_columns.append('tip')
        return_columns.extend(day_columns)
        
        formatted_places = []
        must_visit_names = [mv['name'] for mv in must_visit_for_this_type]

        for place in all_places:
            place_dict = {}
            
            if isinstance(place, dict) and 'name' in place:
                place_name = place['name']
                original_place = df[df['name'] == place_name].iloc[0] if not df[df['name'] == place_name].empty else None
                
                if original_place is not None:
                    for col in return_columns:
                        if col in original_place:
                            place_dict[col] = original_place[col]
                        else:
                            place_dict[col] = place.get(col, '')
                else:
                    for col in return_columns:
                        place_dict[col] = place.get(col, '')
            else:
                for col in return_columns:
                    if hasattr(place, col):
                        place_dict[col] = getattr(place, col)
                    elif isinstance(place, dict):
                        place_dict[col] = place.get(col, '')
                    else:
                        place_dict[col] = ''
            
            # Add must-visit flag and meal type info
            place_dict['is_must_visit'] = place_dict.get('name', '') in must_visit_names
            if meal_type and type_col == 'is_restaurant':
                place_dict['meal_type'] = meal_type
            
            formatted_places.append(place_dict)
        
        return formatted_places
    
    except Exception as e:
        st.error(f"Error in recommend_places: {str(e)}")
        return []


def will_be_open_today(hours_str, start_time, end_time):
    """Check if place will be open at any point during the planned day"""
    if pd.isna(hours_str) or hours_str == "" or str(hours_str).lower() == 'nan':
        return True
    
    try:
        hours_str = str(hours_str)
        if "-" not in hours_str:
            return True
        
        open_str, close_str = hours_str.split("-")
        open_time = parse_time(open_str.strip())
        close_time = parse_time(close_str.strip())
        
        if not open_time or not close_time:
            return True
        
        # Check if there's any overlap between operating hours and planned visit time
        if open_time < close_time:
            return not (close_time < start_time or open_time > end_time)
        else:  # Crosses midnight
            return True
    except Exception:
        return True
    


def generate_itinerary_with_meal_times(date, attractions, restaurants, preferences, transport_mode):
    """Generate itinerary with proper meal time scheduling"""
    itinerary = []
    current_time = datetime.combine(date, time(9, 0))  # Default start at 9 AM
    
    # Extract time preferences from user input
    start_time_pref, end_time_pref = extract_time_preferences(preferences)
    current_time = datetime.combine(date, start_time_pref)
    end_time = datetime.combine(date, end_time_pref)
    
    # Define meal time slots
    meal_times = {
        'lunch': (time(11, 30), time(13, 30)),  # Preferred lunch window
        'dinner': (time(17, 30), time(20, 30))  # Preferred dinner window
    }
    
    # Separate restaurants by meal type (you might want to classify them)
    lunch_restaurants = []
    dinner_restaurants = []
    
    # Simple classification - you might want to make this more sophisticated
    for restaurant in restaurants:
        categories = restaurant.get('categories', '').lower()
        if any(keyword in categories for keyword in ['cafe', 'breakfast', 'brunch']):
            lunch_restaurants.append(restaurant)
        elif any(keyword in categories for keyword in ['bar', 'pub', 'fine dining']):
            dinner_restaurants.append(restaurant)
        else:
            # Default assignment based on order
            if len(lunch_restaurants) < len(dinner_restaurants):
                lunch_restaurants.append(restaurant)
            else:
                dinner_restaurants.append(restaurant)
    
    # Create scheduled items with meal times
    scheduled_items = []
    
    # Add attractions (flexible timing)
    for attraction in attractions:
        scheduled_items.append({
            'place': attraction,
            'type': 'attraction',
            'priority': 0 if attraction.get('is_must_visit', False) else 1,
            'flexible': True
        })
    
    # Sort scheduled items by priority first
    scheduled_items.sort(key=lambda x: (x['priority'], not x.get('flexible', True)))
    
    # Generate itinerary with meal time constraints
    lunch_added = False
    dinner_added = False
    
    for i, item in enumerate(scheduled_items):
        place = item['place']
        place_type = item['type']
        
        # Calculate visit duration
        visit_duration = calculate_visit_duration(place.get('categories', ''))
        
        # Calculate travel time from previous location
        travel_time, actual_transport = 0, None
        if itinerary:  # If there are previous items in itinerary
            prev_place = itinerary[-1]['place']
            distance = calculate_estimated_distance(prev_place, place)
            travel_time, actual_transport = calculate_travel_time(distance, transport_mode)

        # Add travel time to current time
        if travel_time > 0:
            current_time += timedelta(minutes=travel_time)

        # Check opening hours and adjust arrival time
        day_name = date.strftime("%A")
        day_column = f"{day_name}_hours"
        opening_hours = place.get(day_column, '')

        if opening_hours and opening_hours != 'Hours not available':
            try:
                if "-" in str(opening_hours):
                    open_str = str(opening_hours).split("-")[0].strip()
                    open_time = parse_time(open_str)
                    if open_time:
                        opening_datetime = datetime.combine(date, open_time)
                        if current_time < opening_datetime:
                            current_time = opening_datetime
            except:
                pass

        # Check if we need to add lunch before this activity
        if not lunch_added and current_time.time() >= meal_times['lunch'][0]:
            if lunch_restaurants:
                # Find the best lunch restaurant that's open
                best_lunch_restaurant = None
                for restaurant in lunch_restaurants:
                    restaurant_hours = restaurant.get(day_column, '')
                    if is_restaurant_open_during_meal_time(restaurant_hours, 'lunch'):
                        best_lunch_restaurant = restaurant
                        break
                
                if best_lunch_restaurant:
                    # Calculate travel time to lunch restaurant
                    lunch_travel_time = 0
                    if itinerary:
                        prev_place = itinerary[-1]['place']
                        distance = calculate_estimated_distance(prev_place, best_lunch_restaurant)
                        lunch_travel_time, _ = calculate_travel_time(distance, transport_mode)
                    
                    # Adjust time for lunch
                    lunch_time = current_time + timedelta(minutes=lunch_travel_time)
                    
                    # Ensure lunch is within preferred window
                    earliest_lunch = datetime.combine(date, meal_times['lunch'][0])
                    latest_lunch = datetime.combine(date, meal_times['lunch'][1])
                    
                    if lunch_time < earliest_lunch:
                        lunch_time = earliest_lunch
                    elif lunch_time > latest_lunch:
                        lunch_time = latest_lunch
                    
                    # Add lunch to itinerary
                    lunch_item = {
                        'time': lunch_time.strftime("%H:%M"),
                        'place': best_lunch_restaurant,
                        'duration': 60,  # 1 hour for lunch
                        'type': 'restaurant',
                        'meal_type': 'lunch',
                        'travel_time': lunch_travel_time if lunch_travel_time > 0 else None,
                        'transport_mode': transport_mode if lunch_travel_time > 0 else None
                    }
                    
                    itinerary.append(lunch_item)
                    lunch_added = True
                    
                    # Update current time after lunch
                    current_time = lunch_time + timedelta(minutes=60 + 10)  # 60 min lunch + 10 min buffer

        # Check if we need to add dinner before this activity
        if not dinner_added and current_time.time() >= meal_times['dinner'][0]:
            if dinner_restaurants:
                # Find the best dinner restaurant that's open
                best_dinner_restaurant = None
                for restaurant in dinner_restaurants:
                    restaurant_hours = restaurant.get(day_column, '')
                    if is_restaurant_open_during_meal_time(restaurant_hours, 'dinner'):
                        best_dinner_restaurant = restaurant
                        break
                
                if best_dinner_restaurant:
                    # Calculate travel time to dinner restaurant
                    dinner_travel_time = 0
                    if itinerary:
                        prev_place = itinerary[-1]['place']
                        distance = calculate_estimated_distance(prev_place, best_dinner_restaurant)
                        dinner_travel_time, _ = calculate_travel_time(distance, transport_mode)
                    
                    # Adjust time for dinner
                    dinner_time = current_time + timedelta(minutes=dinner_travel_time)
                    
                    # Ensure dinner is within preferred window
                    earliest_dinner = datetime.combine(date, meal_times['dinner'][0])
                    latest_dinner = datetime.combine(date, meal_times['dinner'][1])
                    
                    if dinner_time < earliest_dinner:
                        dinner_time = earliest_dinner
                    elif dinner_time > latest_dinner:
                        dinner_time = latest_dinner
                    
                    # Add dinner to itinerary
                    dinner_item = {
                        'time': dinner_time.strftime("%H:%M"),
                        'place': best_dinner_restaurant,
                        'duration': 90,  # 1.5 hours for dinner
                        'type': 'restaurant',
                        'meal_type': 'dinner',
                        'travel_time': dinner_travel_time if dinner_travel_time > 0 else None,
                        'transport_mode': transport_mode if dinner_travel_time > 0 else None
                    }
                    
                    itinerary.append(dinner_item)
                    dinner_added = True
                    
                    # Update current time after dinner
                    current_time = dinner_time + timedelta(minutes=90 + 10)  # 90 min dinner + 10 min buffer

        # Check if we exceed end time
        estimated_end = current_time + timedelta(minutes=visit_duration)
        if estimated_end > end_time:
            break
        
        # Create itinerary item for the current attraction
        itinerary_item = {
            'time': current_time.strftime("%H:%M"),
            'place': place,
            'duration': visit_duration,
            'type': place_type,
            'travel_time': travel_time if travel_time > 0 else None,
            'transport_mode': transport_mode if travel_time > 0 else None
        }
        
        itinerary.append(itinerary_item)
        
        # Update current time
        current_time = estimated_end
        
        # Add buffer time between activities
        current_time += timedelta(minutes=10)
    
    # Add remaining meals if not added yet and there's still time
    if not lunch_added and current_time.time() <= meal_times['lunch'][1] and lunch_restaurants:
        best_lunch_restaurant = None
        for restaurant in lunch_restaurants:
            restaurant_hours = restaurant.get(day_column, '')
            if is_restaurant_open_during_meal_time(restaurant_hours, 'lunch'):
                best_lunch_restaurant = restaurant
                break
        
        if best_lunch_restaurant:
            lunch_travel_time = 0
            if itinerary:
                prev_place = itinerary[-1]['place']
                distance = calculate_estimated_distance(prev_place, best_lunch_restaurant)
                lunch_travel_time, _ = calculate_travel_time(distance, transport_mode)
            
            lunch_time = current_time + timedelta(minutes=lunch_travel_time)
            
            lunch_item = {
                'time': lunch_time.strftime("%H:%M"),
                'place': best_lunch_restaurant,
                'duration': 60,
                'type': 'restaurant',
                'meal_type': 'lunch',
                'travel_time': lunch_travel_time if lunch_travel_time > 0 else None,
                'transport_mode': transport_mode if lunch_travel_time > 0 else None
            }
            
            itinerary.append(lunch_item)
            current_time = lunch_time + timedelta(minutes=70)  # 60 min + 10 min buffer
    
    if not dinner_added and current_time.time() <= meal_times['dinner'][1] and dinner_restaurants:
        best_dinner_restaurant = None
        for restaurant in dinner_restaurants:
            restaurant_hours = restaurant.get(day_column, '')
            if is_restaurant_open_during_meal_time(restaurant_hours, 'dinner'):
                best_dinner_restaurant = restaurant
                break
        
        if best_dinner_restaurant:
            dinner_travel_time = 0
            if itinerary:
                prev_place = itinerary[-1]['place']
                distance = calculate_estimated_distance(prev_place, best_dinner_restaurant)
                dinner_travel_time, _ = calculate_travel_time(distance, transport_mode)
            
            dinner_time = current_time + timedelta(minutes=dinner_travel_time)
            
            dinner_item = {
                'time': dinner_time.strftime("%H:%M"),
                'place': best_dinner_restaurant,
                'duration': 90,
                'type': 'restaurant',
                'meal_type': 'dinner',
                'travel_time': dinner_travel_time if dinner_travel_time > 0 else None,
                'transport_mode': transport_mode if dinner_travel_time > 0 else None
            }
            
            itinerary.append(dinner_item)
    
    return itinerary


def display_enhanced_travel_tips(city, preferences, weather_info, travel_date, df):
    """Display enhanced travel tips with restaurant recommendations"""
    
    # Generate original travel tips
    travel_tips = generate_travel_tips(city, preferences, weather_info)
    
    # Get featured restaurant recommendations
    featured_restaurants = recommend_featured_restaurants(df, city, travel_date)
    
    st.subheader("💡 Personalized Travel Tips")
    
    # Display original travel tips
    st.markdown(travel_tips)
    
    # Add featured restaurants section
    if featured_restaurants:
        st.markdown("---")
        
        # Use expander for dropdown effect
        with st.expander("🍽️ **Featured Local Restaurants** (Click to expand)", expanded=False):
            st.markdown("*Here are 2 highly-rated local restaurants worth trying:*")
            
            for i, restaurant in enumerate(featured_restaurants, 1):
                st.markdown(f"### {i}. {restaurant['name']} ⭐ {restaurant.get('stars', 'N/A')}")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**📍 Address:** {restaurant.get('address', 'N/A')}")
                    st.write(f"**🏷️ Categories:** {restaurant.get('categories', 'N/A')}")
                    
                    # Display opening hours for the travel day
                    day_name = travel_date.strftime("%A")
                    hours_column = f"{day_name}_hours"
                    if hours_column in restaurant:
                        opening_hours = restaurant[hours_column]
                        if pd.isna(opening_hours) or opening_hours == '' or str(opening_hours).lower() == 'nan':
                            opening_hours = 'Hours not available'
                    else:
                        opening_hours = 'Hours not available'
                    st.write(f"**🕐 Opening Hours ({day_name}):** {opening_hours}")
                    
                    # Display tip if available
                    if restaurant.get('tip'):
                        st.write(f"**💡 Tip:** {restaurant.get('tip')}")
                
                with col2:
                    if restaurant.get('stars'):
                        st.metric("Rating", f"{restaurant['stars']}/5.0 ⭐")
                
                # Add separator between restaurants
                if i < len(featured_restaurants):
                    st.markdown("---")


def generate_travel_tips(city, preferences, weather_info):
    """Generate personalized travel tips using Ollama AI"""
    weather_desc = f"{weather_info['condition']} with temperatures {weather_info['temp_min']}°C to {weather_info['temp_max']}°C"
    
    prompt = f"""
    Generate 2-3 specific and practical travel tips for visiting {city}. 
    
    Context:
    - User preferences: {preferences if preferences else 'General traveler'}
    - City: {city}
    
    Focus on:
    1. Local transportation tips specific to {city}
    2. Cultural etiquette or local customs in {city}
    3. Local food or shopping recommendations
    
    Make each tip specific to {city} and practical. Keep each tip to 1-2 sentences.
    Format as bullet points starting with emoji.
    """
    
    try:
        response = ollama_chat(prompt)
        return response
    except Exception as e:
        # Enhanced fallback tips
        return f"""
        🚗 Use local public transportation or rideshare services for efficient travel around {city}
        🏛️ Many attractions in {city} offer discounted combination tickets - check for city passes
        🍽️ Try local specialties and visit neighborhood restaurants for authentic {city} cuisine
        """

def get_restaurant_recommendations_for_tips(df, city, travel_date, exclude_names=None):
    """
    Get restaurant recommendations specifically for travel tips
    Can exclude restaurants already in itinerary to avoid duplication
    """
    try:
        # Get featured restaurants
        featured_restaurants = recommend_featured_restaurants(df, city, travel_date)
        
        # Exclude restaurants already in itinerary if provided
        if exclude_names:
            featured_restaurants = [
                r for r in featured_restaurants 
                if r['name'] not in exclude_names
            ]
        
        # If we have less than 2 after exclusion, get more
        if len(featured_restaurants) < 2:
            additional_restaurants = recommend_featured_restaurants(df, city, travel_date, top_n=20)
            for restaurant in additional_restaurants:
                if restaurant['name'] not in exclude_names and restaurant not in featured_restaurants:
                    featured_restaurants.append(restaurant)
                    if len(featured_restaurants) >= 2:
                        break
        
        return featured_restaurants[:2]  # Return max 2 restaurants
        
    except Exception as e:
        st.error(f"Error getting restaurant recommendations: {str(e)}")
        return []
    


def main():
    st.title("Pathgenie - Your AI Travel Agent")
    st.markdown("*Plan Less. Travel More.*")
    
    # Input form
    with st.form("travel_form"):
        st.subheader("📍 Trip Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.text_input("Which city are you visiting?", placeholder="e.g., New York, Los Angeles")
            travel_date = st.date_input("Travel date", value=datetime.now().date())
        
        with col2:
            transport_mode = st.selectbox(
                "Preferred transport mode",
                ["Walk", "Bus", "Taxi", "Metro"]
            )
        
        st.subheader("🎯 Preferences")
        
        col3, col4 = st.columns(2)
        
        with col3:
            must_visit = st.text_input(
                "Must-visit places (optional)",
                placeholder="e.g., Central Park, Empire State Building",
                help="Comma-separated list of specific places you want to visit"
            )
        
        with col4:
            cuisines = st.multiselect(
                "Preferred cuisines (optional)",
                list(CUISINE_CATEGORIES.keys()),
                help="Select your preferred types of food"
            )
        
        preferences = st.text_area(
            "Additional preferences (optional)",
            placeholder="e.g., family-friendly, start at 10 AM, end before 8 PM, romantic evening, outdoor activities",
            help="Describe your preferences, including preferred start/end times"
        )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Generate Itinerary", type="primary")
    
    # Process the form submission
    if submitted:
        if not city:
            st.error("Please enter a city name.")
            return
        
        with st.spinner("🔍 Generating your personalized itinerary..."):
            try:
                # Normalize city name
                normalized_city = normalize_city_name(city)               
                # Check if city exists in data
                city_data = df[df['city'].str.lower() == normalized_city.lower()]
                if city_data.empty:
                    st.error(f"Sorry, we don't have data for {normalized_city}. Please try a different city.")
                    return
                
                city_intro = generate_city_introduction(normalized_city)                
                found_places, not_found_places = find_matching_places(must_visit, df, normalized_city)                
                preference_categories = characterize_preferences(preferences)                
                start_time_pref, end_time_pref = extract_time_preferences(preferences)
                day_name = travel_date.strftime("%A")
                day_column = f"{day_name.lower()}_hours"
                weather_info = generate_estimated_weather(travel_date)
                attractions = recommend_places(
                    df, 'is_attraction', day_column, start_time_pref, 
                    normalized_city, found_places, preference_categories
                )
                restaurants = recommend_places(
                    df, 'is_restaurant', day_column, start_time_pref, 
                    normalized_city, found_places, preference_categories, cuisines
                )
                itinerary = generate_itinerary_with_meal_times(
                    travel_date, attractions, restaurants, preferences, transport_mode
                )
                travel_tips = generate_travel_tips(normalized_city, preferences, weather_info)
                st.success(f"✅ Itinerary generated for {normalized_city}!")
                st.subheader(f"🏙️ Welcome to {normalized_city}")
                st.markdown(city_intro)
                st.subheader("🌤️ Weather Forecast")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown(f"<div style='text-align: center;'><h4>Condition</h4><p style='font-size: 16px;'>{weather_info['condition']}</p></div>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"<div style='text-align: center;'><h4>Temperature</h4><p style='font-size: 16px;'>{weather_info['temp_min']}°C - {weather_info['temp_max']}°C</p></div>", unsafe_allow_html=True)
                with col3:
                    st.markdown(f"<div style='text-align: center;'><h4>Humidity</h4><p style='font-size: 16px;'>{weather_info['humidity']}%</p></div>", unsafe_allow_html=True)
                with col4:
                    st.markdown(f"<div style='text-align: center;'><h4>Wind Speed</h4><p style='font-size: 16px;'>{weather_info['wind_speed']} km/h</p></div>", unsafe_allow_html=True)
               
                # Weather tips
                weather_tips = get_weather_tips(weather_info)
                if weather_tips:
                    st.info(" ".join(weather_tips))
                
                # Must-visit places status
                if found_places or not_found_places:
                    st.subheader("📍 Must-Visit Places Status")
                    
                    if found_places:
                        st.success("✅ Found in your itinerary:")
                        for place in found_places:
                            st.write(f"• **{place['input']}** → {place['matched_place']['name']}")
                    
                    if not_found_places:
                        st.warning("⚠️ Could not find these places in our database:")
                        for place in not_found_places:
                            st.write(f"• {place}")
                
                # Itinerary
                st.subheader("🗓️ Your Daily Itinerary")
                if not itinerary:
                    st.warning("Unable to generate itinerary. Please try different preferences or city.")
                    return
                
                # Display itinerary in a nice format
                for i, item in enumerate(itinerary):
                    place = item['place']
                    
                    # Create expandable section for each place
                    with st.expander(f"⏰ {item['time']} - {place['name']} {'⭐' if place.get('is_must_visit') else ''}", expanded=True):
                        
                        # Travel information
                        if item['travel_time']:
                            st.info(f"🚗 {item['travel_time']} min by {item['transport_mode']}")
                        
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**📍 Address:** {place.get('address', 'N/A')}")
                            st.write(f"**⏱️ Duration:** {item['duration']} minutes")
                            
                            day_name = travel_date.strftime("%A")
                            hours_column = f"{day_name}_hours"
                            if hours_column in place:
                                opening_hours = place[hours_column]
                                if pd.isna(opening_hours) or opening_hours == '' or str(opening_hours).lower() == 'nan':
                                    opening_hours = 'Hours not available'
                            else:
                                opening_hours = 'Hours not available'
                            st.write(f"**🕐 Opening Hours ({day_name.capitalize()}):** {opening_hours}")

                            st.write(f"**🏷️ Categories:** {place.get('categories', 'N/A')}")

                            if place.get('tip'):
                                st.write(f"**💡 Tip:** {place.get('tip')}")

                            if place.get('is_must_visit'):
                                st.success("⭐ Must-visit place")
                        
                        with col2:
                            if place.get('stars'):
                                st.metric("Rating", f"{place['stars']}/5.0 ⭐")
                            
                            # Type badge
                            if item['type'] == 'attraction':
                                st.success("🎯 Attraction")
                            else:
                                st.info("🍽️ Restaurant")
                
                
                st.subheader("💡 Personalized Travel Tips")
                st.markdown(travel_tips)

                # Get names of restaurants already in itinerary to avoid duplication
                itinerary_restaurant_names = [
                    item['place']['name'] for item in itinerary 
                    if item['type'] == 'restaurant'
                ]

                # Get featured restaurants (excluding ones already in itinerary)
                featured_restaurants = get_restaurant_recommendations_for_tips(
                    df, normalized_city, travel_date, exclude_names=itinerary_restaurant_names
                )

                # Display featured restaurants in expandable section
                if featured_restaurants:
                    st.markdown("---")
                    with st.expander("🍽️ **Additional Local Restaurant Recommendations** (Click to expand)", expanded=False):
                        st.markdown("*Discover more highly-rated local restaurants:*")
        
                        for i, restaurant in enumerate(featured_restaurants, 1):
                            st.markdown(f"### {i}. {restaurant['name']}")
            
                            col1, col2 = st.columns([2, 1])
            
                            with col1:
                                st.write(f"**📍 Address:** {restaurant.get('address', 'N/A')}")
                                st.write(f"**🏷️ Categories:** {restaurant.get('categories', 'N/A')}")
                
                                # Display opening hours
                                day_name = travel_date.strftime("%A")
                                hours_column = f"{day_name}_hours"
                                opening_hours = restaurant.get(hours_column, 'Hours not available')
                                if pd.isna(opening_hours) or opening_hours == '' or str(opening_hours).lower() == 'nan':
                                    opening_hours = 'Hours not available'
                                st.write(f"**🕐 Opening Hours ({day_name}):** {opening_hours}")
                
                                if restaurant.get('tip'):
                                    st.write(f"**💡 Tip:** {restaurant.get('tip')}")
            
                            with col2:
                                if restaurant.get('stars'):
                                    st.metric("Rating", f"{restaurant['stars']}/5.0 ⭐")
            
                            if i < len(featured_restaurants):
                                st.markdown("---")
                
                # Summary
                st.subheader("📊 Trip Summary")
                
                total_attractions = len([item for item in itinerary if item['type'] == 'attraction'])
                total_restaurants = len([item for item in itinerary if item['type'] == 'restaurant'])
                total_duration = sum(item['duration'] for item in itinerary)
                total_travel_time = sum(item['travel_time'] for item in itinerary if item['travel_time'])
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🎯 Attractions", total_attractions)
                with col2:
                    st.metric("🍽️ Restaurants", total_restaurants)
                with col3:
                    st.metric("⏱️ Total Visit Time", f"{total_duration//60}h {total_duration%60}m")
                with col4:
                    st.metric("🚗 Travel Time", f"{total_travel_time} min")
                
                # Export option
                st.subheader("📤 Export Itinerary")
                
                # Create downloadable itinerary text
                itinerary_text = f"""
# {normalized_city} Travel Itinerary - {travel_date.strftime('%Y-%m-%d')}

## Weather Forecast
- Condition: {weather_info['condition']}
- Temperature: {weather_info['temp_min']}°C - {weather_info['temp_max']}°C
- Humidity: {weather_info['humidity']}%
- Wind Speed: {weather_info['wind_speed']} km/h

## Daily Schedule
"""
                
                for item in itinerary:
                    place = item['place']
                    itinerary_text += f"""
### {item['time']} - {place['name']} {'⭐' if place.get('is_must_visit') else ''}
- **Address:** {place.get('address', 'N/A')}
- **Duration:** {item['duration']} minutes
- **Type:** {item['type'].title()}
- **Rating:** {place.get('stars', 'N/A')}/5.0
- **Categories:** {place.get('categories', 'N/A')}
"""
                    if item['travel_time']:
                        itinerary_text += f"- **Travel:** {item['travel_time']} min by {item['transport_mode']}\n"
                
                itinerary_text += f"""
## Trip Summary
- Total Attractions: {total_attractions}
- Total Restaurants: {total_restaurants}
- Total Visit Time: {total_duration//60}h {total_duration%60}m
- Total Travel Time: {total_travel_time} min

## Travel Tips
{travel_tips}
"""           
                st.download_button(
                    label="📥 Download Itinerary (Text)",
                    data=itinerary_text,
                    file_name=f"{normalized_city}_itinerary_{travel_date.strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"An error occurred while generating your itinerary: {str(e)}")
                st.error("Please try again or contact support.")


st.markdown("---")


if __name__ == "__main__":
    main()

