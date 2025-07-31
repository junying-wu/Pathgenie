# Pathgenie-Your AI Travel Agent

## **Brief Intro**
Planning a trip can be a time-consuming and overwhelming task, especially for those unsure where to begin or how to prioritize. It often involves navigating multiple platforms, filtering through outdated reviews, and managing the fear of missing out on key experiences.

Pathgenie is an AI-powered travel agent designed to streamline this process. By combining local data with natural language understanding, it generates personalized, weather-aware itineraries tailored to individual preferences. From recommended attractions and dining options to smart scheduling aligned with weather conditions and meal times, Pathgenie helps users travel smarter and with greater ease.


## Workflow

1. **User Input:** Provide city, date, must-visit places, and preferences.  
2. **Data Loading & Filtering:** Load and filter business data (restaurants, attractions) by meal times and categories.  
3. **Weather Integration:** Fetch weather forecast via WeatherAPI.  
4. **Content Generation:** Generate city intro and weather-based travel tips using Ollama LLM.  
5. **Itinerary Creation:** Build itinerary including must-visit and recommended places, meal breaks, adjusted for preferences and date. 
6. **Travel Time Estimation:** Calculate approximate travel times between stops.  
7. **Output:** Display itinerary with detailed info and provide download option as a text file.
   
## **Data Instructions**

Pathgenie uses a **local dataset** containing information on restaurants, attractions, and their operating hours.

###  Recommended Dataset Structure

When building Pathgenie, I used the [[Yelp Open Dataset](https://www.yelp.com/dataset/challenge)](https://www.yelp.com/dataset/challenge). Specifically:

* I started with 'yelp_academic_dataset_business.json'
* Then merged it with 'yelp_academic_dataset_tip.json' using Excel’s **VLOOKUP** to include user tips

This generated a custom '.csv' file used in the app

If you're using your own dataset, **make sure it contains the following required columns**:

    'name', 'address', 'city', 'categories', 'stars',
    'Monday_hours', 'Tuesday_hours', 'Wednesday_hours', 'Thursday_hours',
    'Friday_hours', 'Saturday_hours', 'Sunday_hours', 'tip'

> Each row should represent a business or place of interest, with valid open/close hours and a short tip or review for display

### How to Link the Dataset in Code

In your code, locate **line 17**, which defines the dataset path:

    file_path = "current path with the location of the CSV file"

Update "current path with the location of the CSV file" with the **correct path** to your own '.csv' file, for example:

    file_path = r"your/path/to/your_dataset.csv"

## **How to Use Pathgenie**

Follow the steps below to run **Pathgenie** on your local machine:

1. Set Up Ollama (https://ollama.com/) and Run the LLM

* Download and install **[Ollama](https://ollama.com/)**

* Open **Command Prompt** and run "ollama serve" to start the lightweight local language model

* When that is finished, type "ollama pull phi3:mini" in the command prompt
2. Configure the Weather API

* Visit [[WeatherAPI](https://www.weatherapi.com/)](https://www.weatherapi.com/) and sign up for an account

* After signing up, copy your **API key** from your account dashboard

* In your codebase, locate **line 527** and replace the placeholder with your actual key:

   `API_KEY = "your_actual_api_key_here"`

3. Launch Pathgenie

* Open Command Prompt, navigate to your project folder, and run the Streamlit app using the following command (replace app.py with your actual main script filename):

   `python -m streamlit run app.py`

## 🎉After this step... 

You will then be redirected to a page like this:

<img width="691" height="580" alt="Image" src="https://github.com/user-attachments/assets/b885ab5b-1890-4ede-9deb-ee10e406df9e" />

Fill in your travel preferences and click **"Generate Itinerary"**.

For example, if I enter Edmonton, Pathgenie will first greet me with a concise and engaging introduction to the city, highlighting its unique history and cultural landmarks:

<img width="689" height="450" alt="Image" src="https://github.com/user-attachments/assets/c80ec996-2f70-4a53-a162-1479f1cba5f8" />

Right below, you’ll see a detailed weather forecast for your selected travel date:

<img width="677" height="234" alt="Image" src="https://github.com/user-attachments/assets/fa0ef302-111d-4423-ae1d-eaeee7c8da1a" />

Pathgenie will also identify which of your **must-visit places** are available in its database and include them in the itinerary.

It then generates a time-ordered plan that combines attractions and restaurants tailored to your preferences and schedule:

<img width="702" height="547" alt="Image" src="https://github.com/user-attachments/assets/f1a4607b-9451-4ef5-8176-3c17004fba87" />

<img width="682" height="367" alt="Image" src="https://github.com/user-attachments/assets/7172d9ca-980d-4174-8625-8acb1acacaf8" />

You’ll also receive personalized travel tips related to transportation, shopping, and local dining:

<img width="677" height="428" alt="Image" src="https://github.com/user-attachments/assets/6533c1e7-2d68-46e6-ad2f-bac5e2087af5" />

If you're looking for more restaurant options beyond those in the itinerary, we’ve got you covered:

<img width="569" height="571" alt="Image" src="https://github.com/user-attachments/assets/ccde04cc-853a-4fc1-ae47-c1f9166c37d5" />

Finally, you can download the entire itinerary as a text file:

<img width="560" height="206" alt="Image" src="https://github.com/user-attachments/assets/e2e1a5b0-f8c5-4cf5-9be1-8bcb21e46e50" />
And the file downloaded will be like this:

<img width="844" height="754" alt="Image" src="https://github.com/user-attachments/assets/f94efbdf-2ab7-424f-8cb3-8ba2946c1896" />
