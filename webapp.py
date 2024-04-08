import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained models and define functions
def load_models():
    base_models = [
        ('Random Forest', RandomForestRegressor(random_state=42)),
        ('GB', GradientBoostingRegressor(random_state=42)),
        ('Ridge', Ridge()),
    ]
    meta_model = LinearRegression()
    stacking_reg = StackingRegressor(estimators=base_models, final_estimator=meta_model)

    models = {
        'Stacking': stacking_reg
    }

    return models

def predict_yield_for_all_crops(pipeline_paths, new_input, crop_district_averages):
    models = load_models()
    results = {}
    for crop, crop_path in pipeline_paths.items():
        try:
            pipeline_path = f'{crop_path}/pipeline_{crop}_stacking.joblib'
            trained_pipeline = joblib.load(pipeline_path)

            # Adjust new_input with district averages
            crop_district_avg = crop_district_averages[crop]
            for column, value in new_input.items():
                if value == 0 and column in crop_district_avg:
                    district = new_input['District'].lower()
                    district_avg = crop_district_avg[column].get(district)
                    if district_avg is not None:
                        new_input[column] = district_avg

            prediction_df = pd.DataFrame([new_input])
            predicted_yield = trained_pipeline.predict(prediction_df)[0]
            results[f'{crop}'] = {
                'Predicted Yield': predicted_yield,
                'Nutrient Recommendations': classify_soil_nutrient(new_input['N'], new_input['P2O5'], new_input['K2O'], crop),
            }
        except FileNotFoundError:
            print(f'Pipeline for {crop} not found.')

    return results

def classify_soil_nutrient(N, P2O5, K2O, predicted_class):
    nutrient_classifications = []
    predicted_class = predicted_class[0].upper() + predicted_class[1:]

    # print(predicted_class)
    # Classify N
    if N < 280:
        nutrient_classifications.append("N: Low")
        # print('n is low')
        if predicted_class == 'Sugarcane':
            nutrient_classifications.append("Recommended Urea dose: 343.3 kg/ha")
            # print("in sugarcane n is low")
        elif predicted_class == 'Wheat':
            nutrient_classifications.append("Recommended Urea dose: 343.3 kg/ha")
        elif predicted_class == 'Potato':
            nutrient_classifications.append("Recommended Urea dose: 300.91 kg/ha")
        elif predicted_class == 'Mustard':
            nutrient_classifications.append("Recommended Urea dose: 72.28 kg/ha")
        elif predicted_class == 'Bajra':
            nutrient_classifications.append("Recommended Urea dose: 80 - 100 kg/ha")
        elif predicted_class == 'Rice':
            nutrient_classifications.append("Recommended Urea dose: 343.3 kg/ha")
    elif N >= 280 and N <= 560:
        nutrient_classifications.append("N: Medium")
        if predicted_class == 'Sugarcane' :
            nutrient_classifications.append("Recommended Urea dose: 274.64 kg/ha")
        elif predicted_class == 'Wheat' :
            nutrient_classifications.append("Recommended Urea dose: 274.64 kg/ha")
        elif predicted_class == 'Potato' :
            nutrient_classifications.append("Recommended Urea dose: 240.73 kg/ha")
        elif predicted_class == 'Mustard' :
            nutrient_classifications.append("Recommended Urea dose: 57.83 kg/ha")
        elif predicted_class == 'Bajra' :
            nutrient_classifications.append("Recommended Urea dose: 80 - 100 kg/ha")
        elif predicted_class == 'Rice' :
            nutrient_classifications.append("Recommended Urea dose: 274.64 kg/ha")
    else:
        nutrient_classifications.append("N: High")
        if predicted_class == 'Sugarcane' :
            nutrient_classifications.append("Recommended Urea dose: 205.98 kg/ha")
        elif predicted_class == 'Wheat' :
            nutrient_classifications.append("Recommended Urea dose: 205.98 kg/ha")
        elif predicted_class == 'Potato' :
            nutrient_classifications.append("Recommended Urea dose: 180.54 kg/ha")
        elif predicted_class == 'Mustard' :
            nutrient_classifications.append("Recommended Urea dose: 43.37 kg/ha")
        elif predicted_class == 'Bajra' :
            nutrient_classifications.append("Recommended Urea dose: 80 - 100 kg/ha")
        elif predicted_class == 'Rice' :
            nutrient_classifications.append("Recommended Urea dose: 205.98 kg/ha")

    # Classify P2O5
    if P2O5 < 25:
        nutrient_classifications.append("P2O5: Low")
        if predicted_class == 'Sugarcane' :
            nutrient_classifications.append("Recommended DAP dose: 162.75 kg/ha")
        elif predicted_class == 'Wheat':
            nutrient_classifications.append("Recommended DAP dose: 162.75 kg/ha")
        elif predicted_class == 'Potato' :
            nutrient_classifications.append("Recommended DAP dose: 271.25 kg/ha")
        elif predicted_class == 'Mustard' :
            nutrient_classifications.append("Recommended DAP dose: 162.75 kg/ha")
        elif predicted_class == 'Bajra' :
            nutrient_classifications.append("Recommended DAP dose: 40 - 50 kg/ha")
        elif predicted_class == 'Rice' :
            nutrient_classifications.append("Recommended DAP dose: 162.75 kg/ha")
    elif P2O5 >= 25 and P2O5 <= 56:
        nutrient_classifications.append("P2O5: Medium")
        if predicted_class == 'Sugarcane' :
            nutrient_classifications.append("Recommended DAP dose: 130.2 kg/ha")
        elif predicted_class == 'Wheat' :
            nutrient_classifications.append("Recommended DAP dose: 130.2 kg/ha")
        elif predicted_class == 'Potato' :
            nutrient_classifications.append("Recommended DAP dose: 217 kg/ha")
        elif predicted_class == 'Mustard' :
            nutrient_classifications.append("Recommended DAP dose: 130.2 kg/ha")
        elif predicted_class == 'Bajra' :
            nutrient_classifications.append("Recommended DAP dose: 80 - 100 kg/ha")
        elif predicted_class == 'Rice':
            nutrient_classifications.append("Recommended DAP dose: 130.2 kg/ha")
    else:
        nutrient_classifications.append("P2O5: High")
        if predicted_class == 'Sugarcane' :
            nutrient_classifications.append("Recommended DAP dose: 97.65 kg/ha")
        elif predicted_class == 'Wheat' :
            nutrient_classifications.append("Recommended DAP dose: 97.65 kg/ha")
        elif predicted_class == 'Potato':
            nutrient_classifications.append("Recommended DAP dose: 162.75 kg/ha")
        elif predicted_class == 'Mustard':
            nutrient_classifications.append("Recommended DAP dose: 97.65 kg/ha")
        elif predicted_class == 'Bajra' :
            nutrient_classifications.append("Recommended DAP dose: 80 - 100 kg/ha")
        elif predicted_class == 'Rice' :
            nutrient_classifications.append("Recommended DAP dose: 97.65 kg/ha")

    # Classify K2O
    if K2O < 140:
        nutrient_classifications.append("K2O: Low")
        if predicted_class == 'Sugarcane' :
            nutrient_classifications.append("Recommended MOP dose: 83.5 kg/ha")
        elif predicted_class == 'Wheat' :
            nutrient_classifications.append("Recommended MOP dose: 83.5 kg/ha")
        elif predicted_class == 'Potato' :
            nutrient_classifications.append("Recommended MOP dose: 208.75 kg/ha")
        elif predicted_class == 'Mustard' :
            nutrient_classifications.append("Recommended MOP dose: 62.625 kg/ha")
        elif predicted_class == 'Bajra' :
            nutrient_classifications.append("Recommended MOP dose: 40 kg/ha")
        elif predicted_class == 'Rice' :
            nutrient_classifications.append("Recommended MOP dose: 125.25 kg/ha")
    elif K2O >= 140 and K2O <= 280:
        nutrient_classifications.append("K2O: Medium")
        if predicted_class == 'Sugarcane' :
            nutrient_classifications.append("Recommended MOP dose: 66.8 kg/ha")
        elif predicted_class == 'Wheat' :
            nutrient_classifications.append("Recommended MOP dose: 66.8 kg/ha")
        elif predicted_class == 'Potato':
            nutrient_classifications.append("Recommended MOP dose: 167 kg/ha")
        elif predicted_class == 'Mustard' :
            nutrient_classifications.append("Recommended MOP dose: 50.1 kg/ha")
        elif predicted_class == 'Bajra' :
            nutrient_classifications.append("Recommended MOP dose: 80 - 100 kg/ha")
        elif predicted_class == 'Rice' :
            nutrient_classifications.append("Recommended MOP dose: 100.2 kg/ha")
    else:
        nutrient_classifications.append("K2O: High")
        if predicted_class == 'Sugarcane' :
            nutrient_classifications.append("Recommended MOP dose: 50.1 kg/ha")
        elif predicted_class == 'Wheat' :
            nutrient_classifications.append("Recommended MOP dose: 50.1 kg/ha")
        elif predicted_class == 'Potato' :
            nutrient_classifications.append("Recommended MOP dose: 125.5 kg/ha")
        elif predicted_class == 'Mustard' :
            nutrient_classifications.append("Recommended MOP dose: 37.575 kg/ha")
        elif predicted_class == 'Bajra' :
            nutrient_classifications.append("Recommended MOP dose: 80 - 100 kg/ha")
        elif predicted_class == 'Rice' :
            nutrient_classifications.append("Recommended MOP dose: 75.15 kg/ha")
    # print(nutrient_classifications)
    return (nutrient_classifications)

# Function to calculate district-wise averages for weather parameters
def calculate_district_averages_last_5_years(df_crop):
    last_5_years_data = df_crop[df_crop['Start_Year'].isin(range(2017, 2022))]  # Assuming data for 2022 is not available yet
    numeric_columns = last_5_years_data.select_dtypes(include=[np.number])
    
    # Add the 'District' column to the numeric columns for grouping
    numeric_columns['District'] = last_5_years_data['District']
    
    # Calculate the mean for numeric columns grouped by 'District'
    district_averages = numeric_columns.groupby('District').mean().to_dict()

    return district_averages

# Function to compare predicted yields with average yields and sort crops
def compare_and_sort_crops(average_yields, predicted_results):
    differences = {}
    for crop, data in predicted_results.items():
        # Calculate the difference in yield
        yield_difference = data['Predicted Yield'] - average_yields[crop]
        differences[crop] = {
            'Yield Difference': yield_difference,
            'Nutrient Recommendations': data['Nutrient Recommendations']
        }

    # Sort crops based on yield difference
    sorted_crops = sorted(differences.keys(), key=lambda x: differences[x]['Yield Difference'], reverse=True)[:3]
    # return sorted_crops
    return {crop: differences[crop] for crop in sorted_crops}

def average_col(df):
    return df['Yield (Tonnes/Hectare)'].mean()



# Main Streamlit app
def main():
    st.title('Crop Yield Prediction and Nutrient Recommendations')

    # User input (excluding weather parameters)
    # district = st.text_input('Enter District Name:')
    district_options = ['agra', 'aligarh', 'allahabad', 'ambedkar nagar', 'amethi', 'amroha', 'auraiya', 'azamgarh', 'baghpat', 'bahraich', 'ballia', 'balrampur', 'banda', 'barabanki', 'bareilly', 'basti', 'bijnor', 'budaun', 'bulandshahr', 'chandauli', 'chitrakoot', 'deoria', 'etah', 'etawah', 'faizabad', 'farrukhabad', 'fatehpur', 'firozabad', 'gautam buddha nagar', 'ghaziabad', 'ghazipur', 'gonda', 'gorakhpur', 'hamirpur', 'hapur', 'hardoi', 'hathras', 'jalaun', 'jaunpur', 'jhansi', 'kannauj', 'kanpur dehat', 'kanpur nagar', 'kasganj', 'kaushambi', 'kheri', 'kushi nagar', 'lalitpur', 'lucknow', 'maharajganj', 'mahoba', 'mainpuri', 'mathura', 'mau', 'meerut', 'mirzapur', 'moradabad', 'muzaffarnagar', 'pilibhit', 'pratapgarh', 'rae bareli', 'rampur', 'saharanpur', 'sambhal', 'sant kabeer nagar', 'sant ravidas nagar', 'shahjahanpur', 'shamli', 'shravasti', 'siddharth nagar', 'sitapur', 'sonbhadra', 'sultanpur', 'unnao', 'varanasi']
    district = st.selectbox('Select District:', district_options)
    area = st.number_input('Enter Area (Hectare):', min_value=0)
    season = st.selectbox('Select Season:', ['Kharif', 'Rabi'])
    N = st.number_input('N (Nitrogen):')
    P2O5 = st.number_input('P2O5 (Phosphorus):')
    K2O = st.number_input('K2O (Potassium):')
    submit_button = st.button('Submit')

    # Load crop pipeline paths (update these paths for GitHub Codespaces)
    pipeline_paths = {
        'sugarcane': 'Models/',#C:/Arpit-New/ISRO_RRSCN/web-app-required/
        'potato': 'Models/',
        'rice': 'Models/',
        'wheat': 'Models/',
        'mustard': 'Models/',
        'bajra': 'Models/',
    }

    # Load weather data CSVs and calculate district-wise averages
    df_sugarcane = pd.read_csv('CSV/sugarcane.csv')  # Update path accordingly C:/Arpit-New/ISRO_RRSCN/web-app-required/
    df_potato = pd.read_csv('CSV/potato.csv')  # Update path accordingly
    df_rice = pd.read_csv('CSV/rice.csv')  # Update path accordingly
    df_wheat = pd.read_csv('CSV/wheat.csv')  # Update path accordingly
    df_mustard = pd.read_csv('CSV/mustard.csv')  # Update path accordingly
    df_bajra = pd.read_csv('CSV/bajra.csv')  # Update path accordingly
    
    # Calculate district-wise averages for each crop's weather data
    crop_district_averages = {
        'sugarcane': calculate_district_averages_last_5_years(df_sugarcane),
        'potato': calculate_district_averages_last_5_years(df_potato),
        'rice': calculate_district_averages_last_5_years(df_rice),
        'wheat': calculate_district_averages_last_5_years(df_wheat),
        'mustard': calculate_district_averages_last_5_years(df_mustard),
        'bajra': calculate_district_averages_last_5_years(df_bajra),
    }

    yield_average = {}
    dfs = [df_sugarcane, df_potato, df_bajra, df_mustard, df_wheat, df_rice]
    crops = ['sugarcane', 'potato', 'rice', 'wheat','mustard', 'bajra' ]
    
    for i, crop in enumerate(crops):
        yield_average[crop] = average_col(dfs[i])
    
    
    # Create new_input dictionary with user inputs
    new_input = {
        'Area (Hectare)': area,
        'Start_Year': 2022,
        'District': district.lower(),
        'Season': season,
        'Humidity_Sowing': 0,  # Placeholder value for humidity at sowing
        'Humidity_Full': 0,  # Placeholder value for humidity at full
        'Rainfall_Sowing': 0,  # Placeholder value for rainfall at sowing
        'Rainfall_Full': 0,  # Placeholder value for rainfall at full
        'Max_Temperature_Sowing': 0,  # Placeholder value for max temp at sowing
        'Max_Temperature_Full': 0,  # Placeholder value for max temp at full
        'Min_Temperature_Sowing': 0,  # Placeholder value for min temp at sowing
        'Min_Temperature_Full': 0,  # Placeholder value for min temp at full
        'N': N,
        'P2O5': P2O5,
        'K2O': K2O,
    }

    
    if submit_button:
        # Predict yield for all crops
        predicted_yields = predict_yield_for_all_crops(pipeline_paths, new_input, crop_district_averages)

        # Compare predicted yields with average yields and sort crops
        top_crops_with_details = compare_and_sort_crops(average_yields=yield_average, predicted_results=predicted_yields)

        st.header('Top 3 Recommended Crops with Historical Yields:')
        for crop, details in top_crops_with_details.items():
            # Create two columns: one for text data and the other for the plot
            col1, col2 = st.columns([2, 3])  # Adjust the ratio as needed
            
            with col1:
                st.subheader(f"{crop.capitalize()} Recommendations:")
                st.write(f"Predicted Yield Difference: {details['Yield Difference']:.2f} Tonnes/Hectare")
                st.write('Nutrient Recommendations:')   
                for recommendation in details['Nutrient Recommendations']:
                    st.write(f"- {recommendation}")
            
            # Display the historical yield plot in the second column
            with col2:
                df_map = {
                    'sugarcane': df_sugarcane,
                    'potato': df_potato,
                    'rice': df_rice,
                    'wheat': df_wheat,
                    'mustard': df_mustard,
                    'bajra': df_bajra,
                }
                df_crop = df_map[crop].copy()
                df_crop['District'] = df_crop['District'].str.lower()
                district_data = df_crop[df_crop['District'] == district.lower()].copy()

                if not district_data.empty:
                    plt.figure(figsize=(8, 6))  # Increase the figure size for better visibility
                    district_data = district_data.sort_values(by='Start_Year')
                    plt.plot(district_data['Start_Year'], district_data['Yield (Tonnes/Hectare)'], marker='o', linestyle='-', label=crop.capitalize())
                    plt.title(f'Historical Yield for {crop.capitalize()} in {district.capitalize()}')
                    plt.xlabel('Year')
                    plt.ylabel('Yield (Tonnes/Hectare)')
                    
                    years = district_data['Start_Year'].unique()
                    num_years = len(years)
                    
                    if num_years <= 10:
                        plt.xticks(years, [int(year) for year in years])
                    else:
                        # Display every 2nd or 3rd year on the x-axis
                        step = 2 if num_years <= 20 else 3
                        plt.xticks(years[::step], [int(year) for year in years[::step]])
                    
                    plt.xticks(rotation=45)  # Rotate the x-axis labels by 45 degrees
                    plt.tight_layout()  # Adjust the layout to prevent label overlapping
                    
                    plt.grid(True)
                    plt.legend()
                    col2.pyplot(plt)
                else:
                    col2.write(f"No historical yield data available for {crop.capitalize()} in {district.capitalize()}.")
           

if __name__ == "__main__":
    main()
