import os
import pandas as pd

# Create the 'interpolated' folder if it doesn't exist
os.makedirs('interpolated', exist_ok=True)

# Get a list of all CSV files in the current directory
csv_files = [f for f in os.listdir() if f.endswith('.csv')]

# Process each CSV file
for file in csv_files:
    # Load the yearly GDP per capita data
    yearly_data = pd.read_csv(file)
    
    # Convert the 'date' column to datetime format
    yearly_data['date'] = pd.to_datetime(yearly_data['date'], errors='coerce')
    
    # Drop rows where 'date' couldn't be converted
    yearly_data = yearly_data.dropna(subset=['date'])
    
    # Drop duplicate rows based on the 'date' column
    yearly_data = yearly_data.drop_duplicates(subset='date', keep='first')
    
    # Set the 'date' column as the index
    yearly_data.set_index('date', inplace=True)
    
    # Resample the data to a monthly frequency and interpolate linearly
    monthly_data = yearly_data.resample('M').interpolate(method='linear')
    
    # Define the output file path
    output_file = os.path.join('interpolated', file)
    
    # Save the interpolated monthly data to the new CSV file
    monthly_data.to_csv(output_file)
    
    # Print confirmation
    print(f"Interpolated data for {file} saved to {output_file}")
