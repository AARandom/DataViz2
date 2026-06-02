import pandas as pd
from pathlib import Path

def process_peaks():
    # Set current directory as the base
    BASE_DIR = Path.cwd()
    
    # Input files
    journey_files = [
        (BASE_DIR / "journey_to_work_vista_2024_2025.csv", "Work"),
        (BASE_DIR / "journey_to_education_vista_2024_2025.csv", "Education"),
    ]

    journey_frames = []
    
    for path, label in journey_files:
        print(f"Processing {path.name}...")
        
        # Load only necessary columns
        df = pd.read_csv(path, usecols=[
            "main_journey_mode",
            "start_time",
            "start_LGA",
            "end_LGA",
            "journey_weight",
        ])
        
        # Filter for Train and convert time to numeric
        journeys = df[df["main_journey_mode"] == "Train"].copy()
        journeys["start_time"] = pd.to_numeric(journeys["start_time"], errors="coerce")

        # Define PTV Peak Periods (minutes from midnight)
        # AM: 7:00 am - 9:00 am (420 - 540)
        # PM: 3:00 pm - 7:00 pm (900 - 1140)
        journeys["peak_period"] = "Other"
        journeys.loc[journeys["start_time"].between(420, 540, inclusive="both"), "peak_period"] = "AM Peak"
        journeys.loc[journeys["start_time"].between(900, 1140, inclusive="both"), "peak_period"] = "PM Peak"

        # Drop non-peak data
        journeys = journeys[journeys["peak_period"] != "Other"].copy()

        # Classify Direction (Inbound/Outbound relative to Melbourne LGA)
        journeys["direction"] = "Other"
        journeys.loc[
            (journeys["start_LGA"] != "Melbourne (C)") & (journeys["end_LGA"] == "Melbourne (C)"),
            "direction",
        ] = "Inbound"
        journeys.loc[
            (journeys["start_LGA"] == "Melbourne (C)") & (journeys["end_LGA"] != "Melbourne (C)"),
            "direction",
        ] = "Outbound"

        journeys["journey_type"] = label
        journeys["passenger_count"] = pd.to_numeric(journeys["journey_weight"], errors="coerce")
        journey_frames.append(journeys)

    # Combine and save
    final_df = pd.concat(journey_frames, ignore_index=True)
    output_file = BASE_DIR / "vista_train_peak_direction.csv"
    final_df.to_csv(output_file, index=False)
    
    print(f"Success! Output saved to: {output_file}")

if __name__ == "__main__":
    process_peaks()