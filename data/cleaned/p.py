import pandas as pd

# 1. Load the datasets
try:
    df_19 = pd.read_csv('station_volumes_2019_2020.csv')
    df_24 = pd.read_csv('station_volumes_2024_2025.csv')
    print("Files successfully loaded.")
except FileNotFoundError:
    print("Error: Make sure 'station_volumes_2019_2020.csv' and 'station_volumes_2024_2025.csv'")
    print("are located in the exact same directory as this script.")
    exit()

# 2. Define train line networks mapping based on your requirements
lines_dict = {
    "Alamein": [
        "Flinders Street", "Richmond", "East Richmond", "Burnley", "Hawthorn", 
        "Glenferrie", "Auburn", "Camberwell", "Riversdale", "Willison", 
        "Hartwell", "Burwood", "Ashburton", "Alamein"
    ],
    "Belgrave": [
        "Flinders Street", "Richmond", "East Richmond", "Burnley", "Hawthorn", 
        "Glenferrie", "Auburn", "Camberwell", "East Camberwell", "Canterbury", 
        "Chatham", "Surrey Hills", "Mont Albert", "Box Hill", "Laburnum", 
        "Blackburn", "Nunawading", "Mitcham", "Heatherdale", "Ringwood", 
        "Heathmont", "Bayswater", "Boronia", "Ferntree Gully", "Upper Ferntree Gully", 
        "Upwey", "Tecoma", "Belgrave", "Union"
    ],
    "Lilydale": [
        "Flinders Street", "Richmond", "East Richmond", "Burnley", "Hawthorn", 
        "Glenferrie", "Auburn", "Camberwell", "East Camberwell", "Canterbury", 
        "Chatham", "Surrey Hills", "Mont Albert", "Box Hill", "Laburnum", 
        "Blackburn", "Nunawading", "Mitcham", "Heatherdale", "Ringwood", 
        "Ringwood East", "Croydon", "Mooroolbark", "Lilydale", "Union"
    ],
    "Glen Waverley": [
        "Flinders Street", "Richmond", "East Richmond", "Burnley", "Heyington", 
        "Kooyong", "Tooronga", "Gardiner", "Glen Iris", "Darling", "East Malvern", 
        "Holmesglen", "Jordanville", "Mount Waverley", "Syndal", "Glen Waverley"
    ],
    "Hurstbridge": [
        "Flinders Street", "Southern Cross", "Flagstaff", "Melbourne Central", 
        "Parliament", "Jolimont", "West Richmond", "North Richmond", "Collingwood", 
        "Victoria Park", "Clifton Hill", "Westgarth", "Dennis", "Fairfield", 
        "Alphington", "Darebin", "Ivanhoe", "Eaglemont", "Heidelberg", "Rosanna", 
        "Macleod", "Watsonia", "Greensborough", "Montmorency", "Eltham", 
        "Diamond Creek", "Wattle Glen", "Hurstbridge"
    ],
    "Mernda": [
        "Flinders Street", "Southern Cross", "Flagstaff", "Melbourne Central", 
        "Parliament", "Jolimont", "West Richmond", "North Richmond", "Collingwood", 
        "Victoria Park", "Clifton Hill", "Rushall", "Merri", "Northcote", 
        "Thornbury", "Croxton", "Preston", "Regent", "Reservoir", "Ruthven", 
        "Keon Park", "Thomastown", "Lalor", "Epping", "South Morang", 
        "Middle Gorge", "Hawkstowe", "Mernda", "Bell"
    ],
    "Craigieburn": [
        "Flinders Street", "Southern Cross", "Flagstaff", "Melbourne Central", 
        "Parliament", "North Melbourne", "Kensington", "Newmarket", "Ascot Vale", 
        "Moonee Ponds", "Essendon", "Glenbervie", "Strathmore", "Pascoe Vale", 
        "Oak Park", "Glenroy", "Jacana", "Broadmeadows", "Coolaroo", 
        "Roxburgh Park", "Craigieburn"
    ],
    "Upfield": [
        "Flinders Street", "Southern Cross", "Flagstaff", "Melbourne Central", 
        "Parliament", "North Melbourne", "Macaulay", "Flemington Bridge", 
        "Royal Park", "Jewell", "Brunswick", "Anstey", "Moreland", "Coburg", 
        "Batman", "Merlynston", "Fawkner", "Gowrie", "Upfield"
    ],
    "Frankston": [
        "Flinders Street", "Southern Cross", "Flagstaff", "Melbourne Central", 
        "Parliament", "Richmond", "South Yarra", "Hawksburn", "Toorak", 
        "Armadale", "Malvern", "Caulfield", "Glenhuntly", "Ormond", "McKinnon", 
        "Mckinnon", "Bentleigh", "Patterson", "Moorabbin", "Southland", 
        "Cheltenham", "Mentone", "Parkdale", "Mordialloc", "Aspendale", 
        "Edithvale", "Chelsea", "Bonbeach", "Carrum", "Seaford", "Kananook", 
        "Frankston", "Highett"
    ],
    "Werribee": [
        "Flinders Street", "Southern Cross", "Flagstaff", "Melbourne Central", 
        "Parliament", "North Melbourne", "South Kensington", "Footscray", 
        "Seddon", "Yarraville", "Spotswood", "Newport", "Seaholme", "Altona", 
        "Westona", "Laverton", "Aircraft", "Williams Landing", "Hoppers Crossing", 
        "Werribee"
    ],
    "Williamstown": [
        "Flinders Street", "Southern Cross", "Flagstaff", "Melbourne Central", 
        "Parliament", "North Melbourne", "South Kensington", "Footscray", 
        "Seddon", "Yarraville", "Spotswood", "Newport", "North Williamstown", 
        "Williamstown Beach", "Williamstown"
    ],
    "Sandringham": [
        "Flinders Street", "Richmond", "South Yarra", "Prahran", "Windsor", 
        "Balaclava", "Ripponlea", "Elsternwick", "Gardenvale", "North Brighton", 
        "Middle Brighton", "Brighton Beach", "Hampton", "Sandringham"
    ],
    "Sunbury": [
        "Sunbury", "Diggers Rest", "Watergardens", "Keilor Plains", "St Albans", 
        "Ginifer", "Albion", "Sunshine", "Tottenham", "West Footscray", 
        "Middle Footscray", "Footscray", "South Kensington", "North Melbourne", 
        "Southern Cross", "Flagstaff", "Melbourne Central", "Parliament", 
        "Flinders Street"
    ],
    "Cranbourne": [
        "Flinders Street", "Southern Cross", "Flagstaff", "Melbourne Central", 
        "Parliament", "Richmond", "South Yarra", "Hawksburn", "Toorak", 
        "Armadale", "Malvern", "Caulfield", "Carnegie", "Murrumbeena", 
        "Hughesdale", "Oakleigh", "Huntingdale", "Clayton", "Westall", 
        "Springvale", "Sandown Park", "Noble Park", "Yarraman", "Dandenong", 
        "Lynbrook", "Merinda Park", "Cranbourne"
    ],
    "Pakenham": [
        "Flinders Street", "Southern Cross", "Flagstaff", "Melbourne Central", 
        "Parliament", "Richmond", "South Yarra", "Hawksburn", "Toorak", 
        "Armadale", "Malvern", "Caulfield", "Carnegie", "Murrumbeena", 
        "Hughesdale", "Oakleigh", "Huntingdale", "Clayton", "Westall", 
        "Springvale", "Sandown Park", "Noble Park", "Yarraman", "Dandenong", 
        "Hallam", "Narre Warren", "Berwick", "Beaconsfield", "Officer", 
        "Cardinia Road", "Pakenham", "East Pakenham"
    ],
    "Stony Point": [
        "Frankston", "Leawarra", "Baxter", "Somerville", "Tyabb", "Hastings", 
        "Bittern", "Morradoo", "Crib Point", "Stony Point"
    ],
    "Flemington Racecourse": [
        "Flinders Street", "Southern Cross", "North Melbourne", "Showgrounds", 
        "Flemington Racecourse"
    ]
}

# 3. Create a case-insensitive reverse map (Station -> list of Train Lines)
station_to_lines = {}
for line, stations in lines_dict.items():
    for station in stations:
        norm_name = station.strip().lower()
        if norm_name not in station_to_lines:
            station_to_lines[norm_name] = []
        station_to_lines[norm_name].append(line)

# 4. Map helper function
def lookup_train_lines(station_name):
    if not isinstance(station_name, str):
        return "Unknown"
    norm_name = station_name.strip().lower()
    lines = station_to_lines.get(norm_name, [])
    if not lines:
        return "Unknown"
    return ", ".join(lines)

# 5. Apply the column transformation
df_19['train_line_name'] = df_19['station_name'].apply(lookup_train_lines)
df_24['train_line_name'] = df_24['station_name'].apply(lookup_train_lines)

# 6. Save the data back to your local files (overwrites original with new column added)
df_19.to_csv('station_volumes_2019_2020.csv', index=False)
df_24.to_csv('station_volumes_2024_2025.csv', index=False)

print("Processing complete! 'train_line_name' column successfully appended.")