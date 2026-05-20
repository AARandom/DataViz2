import pandas as pd
import os

def extract_train_shapes(gtfs_dir, output_csv):
    print("Loading GTFS data...")
    
    # Read directly from extracted GTFS files
    routes_path = os.path.join(gtfs_dir, 'routes.txt')
    trips_path = os.path.join(gtfs_dir, 'trips.txt')
    shapes_path = os.path.join(gtfs_dir, 'shapes.txt')
    
    routes = pd.read_csv(routes_path)
    trips = pd.read_csv(trips_path)
    shapes = pd.read_csv(shapes_path)

    print("1. Filtering out Replacement Buses...")
    train_routes = routes[~routes['route_short_name'].str.contains('Replacement Bus', na=False)]
    train_routes = train_routes[~train_routes['route_id'].str.contains('-R:', na=False)]

    print("2. Mapping trips to train routes...")
    valid_trips = trips[trips['route_id'].isin(train_routes['route_id'])]
    trip_shapes = valid_trips[['route_id', 'shape_id']].drop_duplicates()

    print("3. Finding the longest continuous track per train line...")
    shape_lengths = shapes.groupby('shape_id').size().reset_index(name='point_count')
    merged_shapes = trip_shapes.merge(shape_lengths, on='shape_id')
    
    # Isolate the exact shape_id that has the maximum points for each route_id
    longest_shape_idx = merged_shapes.groupby('route_id')['point_count'].idxmax()
    longest_shapes = merged_shapes.loc[longest_shape_idx]

    print("4. Extracting GPS coordinates...")
    clean_shapes = shapes[shapes['shape_id'].isin(longest_shapes['shape_id'])]

    # Merge the human-readable names back
    final_data = clean_shapes.merge(longest_shapes[['shape_id', 'route_id']], on='shape_id')
    final_data = final_data.merge(train_routes[['route_id', 'route_short_name']], on='route_id')

    # Rename columns so Vega-Lite instantly understands them
    final_data = final_data.rename(columns={
        'route_id': 'train_line_id',
        'route_short_name': 'train_line_name',
        'shape_pt_lat': 'latitude',
        'shape_pt_lon': 'longitude',
        'shape_pt_sequence': 'sequence_order'
    })

    # Clean up the ID (Turns "aus:vic:vic-02-ALM:" into just "ALM")
    final_data['train_line_id'] = final_data['train_line_id'].str.replace('aus:vic:vic-02-', '').str.replace(':', '')

    # Sort strictly by line and sequence
    final_data = final_data.sort_values(['train_line_id', 'sequence_order'])

    # THE FIX: 'shape_id' is now included in the final export!
    final_data[['train_line_id', 'train_line_name', 'shape_id', 'latitude', 'longitude', 'sequence_order']].to_csv(output_csv, index=False)
    print(f"Success! Clean train paths saved to {output_csv}")

# Run the function
extract_train_shapes('data/metropolitan_train_gtfs_schedule', 'data/clean_train_shapes.csv')