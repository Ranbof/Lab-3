import pandas as pd

def download_data():
    df = pd.read_csv('https://raw.githubusercontent.com/Ranbof/Data-lab-3/refs/heads/main/taxi_dataset.csv')
    df.to_csv("taxi_dataset.csv", index=False)
    return df

def preprocess_taxi_data(df):
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])
    df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds()
    
    df = df[(df['trip_duration'] > 60) & (df['trip_duration'] < 3600*3)]
    
    nyc_min_long, nyc_max_long = -74.05, -73.75
    nyc_min_lat, nyc_max_lat = 40.60, 40.90
    df = df[
        (df['pickup_longitude'].between(nyc_min_long, nyc_max_long)) &
        (df['pickup_latitude'].between(nyc_min_lat, nyc_max_lat)) &
        (df['dropoff_longitude'].between(nyc_min_long, nyc_max_long)) &
        (df['dropoff_latitude'].between(nyc_min_lat, nyc_max_lat))
    ]
    
    def haversine_distance(lon1, lat1, lon2, lat2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a)) 
        km = 6371 * c
        return km
    
    df['trip_distance'] = df.apply(
        lambda x: haversine_distance(
            x['pickup_longitude'], x['pickup_latitude'],
            x['dropoff_longitude'], x['dropoff_latitude']
        ), axis=1
    )
    
    df = df[(df['trip_distance'] > 0.1) & (df['trip_distance'] < 30)]
    
    features = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude', 
                'dropoff_longitude', 'dropoff_latitude', 'trip_distance']
    target = 'trip_duration'
    
    df[features].to_csv('taxi_features.csv', index=False)
    df[target].to_csv('taxi_target.csv', index=False)
    
    return True

if __name__ == "__main__":
    df = download_data()
    preprocess_taxi_data(df)
