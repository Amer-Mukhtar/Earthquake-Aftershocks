"""
PART 1: DATA PROCESSING (OPTIMIZED VERSION)
========================
Earthquake Aftershock Prediction System
Predicts location (lat, lon) and magnitude of aftershocks from mainshock data
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (15, 8)


class AftershockDataProcessor:
    """
    Process earthquake data to identify mainshocks and their aftershocks
    Prepare data for predicting aftershock location and magnitude
    """
    
    def __init__(self, csv_path):
        """
        Initialize data processor
        
        Parameters:
        -----------
        csv_path : str
            Path to earthquake CSV file containing mainshocks and aftershocks
        """
        self.csv_path = csv_path
        self.df = None
        self.scaler_input = MinMaxScaler()
        self.scaler_output = MinMaxScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def load_data(self):
        """Load earthquake data from CSV"""
        print("="*70)
        print(" LOADING EARTHQUAKE DATA")
        print("="*70)
        
        self.df = pd.read_csv(self.csv_path)
        print(f" Loaded {len(self.df)} earthquake records from {self.csv_path}")
        
        # Display first few rows
        print("\n Sample data:")
        print(self.df.head())
        
        # Display basic info
        print(f"\n Dataset Info:")
        print(f"   Total records: {len(self.df)}")
        print(f"   Columns: {list(self.df.columns)}")
        
        return self.df
    
    def preprocess_data(self):
        """Clean and preprocess earthquake data"""
        print("\n" + "="*70)
        print("🔧 PREPROCESSING DATA")
        print("="*70)
        
        # Convert time to datetime
        print("\n1️ Converting time to datetime...")
        self.df['time'] = pd.to_datetime(self.df['time'])
        self.df = self.df.sort_values('time').reset_index(drop=True)
        print("    Time converted and sorted chronologically")
        
        # Extract and convert features
        print("\n2️ Extracting features...")
        self.df['magnitude'] = pd.to_numeric(self.df['mag'], errors='coerce')
        self.df['depth_km'] = pd.to_numeric(self.df['depth'], errors='coerce')
        self.df['latitude'] = pd.to_numeric(self.df['latitude'], errors='coerce')
        self.df['longitude'] = pd.to_numeric(self.df['longitude'], errors='coerce')
        print("    Features extracted: magnitude, depth, latitude, longitude")
        
        # Remove NaN values
        print("\n3️ Cleaning data...")
        original_count = len(self.df)
        self.df = self.df.dropna(subset=['magnitude', 'depth_km', 'latitude', 'longitude'])
        removed_count = original_count - len(self.df)
        print(f"    Removed {removed_count} records with missing values")
        print(f"    Clean dataset: {len(self.df)} records")
        
        # Display statistics
        print("\n Data Statistics:")
        print(f"   Date range: {self.df['time'].min()} to {self.df['time'].max()}")
        print(f"   Magnitude: {self.df['magnitude'].min():.2f} to {self.df['magnitude'].max():.2f} (avg: {self.df['magnitude'].mean():.2f})")
        print(f"   Depth: {self.df['depth_km'].min():.2f} to {self.df['depth_km'].max():.2f} km (avg: {self.df['depth_km'].mean():.2f})")
        print(f"   Latitude: {self.df['latitude'].min():.2f} to {self.df['latitude'].max():.2f}")
        print(f"   Longitude: {self.df['longitude'].min():.2f} to {self.df['longitude'].max():.2f}")
        
        return self.df
    
    def identify_mainshock_aftershock_pairs(
        self,
        time_window_days=30,
        distance_km=100,
        min_mainshock_mag=5.0,
        min_mag_gap=0.5,
    ):
        """
        Identify mainshock-aftershock pairs from the dataset (OPTIMIZED VERSION)
        
        Parameters:
        -----------
        time_window_days : int
            Time window after mainshock to look for aftershocks (days)
        distance_km : float
            Maximum distance from mainshock to consider as aftershock (km)
        min_mainshock_mag : float
            Minimum magnitude to consider as potential mainshock (reduces computation)
        min_mag_gap : float
            Minimum magnitude difference (mainshock - aftershock). Ensures
            aftershocks are meaningfully smaller than the mainshock.
        
        Returns:
        --------
        pairs : list of dicts
            List of mainshock-aftershock pairs
        """
        print("\n" + "="*70)
        print(" IDENTIFYING MAINSHOCK-AFTERSHOCK PAIRS")
        print("="*70)
        
        print(f"\n Criteria:")
        print(f"   • Time window: {time_window_days} days after mainshock")
        print(f"   • Distance threshold: {distance_km} km from mainshock")
        print(f"   • Minimum mainshock magnitude: {min_mainshock_mag}")
        print(f"   • Aftershock magnitude < Mainshock magnitude - {min_mag_gap}")
        
        pairs = []
        
        # Filter potential mainshocks (larger events only)
        mainshocks = self.df[self.df['magnitude'] >= min_mainshock_mag].copy()
        print(f"\n Found {len(mainshocks)} potential mainshocks (mag >= {min_mainshock_mag})")
        
        # Sort mainshocks by magnitude (largest first)
        mainshocks = mainshocks.sort_values('magnitude', ascending=False).reset_index(drop=True)
        
        # Convert time window to timedelta
        time_window = pd.Timedelta(days=time_window_days)
        
        print(f" Processing mainshocks...")
        
        # For each mainshock
        for i, mainshock in mainshocks.iterrows():
            if i % 10 == 0:
                print(f"   Processing mainshock {i+1}/{len(mainshocks)}...", end='\r')
            
            # Skip if this event is likely an aftershock of an even larger,
            # earlier nearby event (simple "local mainshock" filter).
            prev_time_mask = (
                (self.df['time'] >= mainshock['time'] - time_window) &
                (self.df['time'] < mainshock['time'])
            )
            prev_events = self.df[prev_time_mask]
            if len(prev_events) > 0:
                prev_distances = self.haversine_distance_vectorized(
                    mainshock['latitude'],
                    mainshock['longitude'],
                    prev_events['latitude'].values,
                    prev_events['longitude'].values
                )
                prev_is_near = prev_distances <= distance_km
                if np.any(prev_is_near & (prev_events['magnitude'].values >= mainshock['magnitude'])):
                    # There is a larger or equal event nearby and earlier in time;
                    # treat that earlier event as the mainshock instead.
                    continue
            
            # Filter potential aftershocks by time window first (HUGE optimization!)
            time_mask = (
                (self.df['time'] > mainshock['time']) & 
                (self.df['time'] <= mainshock['time'] + time_window)
            )
            
            # Filter by magnitude (and enforce minimum magnitude gap)
            mag_mask = self.df['magnitude'] < (mainshock['magnitude'] - min_mag_gap)
            
            # Combine filters
            potential_aftershocks = self.df[time_mask & mag_mask]
            
            if len(potential_aftershocks) == 0:
                continue
            
            # Calculate distances using vectorized operations
            distances = self.haversine_distance_vectorized(
                mainshock['latitude'], 
                mainshock['longitude'],
                potential_aftershocks['latitude'].values,
                potential_aftershocks['longitude'].values
            )
            
            # Filter by distance
            valid_indices = distances <= distance_km
            valid_aftershocks = potential_aftershocks[valid_indices]
            valid_distances = distances[valid_indices]
            
            # Create pairs
            for idx, (_, aftershock) in enumerate(valid_aftershocks.iterrows()):
                time_diff_hours = (aftershock['time'] - mainshock['time']).total_seconds() / 3600
                
                pairs.append({
                    'mainshock_mag': mainshock['magnitude'],
                    'mainshock_depth': mainshock['depth_km'],
                    'mainshock_lat': mainshock['latitude'],
                    'mainshock_lon': mainshock['longitude'],
                    'mainshock_time': mainshock['time'],
                    'aftershock_mag': aftershock['magnitude'],
                    'aftershock_depth': aftershock['depth_km'],
                    'aftershock_lat': aftershock['latitude'],
                    'aftershock_lon': aftershock['longitude'],
                    'aftershock_time': aftershock['time'],
                    'time_diff_hours': time_diff_hours,
                    'distance_km': valid_distances[idx]
                })
        
        print(f"\n Identified {len(pairs)} mainshock-aftershock pairs")
        
        if len(pairs) > 0:
            avg_time_diff = np.mean([p['time_diff_hours'] for p in pairs])
            avg_distance = np.mean([p['distance_km'] for p in pairs])
            mag_diffs = [p['mainshock_mag'] - p['aftershock_mag'] for p in pairs]
            print(f"   • Average time difference: {avg_time_diff:.1f} hours")
            print(f"   • Average distance: {avg_distance:.1f} km")
            print(f"   • Magnitude difference (main - after): "
                  f"min={np.min(mag_diffs):.2f}, max={np.max(mag_diffs):.2f}, "
                  f"avg={np.mean(mag_diffs):.2f}")
        
        return pairs
    
    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two points on Earth using Haversine formula
        
        Returns:
        --------
        distance : float
            Distance in kilometers
        """
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def haversine_distance_vectorized(self, lat1, lon1, lat2_array, lon2_array):
        """
        Vectorized Haversine distance calculation for multiple points
        
        Returns:
        --------
        distances : numpy array
            Distances in kilometers
        """
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = np.radians(lat1), np.radians(lon1)
        lat2_array = np.radians(lat2_array)
        lon2_array = np.radians(lon2_array)
        
        dlat = lat2_array - lat1
        dlon = lon2_array - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2_array) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def create_training_data(self, pairs):
        """
        Create training data from mainshock-aftershock pairs
        
        Input (X): Mainshock features (magnitude, depth, latitude, longitude)
        Output (y): Aftershock features (magnitude, latitude, longitude)
        """
        print("\n" + "="*70)
        print(" CREATING TRAINING DATA")
        print("="*70)
        
        # Input features: mainshock characteristics
        X = []
        for pair in pairs:
            X.append([
                pair['mainshock_mag'],
                pair['mainshock_depth'],
                pair['mainshock_lat'],
                pair['mainshock_lon']
            ])
        
        # Output features: aftershock characteristics (what we want to predict)
        y = []
        for pair in pairs:
            y.append([
                pair['aftershock_mag'],
                pair['aftershock_lat'],
                pair['aftershock_lon']
            ])
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n Created training data:")
        print(f"   • Input (X) shape: {X.shape} [samples, mainshock_features]")
        print(f"   • Output (y) shape: {y.shape} [samples, aftershock_features]")
        print(f"\n   Input features: [magnitude, depth, latitude, longitude] of mainshock")
        print(f"   Output features: [magnitude, latitude, longitude] of aftershock")
        
        return X, y
    
    def normalize_data(self, X, y):
        """Normalize input and output features separately"""
        print("\n Normalizing features...")
        
        X_normalized = self.scaler_input.fit_transform(X)
        y_normalized = self.scaler_output.fit_transform(y)
        
        print("    Input features normalized")
        print("    Output features normalized")
        
        return X_normalized, y_normalized
    
    def split_data(self, X, y, test_size=0.2):
        """Split data into training and testing sets"""
        print("\n" + "="*70)
        print(" SPLITTING DATA")
        print("="*70)
        
        # Random split (since mainshock-aftershock pairs are independent)
        from sklearn.model_selection import train_test_split
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        print(f"\n Split ratio: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")
        print(f"   Training samples: {len(self.X_train)}")
        print(f"   Testing samples: {len(self.X_test)}")
        print(f"\n   X_train shape: {self.X_train.shape}")
        print(f"   X_test shape: {self.X_test.shape}")
        print(f"   y_train shape: {self.y_train.shape}")
        print(f"   y_test shape: {self.y_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def save_processed_data(self, output_dir='processed_data'):
        """Save processed data and scalers for model training"""
        import os
        
        print("\n" + "="*70)
        print(" SAVING PROCESSED DATA")
        print("="*70)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save numpy arrays
        np.save(f'{output_dir}/X_train.npy', self.X_train)
        np.save(f'{output_dir}/X_test.npy', self.X_test)
        np.save(f'{output_dir}/y_train.npy', self.y_train)
        np.save(f'{output_dir}/y_test.npy', self.y_test)
        print(f"    Training and testing data saved")
        
        # Save scalers
        with open(f'{output_dir}/scaler_input.pkl', 'wb') as f:
            pickle.dump(self.scaler_input, f)
        with open(f'{output_dir}/scaler_output.pkl', 'wb') as f:
            pickle.dump(self.scaler_output, f)
        print(f"    Scalers saved")
        
        # Save metadata
        metadata = {
            'input_features': ['mainshock_magnitude', 'mainshock_depth', 'mainshock_latitude', 'mainshock_longitude'],
            'output_features': ['aftershock_magnitude', 'aftershock_latitude', 'aftershock_longitude'],
            'n_input_features': 4,
            'n_output_features': 3,
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'total_pairs': len(self.X_train) + len(self.X_test)
        }
        
        with open(f'{output_dir}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print(f"    Metadata saved")
        
        print(f"\n All files saved in '{output_dir}/' directory")
        print(f"   - X_train.npy, X_test.npy (mainshock data)")
        print(f"   - y_train.npy, y_test.npy (aftershock data)")
        print(f"   - scaler_input.pkl, scaler_output.pkl")
        print(f"   - metadata.pkl")
    
    def visualize_data(self, pairs):
        """Create visualizations of the processed data"""
        print("\n" + "="*70)
        print(" CREATING VISUALIZATIONS")
        print("="*70)
        
        if len(pairs) == 0:
            print("    No pairs to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(' Mainshock-Aftershock Analysis', fontsize=16, fontweight='bold')
        
        # 1. Magnitude relationship
        ax1 = axes[0, 0]
        mainshock_mags = [p['mainshock_mag'] for p in pairs]
        aftershock_mags = [p['aftershock_mag'] for p in pairs]
        ax1.scatter(mainshock_mags, aftershock_mags, alpha=0.6, s=50)
        ax1.plot([min(mainshock_mags), max(mainshock_mags)], 
                [min(mainshock_mags), max(mainshock_mags)], 
                'r--', linewidth=2, label='Equal Magnitude')
        ax1.set_xlabel('Mainshock Magnitude')
        ax1.set_ylabel('Aftershock Magnitude')
        ax1.set_title('Mainshock vs Aftershock Magnitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distance distribution
        ax2 = axes[0, 1]
        distances = [p['distance_km'] for p in pairs]
        ax2.hist(distances, bins=30, edgecolor='black', alpha=0.7, color='orange')
        ax2.set_xlabel('Distance from Mainshock (km)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Aftershock Distance Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Time distribution
        ax3 = axes[1, 0]
        time_diffs = [p['time_diff_hours'] for p in pairs]
        ax3.hist(time_diffs, bins=30, edgecolor='black', alpha=0.7, color='green')
        ax3.set_xlabel('Time After Mainshock (hours)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Aftershock Timing Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Geographic plot
        ax4 = axes[1, 1]
        mainshock_lats = [p['mainshock_lat'] for p in pairs[:100]]  # Limit for clarity
        mainshock_lons = [p['mainshock_lon'] for p in pairs[:100]]
        aftershock_lats = [p['aftershock_lat'] for p in pairs[:100]]
        aftershock_lons = [p['aftershock_lon'] for p in pairs[:100]]
        
        ax4.scatter(mainshock_lons, mainshock_lats, c='red', s=100, 
                   marker='*', label='Mainshocks', alpha=0.7, edgecolors='black')
        ax4.scatter(aftershock_lons, aftershock_lats, c='blue', s=30, 
                   label='Aftershocks', alpha=0.6)
        ax4.set_xlabel('Longitude')
        ax4.set_ylabel('Latitude')
        ax4.set_title('Geographic Distribution (First 100 pairs)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('aftershock_analysis.png', dpi=300, bbox_inches='tight')
        print("    Visualizations saved to 'aftershock_analysis.png'")
        plt.show()
    
    def run_pipeline(self):
        """Run complete data processing pipeline"""
        print("\n" + "="*70)
        print(" STARTING DATA PROCESSING PIPELINE")
        print("="*70)
        
        # Load data
        self.load_data()
        
        # Preprocess
        self.preprocess_data()
        
        # Identify mainshock-aftershock pairs
        pairs = self.identify_mainshock_aftershock_pairs()
        
        if len(pairs) == 0:
            print("\n ERROR: No mainshock-aftershock pairs found!")
            print("   Try adjusting the time_window_days or distance_km parameters")
            return
        
        # Create training data
        X, y = self.create_training_data(pairs)
        
        # Normalize
        X_norm, y_norm = self.normalize_data(X, y)
        
        # Split data
        self.split_data(X_norm, y_norm)
        
        # Visualize
        self.visualize_data(pairs)
        
        # Save processed data
        self.save_processed_data()
        
        print("\n" + "="*70)
        print(" DATA PROCESSING COMPLETED!")
        print("="*70)
        print("\n Next Step: Run 'model_training.py' to train the LSTM model")
        print(" The model will predict aftershock magnitude and location from mainshock data")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main function to run data processing"""
    
    print("\n" + "" * 35)
    print("EARTHQUAKE AFTERSHOCK PREDICTION ")
    print("PART 1: DATA PROCESSING (OPTIMIZED)")
    print("" * 35)
    
    print("\n This system will:")
    print("   1. Load your earthquake dataset (mainshocks + aftershocks)")
    print("   2. Identify mainshock-aftershock pairs")
    print("   3. Prepare data for BILSTM training")
    print("   4. The model will predict: aftershock magnitude, latitude, longitude")
    
    # Get CSV file path from user
    csv_path = input("\n Enter path to earthquake CSV file: ").strip()
    
    # Initialize processor
    processor = AftershockDataProcessor(csv_path=csv_path)
    
    # Run pipeline
    processor.run_pipeline()
    
    print("\n Ready for model training!")


if __name__ == "__main__":
    main()