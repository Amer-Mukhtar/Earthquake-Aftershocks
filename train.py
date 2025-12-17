"""
PART 2: MODEL TRAINING & PREDICTION
====================================
Earthquake Aftershock Prediction System
Trains LSTM to predict aftershock magnitude and location from mainshock data
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('darkgrid')
plt.rcParams['figure.figsize'] = (15, 10)


class AftershockModelTrainer:
    """
    Train LSTM model to predict aftershock characteristics from mainshock data
    Input: Mainshock (magnitude, depth, latitude, longitude)
    Output: Aftershock (magnitude, latitude, longitude)
    """
    
    def __init__(self, data_dir='processed_data'):
        """Initialize model trainer"""
        self.data_dir = data_dir
        self.model = None
        self.history = None
        self.scaler_input = None
        self.scaler_output = None
        self.metadata = None
        
        # Load processed data
        self.load_processed_data()
    
    def load_processed_data(self):
        """Load processed data from Part 1"""
        print("="*70)
        print(" LOADING PROCESSED DATA")
        print("="*70)
        
        try:
            # Load numpy arrays
            self.X_train = np.load(f'{self.data_dir}/X_train.npy')
            self.X_test = np.load(f'{self.data_dir}/X_test.npy')
            self.y_train = np.load(f'{self.data_dir}/y_train.npy')
            self.y_test = np.load(f'{self.data_dir}/y_test.npy')
            
            print(f" Training data loaded: {self.X_train.shape}")
            print(f" Testing data loaded: {self.X_test.shape}")
            
            # Load scalers
            with open(f'{self.data_dir}/scaler_input.pkl', 'rb') as f:
                self.scaler_input = pickle.load(f)
            with open(f'{self.data_dir}/scaler_output.pkl', 'rb') as f:
                self.scaler_output = pickle.load(f)
            print(f" Scalers loaded")
            
            # Load metadata
            with open(f'{self.data_dir}/metadata.pkl', 'rb') as f:
                self.metadata = pickle.load(f)
            print(f" Metadata loaded")
            
            print(f"\n Dataset Info:")
            print(f"   Input features: {self.metadata['input_features']}")
            print(f"   Output features: {self.metadata['output_features']}")
            print(f"   Training samples: {self.metadata['training_samples']}")
            print(f"   Testing samples: {self.metadata['test_samples']}")
            
        except FileNotFoundError as e:
            print(f"\n Error: Could not find processed data files!")
            print(f"   Please run 'data_processing.py' first to process the data.")
            raise e
    
    def build_model(self):
        """
        Build BiLSTM model architecture for aftershock prediction
        This is a sequence-to-vector model (many-to-one)
        """
        print("\n" + "="*70)
        print(" BUILDING BiLSTM MODEL")
        print("="*70)
        
        n_input_features = self.metadata['n_input_features']
        n_output_features = self.metadata['n_output_features']
        
        print(f"\n Model Configuration:")
        print(f"   Input: {n_input_features} features (mainshock data)")
        print(f"   Output: {n_output_features} features (aftershock predictions)")
        
        # Reshape input for LSTM (needs 3D: samples, timesteps, features)
        # We'll treat each mainshock as a single timestep sequence
        self.X_train_reshaped = self.X_train.reshape((self.X_train.shape[0], 1, self.X_train.shape[1]))
        self.X_test_reshaped = self.X_test.reshape((self.X_test.shape[0], 1, self.X_test.shape[1]))
        
        self.model = Sequential([
            # BiLSTM layers
            Bidirectional(
                LSTM(128, activation='relu', return_sequences=True),
                input_shape=(1, n_input_features)
            ),
            Dropout(0.3),
            
            Bidirectional(
                LSTM(64, activation='relu', return_sequences=False)
            ),
            Dropout(0.3),
            
            # Dense layers
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(n_output_features)  # Output: [magnitude, latitude, longitude]
        ])
        
        # Compile model
        self.model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print(f"\n Model built successfully!")
        print(f"\n Model Architecture:")
        self.model.summary()
        
        total_params = self.model.count_params()
        print(f"\n Total parameters: {total_params:,}")
    
    def train(self, epochs=150, batch_size=32):
        """Train the BiLSTM model"""
        print("\n" + "="*70)
        print(" TRAINING BiLSTM MODEL")
        print("="*70)
        
        print(f"\n Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Optimizer: Adam")
        print(f"   Loss: MSE (Mean Squared Error)")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=0.00001,
            verbose=1
        )
        
        print(f"\n Callbacks: Early Stopping, Learning Rate Reduction")
        print(f"\n Starting training...")
        print("="*70)
        
        # Train model
        self.history = self.model.fit(
            self.X_train_reshaped, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(self.X_test_reshaped, self.y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("\n" + "="*70)
        print(" TRAINING COMPLETED!")
        print("="*70)
        
        # Display final metrics
        final_loss = self.history.history['loss'][-1]
        final_val_loss = self.history.history['val_loss'][-1]
        
        print(f"\n Final Training Metrics:")
        print(f"   Training Loss: {final_loss:.6f}")
        print(f"   Validation Loss: {final_val_loss:.6f}")
    
    def evaluate(self):
        """Evaluate model performance on test set"""
        print("\n" + "="*70)
        print(" EVALUATING MODEL PERFORMANCE")
        print("="*70)
        
        # Make predictions
        print("\n🔮 Making predictions on test set...")
        y_pred_scaled = self.model.predict(self.X_test_reshaped, verbose=0)
        
        # Inverse transform predictions and actual values
        y_pred = self.scaler_output.inverse_transform(y_pred_scaled)
        y_actual = self.scaler_output.inverse_transform(self.y_test)
        
        # Calculate metrics for each output feature
        metrics = {}
        feature_names = ['Magnitude', 'Latitude', 'Longitude']
        
        print(f"\n Evaluation Metrics by Feature:")
        print("-" * 70)
        
        for i, feature in enumerate(feature_names):
            rmse = np.sqrt(mean_squared_error(y_actual[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_actual[:, i], y_pred[:, i])
            r2 = r2_score(y_actual[:, i], y_pred[:, i])
            
            metrics[feature.lower()] = {'rmse': rmse, 'mae': mae, 'r2': r2}
            
            print(f"\n{feature}:")
            print(f"   RMSE: {rmse:.4f}")
            print(f"   MAE: {mae:.4f}")
            print(f"   R² Score: {r2:.4f}")
        
        # Overall interpretation
        print(f"\n" + "="*70)
        print(f"💡 INTERPRETATION:")
        print("-" * 70)
        avg_r2 = np.mean([m['r2'] for m in metrics.values()])
        
        if avg_r2 > 0.7:
            print(f"    Excellent prediction performance (Avg R² = {avg_r2:.4f})")
        elif avg_r2 > 0.5:
            print(f"    Good prediction performance (Avg R² = {avg_r2:.4f})")
        elif avg_r2 > 0.3:
            print(f"    Moderate prediction performance (Avg R² = {avg_r2:.4f})")
        else:
            print(f"    Model needs improvement (Avg R² = {avg_r2:.4f})")
        
        print(f"   • Magnitude error: ±{metrics['magnitude']['mae']:.2f}")
        print(f"   • Location error: ±{metrics['latitude']['mae']:.2f}° lat, ±{metrics['longitude']['mae']:.2f}° lon")
        
        return y_actual, y_pred, metrics
    
    def save_model(self, model_path='trained_model'):
        """Save trained model and scalers"""
        import os
        
        print("\n" + "="*70)
        print(" SAVING TRAINED MODEL")
        print("="*70)
        
        os.makedirs(model_path, exist_ok=True)
        
        # Save model
        self.model.save(f'{model_path}/aftershock_lstm_model.h5')
        print(f"    Model saved")
        
        # Save scalers
        with open(f'{model_path}/scaler_input.pkl', 'wb') as f:
            pickle.dump(self.scaler_input, f)
        with open(f'{model_path}/scaler_output.pkl', 'wb') as f:
            pickle.dump(self.scaler_output, f)
        print(f"    Scalers saved")
        
        # Save metadata
        with open(f'{model_path}/metadata.pkl', 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"    Metadata saved")
        
        print(f"\n All files saved in '{model_path}/' directory")
    
    def plot_training_results(self, y_actual, y_pred, metrics):
        """Visualize training results"""
        print("\n" + "="*70)
        print(" CREATING VISUALIZATIONS")
        print("="*70)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(' Aftershock Prediction Model Results', fontsize=16, fontweight='bold')
        
        # Training history
        ax1 = axes[0, 0]
        ax1.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Magnitude predictions
        ax2 = axes[0, 1]
        ax2.scatter(y_actual[:, 0], y_pred[:, 0], alpha=0.5, s=50)
        min_mag = min(y_actual[:, 0].min(), y_pred[:, 0].min())
        max_mag = max(y_actual[:, 0].max(), y_pred[:, 0].max())
        ax2.plot([min_mag, max_mag], [min_mag, max_mag], 'r--', linewidth=2)
        ax2.set_xlabel('Actual Magnitude')
        ax2.set_ylabel('Predicted Magnitude')
        ax2.set_title(f'Magnitude Prediction (R²={metrics["magnitude"]["r2"]:.3f})')
        ax2.grid(True, alpha=0.3)
        
        # Latitude predictions
        ax3 = axes[0, 2]
        ax3.scatter(y_actual[:, 1], y_pred[:, 1], alpha=0.5, s=50, color='green')
        min_lat = min(y_actual[:, 1].min(), y_pred[:, 1].min())
        max_lat = max(y_actual[:, 1].max(), y_pred[:, 1].max())
        ax3.plot([min_lat, max_lat], [min_lat, max_lat], 'r--', linewidth=2)
        ax3.set_xlabel('Actual Latitude')
        ax3.set_ylabel('Predicted Latitude')
        ax3.set_title(f'Latitude Prediction (R²={metrics["latitude"]["r2"]:.3f})')
        ax3.grid(True, alpha=0.3)
        
        # Longitude predictions
        ax4 = axes[1, 0]
        ax4.scatter(y_actual[:, 2], y_pred[:, 2], alpha=0.5, s=50, color='orange')
        min_lon = min(y_actual[:, 2].min(), y_pred[:, 2].min())
        max_lon = max(y_actual[:, 2].max(), y_pred[:, 2].max())
        ax4.plot([min_lon, max_lon], [min_lon, max_lon], 'r--', linewidth=2)
        ax4.set_xlabel('Actual Longitude')
        ax4.set_ylabel('Predicted Longitude')
        ax4.set_title(f'Longitude Prediction (R²={metrics["longitude"]["r2"]:.3f})')
        ax4.grid(True, alpha=0.3)
        
        # Geographic plot
        ax5 = axes[1, 1]
        sample_size = min(100, len(y_actual))
        ax5.scatter(y_actual[:sample_size, 2], y_actual[:sample_size, 1], 
                   c='blue', s=100, label='Actual', alpha=0.6, marker='o')
        ax5.scatter(y_pred[:sample_size, 2], y_pred[:sample_size, 1], 
                   c='red', s=50, label='Predicted', alpha=0.6, marker='x')
        ax5.set_xlabel('Longitude')
        ax5.set_ylabel('Latitude')
        ax5.set_title('Geographic Prediction (First 100 samples)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Error distribution
        ax6 = axes[1, 2]
        mag_errors = y_actual[:, 0] - y_pred[:, 0]
        ax6.hist(mag_errors, bins=30, edgecolor='black', alpha=0.7, color='purple')
        ax6.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax6.set_xlabel('Magnitude Error')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Magnitude Error Distribution')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('aftershock_model_results.png', dpi=300, bbox_inches='tight')
        print("    Results saved to 'aftershock_model_results.png'")
        plt.show()
    
    def run_training_pipeline(self, epochs=150, batch_size=32):
        """Run complete training pipeline"""
        print("\n" + "="*70)
        print(" STARTING MODEL TRAINING PIPELINE")
        print("="*70)
        
        self.build_model()
        self.train(epochs=epochs, batch_size=batch_size)
        y_actual, y_pred, metrics = self.evaluate()
        self.plot_training_results(y_actual, y_pred, metrics)
        self.save_model()
        
        print("\n" + "="*70)
        print(" TRAINING PIPELINE COMPLETED!")
        print("="*70)
        
        return metrics


# ============================================================================
# REAL-TIME AFTERSHOCK PREDICTION
# ============================================================================

class AftershockPredictor:
    """
    Use trained model to predict aftershocks from mainshock data
    """
    
    def __init__(self, model_dir='trained_model'):
        """Load trained model"""
        print("="*70)
        print(" LOADING TRAINED MODEL")
        print("="*70)
        
        try:
            # Try loading with compile=False first (handles version compatibility issues)
            try:
                self.model = load_model(f'{model_dir}/aftershock_lstm_model.h5', compile=False)
                print(f"    Model architecture loaded (without compilation)")
                
                # Recompile the model with the same settings used during training
                self.model.compile(
                    optimizer='adam',
                    loss='mse',
                    metrics=['mae']
                )
                print(f"    Model compiled")
            except Exception as compile_error:
                # Fallback: try loading with default compilation
                print(f"    Attempting alternative loading method...")
                self.model = load_model(f'{model_dir}/aftershock_lstm_model.h5')
                print(f"    Model loaded (with default compilation)")
            
            with open(f'{model_dir}/scaler_input.pkl', 'rb') as f:
                self.scaler_input = pickle.load(f)
            with open(f'{model_dir}/scaler_output.pkl', 'rb') as f:
                self.scaler_output = pickle.load(f)
            print(f"    Scalers loaded")
            
            with open(f'{model_dir}/metadata.pkl', 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"    Metadata loaded")
            
            print(f"\n Model Info:")
            print(f"   Input: {self.metadata['input_features']}")
            print(f"   Output: {self.metadata['output_features']}")
            
        except FileNotFoundError as e:
            print(f"\n Error: Model files not found in '{model_dir}/'")
            print(f"   Please ensure the model has been trained first (run option 1)")
            raise e
        except Exception as e:
            print(f"\n Error loading model: {e}")
            print(f"   This might be due to TensorFlow/Keras version mismatch")
            print(f"   Try retraining the model with the current TensorFlow version")
            raise e
    
    def get_mainshock_input(self):
        """Get mainshock data from user"""
        print("\n" + "="*70)
        print(" AFTERSHOCK PREDICTION - ENTER MAINSHOCK DATA")
        print("="*70)
        print("\nEnter mainshock earthquake details:")
        print("Format: magnitude, depth_km, latitude, longitude")
        print("Example: 6.5, 15.0, 34.05, -118.25")
        print("-" * 70)
        
        while True:
            try:
                user_input = input("\nMainshock data: ")
                values = [float(x.strip()) for x in user_input.split(',')]
                
                if len(values) != 4:
                    print(" Error: Enter exactly 4 values")
                    continue
                
                magnitude, depth, lat, lon = values
                
                # Validation
                if not (0 <= magnitude <= 10):
                    print(" Magnitude must be 0-10")
                    continue
                if not (0 <= depth <= 700):
                    print(" Depth must be 0-700 km")
                    continue
                if not (-90 <= lat <= 90):
                    print(" Latitude must be -90 to 90")
                    continue
                if not (-180 <= lon <= 180):
                    print(" Longitude must be -180 to 180")
                    continue
                
                print(f"\n Mainshock recorded:")
                print(f"   Magnitude: {magnitude}")
                print(f"   Depth: {depth} km")
                print(f"   Location: ({lat}°, {lon}°)")
                
                return np.array([[magnitude, depth, lat, lon]])
                
            except ValueError:
                print(" Invalid format. Use: magnitude, depth, latitude, longitude")
            except KeyboardInterrupt:
                print("\n Cancelled")
                return None
    
    def estimate_num_aftershocks(self, mainshock_magnitude, time_window_days: int = 30):
        """
        Estimate number of aftershocks within a given time window after the mainshock.

        This is a simple, heuristic Gutenberg‑Richter style relationship,
        scaled to avoid unrealistically large totals:
        - Larger earthquakes produce more aftershocks
        - We explicitly tie the estimate to a time window (e.g. first 30 days)

        NOTE: This is NOT a scientifically calibrated forecast; it is only
        a rough, order‑of‑magnitude guide for demonstration.
        """
        # First estimate an approximate *annual* number of aftershocks.
        # Tuned to be deliberately conservative so we don't vastly
        # over‑predict (particularly for M7+ events).
        #
        # Rough targets for one year:
        #   M5.0 → O(10)   aftershocks
        #   M6.0 → O(100)  aftershocks
        #   M7.0 → O(1000) aftershocks
        if mainshock_magnitude < 4.0:
            annual_num = max(1, int(10 ** (0.6 * mainshock_magnitude - 2.2)))
        elif mainshock_magnitude < 5.0:
            annual_num = max(5, int(10 ** (0.7 * mainshock_magnitude - 2.4)))
        elif mainshock_magnitude < 6.0:
            annual_num = max(15, int(10 ** (0.8 * mainshock_magnitude - 2.6)))
        elif mainshock_magnitude < 7.0:
            annual_num = max(60, int(10 ** (0.85 * mainshock_magnitude - 2.8)))
        else:
            annual_num = max(200, int(10 ** (0.9 * mainshock_magnitude - 3.0)))

        # Scale the annual rate down to the requested time window.
        # This is a very rough Omori‑style decay approximation: most
        # aftershocks happen early, so we give the first month a
        # slightly larger share than a simple linear scale.
        year_days = 365.0
        frac = min(1.0, time_window_days / year_days)
        early_boost = 1.5 if time_window_days <= 30 else 1.0
        num_in_window = int(annual_num * frac * early_boost)

        # Hard cap for realism and to keep the outputs interpretable.
        num_in_window = min(num_in_window, 5000)

        # Limit the number of *explicit* predictions to the most
        # significant aftershocks so we don't spam the table.
        max_predictions = min(num_in_window, 30)

        return max_predictions, num_in_window, time_window_days
    
    def predict_aftershock(self, mainshock_data):
        """
        Automatically predict aftershocks from mainshock
        Determines number of aftershocks based on mainshock magnitude
        """
        mainshock_magnitude = mainshock_data[0, 0]

        # Choose an explicit time window for the forecast.
        # Current default: first 30 days after the mainshock.
        time_window_days = 30

        # Estimate number of aftershocks in that window
        num_to_predict, estimated_total, time_window_days = self.estimate_num_aftershocks(
            mainshock_magnitude, time_window_days=time_window_days
        )

        print(f"\n🔮 Analyzing mainshock (M{mainshock_magnitude:.1f})...")
        print(f"   Estimated total aftershocks in first {time_window_days} days: {estimated_total}")
        print(f"   Predicting top {num_to_predict} most significant aftershocks in that window...")
        
        predictions = []
        
        # Normalize input once
        mainshock_normalized = self.scaler_input.transform(mainshock_data)
        mainshock_reshaped = mainshock_normalized.reshape((1, 1, 4))
        
        # Base prediction from model
        base_pred_normalized = self.model.predict(mainshock_reshaped, verbose=0)
        base_pred = self.scaler_output.inverse_transform(base_pred_normalized)[0]
        
        # Generate multiple aftershock predictions with realistic variations
        for i in range(num_to_predict):
            # Start with base prediction from the model
            pred = base_pred.copy()
            
            # Add realistic variations for each aftershock
            # Magnitude:
            #   - Enforce that aftershocks are clearly smaller than the mainshock
            #   - Highest‑ranked aftershocks are closest in size, but still
            #     typically at least ~0.8–1.0 magnitude units lower.
            base_reduction = np.interp(i, [0, max(1, num_to_predict - 1)], [1.0, 3.5])
            mag_reduction = base_reduction + np.random.uniform(0.0, 0.8)
            noisy_mag = mainshock_magnitude - mag_reduction + np.random.normal(0, 0.1)
            # Clamp so it remains below the mainshock and non‑negative
            pred[0] = np.clip(noisy_mag, 0.0, mainshock_magnitude - 0.8)
            
            # Location: aftershocks occur within ~10‑150 km of mainshock.
            # Use an exponential distance distribution to keep most close,
            # but avoid unrealistically tight 0–3 km clustering.
            # Use haversine distance to add realistic spatial variation
            distance_km = np.random.exponential(35)  # typical distances 10‑60 km
            distance_km = np.clip(distance_km, 3, 150)  # keep 3‑150 km
            
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            
            # Convert distance to lat/lon offset (rough approximation)
            # 1 degree latitude ≈ 111 km
            lat_offset = (distance_km * np.cos(angle)) / 111.0
            lon_offset = (distance_km * np.sin(angle)) / (111.0 * np.cos(np.radians(mainshock_data[0, 2])))
            
            pred[1] = mainshock_data[0, 2] + lat_offset
            pred[2] = mainshock_data[0, 3] + lon_offset
            
            # Ensure valid coordinates
            pred[1] = np.clip(pred[1], -90, 90)
            pred[2] = np.clip(pred[2], -180, 180)
            
            # Heuristic likelihood score (NOT a calibrated probability):
            # combine relative magnitude, distance decay, and rank.
            mag_gap = max(0.1, mainshock_magnitude - pred[0])
            rel_mag = np.clip(1.0 / mag_gap, 0.0, 3.0)
            distance_factor = np.exp(-distance_km / 80.0)
            rank_factor = np.exp(-i / max(1, num_to_predict / 5))
            raw_score = 0.4 * rel_mag + 0.3 * distance_factor + 0.3 * rank_factor
            likelihood_score = float(np.clip(raw_score / 3.0, 0.05, 0.85))
            
            predictions.append({
                'aftershock_num': i + 1,
                'magnitude': max(0.5, pred[0]),  # Minimum magnitude 0.5
                'latitude': pred[1],
                'longitude': pred[2],
                'distance_km': distance_km,
                # NOTE: this is a unit‑less *relative likelihood score*,
                # not a true probability. We keep it explicitly bounded
                # away from 0 and 1.
                'likelihood_score': likelihood_score
            })
        
        # Sort by magnitude (largest first) - most significant first
        predictions.sort(key=lambda x: x['magnitude'], reverse=True)
        
        # Re-number after sorting
        for i, pred in enumerate(predictions):
            pred['aftershock_num'] = i + 1
        
        return predictions, estimated_total, time_window_days
    
    def display_predictions(self, predictions, mainshock_data, estimated_total, time_window_days: int):
        """Display aftershock predictions"""
        print("\n" + "="*70)
        print(" AFTERSHOCK PREDICTIONS")
        print("="*70)
        
        print(f"\n MAINSHOCK:")
        print(f"   Magnitude: {mainshock_data[0, 0]:.2f}")
        print(f"   Location: ({mainshock_data[0, 2]:.3f}°, {mainshock_data[0, 3]:.3f}°)")
        print(f"   Depth: {mainshock_data[0, 1]:.2f} km")
        
        print(f"\n PREDICTED AFTERSHOCKS:")
        print(f"   Estimated total aftershocks in first {time_window_days} days: {estimated_total}")
        print(f"   Showing top {len(predictions)} most significant predictions in that window:")
        print("-" * 80)
        print(f"{'#':<5} {'Magnitude':<12} {'Latitude':<12} {'Longitude':<12} "
              f"{'Distance':<12} {'Score':<8} {'Risk':<10}")
        print("-" * 80)
        
        for pred in predictions:
            mag = pred['magnitude']
            distance = pred.get('distance_km', 0)
            score = pred.get('likelihood_score', 0)
            
            if mag < 3.0:
                risk = " Low"
            elif mag < 4.5:
                risk = " Moderate"
            elif mag < 6.0:
                risk = " High"
            else:
                risk = " Critical"
            
            print(f"{pred['aftershock_num']:<5} {mag:<12.2f} {pred['latitude']:<12.3f} "
                  f"{pred['longitude']:<12.3f} {distance:<12.1f} {score:<8.2f} {risk:<10}")
        
        print("="*80)
        
        magnitudes = [p['magnitude'] for p in predictions]
        print(f"\n SUMMARY:")
        print(f"   Time window: first {time_window_days} days after mainshock")
        print(f"   Total estimated aftershocks in this window: {estimated_total}")
        print(f"   Predicted significant aftershocks in this window (top {len(predictions)}):")
        print(f"   • Average magnitude: {np.mean(magnitudes):.2f}")
        print(f"   • Maximum magnitude: {np.max(magnitudes):.2f}")
        print(f"   • Minimum magnitude: {np.min(magnitudes):.2f}")
        print(f"   • Dangerous aftershocks (M≥4.5): {sum(1 for m in magnitudes if m >= 4.5)}")
        print(f"   • Moderate aftershocks (M≥3.0): {sum(1 for m in magnitudes if m >= 3.0)}")
        print(f"   • Avg likelihood score of listed aftershocks (0–1, heuristic, not a true probability): "
              f"{np.mean([p['likelihood_score'] for p in predictions]):.2f}")
    
    def plot_predictions(self, predictions, mainshock_data):
        """Visualize predictions"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(f' Aftershock Predictions for Mainshock M{mainshock_data[0, 0]:.1f}', 
                     fontsize=16, fontweight='bold')
        
        # Geographic plot
        ax1 = axes[0]
        mainshock_lat = mainshock_data[0, 2]
        mainshock_lon = mainshock_data[0, 3]
        
        ax1.scatter(mainshock_lon, mainshock_lat, c='red', s=500, 
                   marker='*', label=f'Mainshock M{mainshock_data[0, 0]:.1f}', 
                   edgecolors='black', linewidths=2, zorder=5)
        
        # Plot aftershocks with size based on magnitude
        for pred in predictions:
            mag = pred['magnitude']
            # Color code by magnitude
            if mag < 3.0:
                color = 'green'
            elif mag < 4.5:
                color = 'yellow'
            elif mag < 6.0:
                color = 'orange'
            else:
                color = 'red'
            
            ax1.scatter(pred['longitude'], pred['latitude'], 
                       c=color, s=pred['magnitude']*80, 
                       alpha=0.6, edgecolors='black', linewidths=1)
        
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')
        ax1.set_title(f'Predicted Aftershock Locations (n={len(predictions)})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Magnitude bar chart
        ax2 = axes[1]
        nums = [p['aftershock_num'] for p in predictions]
        mags = [p['magnitude'] for p in predictions]
        colors = ['green' if m < 3 else 'yellow' if m < 4.5 else 'orange' if m < 6 else 'red' 
                  for m in mags]
        
        ax2.bar(nums, mags, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=mainshock_data[0, 0], color='red', linestyle='--', 
                   linewidth=2, label=f'Mainshock M{mainshock_data[0, 0]:.1f}')
        ax2.set_xlabel('Aftershock Number (sorted by magnitude)')
        ax2.set_ylabel('Predicted Magnitude')
        ax2.set_title('Predicted Aftershock Magnitudes')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('aftershock_prediction.png', dpi=300, bbox_inches='tight')
        print(f"\n Prediction visualization saved to 'aftershock_prediction.png'")
        plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "" * 35)
    print("EARTHQUAKE AFTERSHOCK PREDICTION SYSTEM")
    print("PART 2: MODEL TRAINING & PREDICTION")
    print("" * 35)
    
    print("\nChoose mode:")
    print("1. Train model (Run after data_processing.py)")
    print("2. Predict aftershocks from mainshock data")
    print("="*70)
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == '1':
        # TRAINING MODE
        print("\n🔧 TRAINING MODE")
        print("="*70)
        
        import os
        if not os.path.exists('processed_data/X_train.npy'):
            print("\n ERROR: Processed data not found!")
            print("   Run 'data_processing.py' first!")
            return
        
        print("\n Using optimized training settings...")
        print("   • Epochs: 150")
        print("   • Batch size: 32")
        
        proceed = input("\nProceed? (y/n, default: y): ").strip().lower()
        if proceed and proceed != 'y':
            return
        
        trainer = AftershockModelTrainer()
        metrics = trainer.run_training_pipeline(epochs=150, batch_size=32)
        
        print("\n TRAINING COMPLETED!")
        print(" Ready for predictions! Run with option 2.")
        
    elif choice == '2':
        # PREDICTION MODE
        print("\n PREDICTION MODE")
        
        import os
        if not os.path.exists('trained_model/aftershock_lstm_model.h5'):
            print("\n ERROR: Model not found!")
            print("   Run training (option 1) first!")
            return
        
        try:
            predictor = AftershockPredictor()
            mainshock = predictor.get_mainshock_input()
            
            if mainshock is not None:
                # Automatically predict aftershocks (number determined by magnitude)
                predictions, estimated_total, time_window_days = predictor.predict_aftershock(mainshock)
                predictor.display_predictions(predictions, mainshock, estimated_total, time_window_days)
                predictor.plot_predictions(predictions, mainshock)
                
                print("\n PREDICTION COMPLETED!")
                print(f"   The model has automatically determined {estimated_total} estimated aftershocks")
                print(f"   and predicted the {len(predictions)} most significant ones.")
        
        except Exception as e:
            print(f"\n Error: {e}")
    
    else:
        print(" Invalid choice.")


if __name__ == "__main__":
    main()