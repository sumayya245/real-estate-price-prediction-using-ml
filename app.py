from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import numpy as np
import requests
import io
import json
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'real_estate_ml_predictor_2024'

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

class RealEstateMLPredictor:
    def __init__(self):
        self.df = None
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_metrics = {}
        self.feature_columns = []
        self.target_column = 'price_lakhs'
        self.models_trained = False
        self.model_save_path = 'models/'
        
    def save_models(self):
        """Save trained models and preprocessors"""
        try:
            # Save models
            with open(f'{self.model_save_path}models.pkl', 'wb') as f:
                pickle.dump(self.models, f)
            
            # Save scalers
            with open(f'{self.model_save_path}scalers.pkl', 'wb') as f:
                pickle.dump(self.scalers, f)
            
            # Save encoders
            with open(f'{self.model_save_path}encoders.pkl', 'wb') as f:
                pickle.dump(self.encoders, f)
            
            # Save metrics
            with open(f'{self.model_save_path}metrics.pkl', 'wb') as f:
                pickle.dump(self.model_metrics, f)
            
            # Save feature columns
            with open(f'{self.model_save_path}features.pkl', 'wb') as f:
                pickle.dump(self.feature_columns, f)
            
            # Save training timestamp
            with open(f'{self.model_save_path}timestamp.txt', 'w') as f:
                f.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            
            print("Models saved successfully!")
            return True
        except Exception as e:
            print(f"Error saving models: {e}")
            return False
    
    def load_models(self):
        """Load pre-trained models and preprocessors"""
        try:
            # Check if all required files exist
            required_files = ['models.pkl', 'scalers.pkl', 'encoders.pkl', 'metrics.pkl', 'features.pkl']
            for file in required_files:
                if not os.path.exists(f'{self.model_save_path}{file}'):
                    print(f"Model file {file} not found. Need to train models.")
                    return False
            
            # Load models
            with open(f'{self.model_save_path}models.pkl', 'rb') as f:
                self.models = pickle.load(f)
            
            # Load scalers
            with open(f'{self.model_save_path}scalers.pkl', 'rb') as f:
                self.scalers = pickle.load(f)
            
            # Load encoders
            with open(f'{self.model_save_path}encoders.pkl', 'rb') as f:
                self.encoders = pickle.load(f)
            
            # Load metrics
            with open(f'{self.model_save_path}metrics.pkl', 'rb') as f:
                self.model_metrics = pickle.load(f)
            
            # Load feature columns
            with open(f'{self.model_save_path}features.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            # Load dataset for unique values
            self.load_data()
            
            self.models_trained = True
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_training_timestamp(self):
        """Get the timestamp of when models were last trained"""
        try:
            with open(f'{self.model_save_path}timestamp.txt', 'r') as f:
                return f.read().strip()
        except:
            return "Unknown"
    
    def load_data(self):
        """Load and preprocess the dataset"""
        try:
            print("Loading dataset...")
            url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/telangana_realestate-lUjsfo48rCDqURGbZzo64MRfeS5J2Z.csv"
            
            # Try to load from cache first
            cache_file = 'data/dataset_cache.csv'
            if os.path.exists(cache_file):
                print("Loading from cache...")
                self.df = pd.read_csv(cache_file)
            else:
                print("Downloading dataset...")
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                self.df = pd.read_csv(io.StringIO(response.text))
                
                # Cache the dataset
                self.df.to_csv(cache_file, index=False)
                print("Dataset cached successfully!")
            
            print(f"Dataset loaded with shape: {self.df.shape}")
            
            # Data cleaning and preprocessing
            print("Cleaning data...")
            initial_rows = len(self.df)
            
            # Remove rows with missing values
            self.df = self.df.dropna()
            print(f"Removed {initial_rows - len(self.df)} rows with missing values")
            
            # Convert numeric columns
            numeric_cols = ['area_sqft', 'bhk', 'bath', 'age', 'price_lakhs']
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            # Remove rows with invalid numeric values
            self.df = self.df.dropna()
            
            # Remove outliers using IQR method for price
            if 'price_lakhs' in self.df.columns:
                Q1 = self.df['price_lakhs'].quantile(0.25)
                Q3 = self.df['price_lakhs'].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                before_outlier_removal = len(self.df)
                self.df = self.df[(self.df['price_lakhs'] >= lower_bound) & (self.df['price_lakhs'] <= upper_bound)]
                print(f"Removed {before_outlier_removal - len(self.df)} outliers")
            
            # Ensure minimum data requirements
            if len(self.df) < 100:
                raise ValueError("Insufficient data after cleaning. Need at least 100 records.")
            
            print(f"Final dataset shape: {self.df.shape}")
            return self.df
            
        except requests.RequestException as e:
            raise Exception(f"Failed to download dataset: {str(e)}")
        except Exception as e:
            raise Exception(f"Data loading error: {str(e)}")
    
    def preprocess_data(self):
        """Preprocess data for machine learning"""
        try:
            print("Preprocessing data...")
            
            # Encode categorical variables
            categorical_cols = ['location', 'city', 'property_type', 'parking', 'furnishing', 'near_metro']
            
            for col in categorical_cols:
                if col in self.df.columns:
                    # Create or use existing encoder
                    if col not in self.encoders:
                        le = LabelEncoder()
                        self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
                        self.encoders[col] = le
                    else:
                        # Use existing encoder
                        le = self.encoders[col]
                        try:
                            self.df[col + '_encoded'] = le.transform(self.df[col].astype(str))
                        except ValueError:
                            # Handle new categories by refitting
                            self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
                            self.encoders[col] = le
            
            # Select features
            encoded_cols = [col for col in self.df.columns if col.endswith('_encoded')]
            numeric_cols = ['area_sqft', 'bhk', 'bath', 'age']
            
            self.feature_columns = encoded_cols + numeric_cols
            
            # Ensure all feature columns exist
            missing_cols = [col for col in self.feature_columns if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing feature columns: {missing_cols}")
            
            X = self.df[self.feature_columns].copy()
            y = self.df[self.target_column].copy()
            
            # Handle any remaining NaN values
            X = X.fillna(0)
            
            print(f"Features: {len(self.feature_columns)}")
            print(f"Samples: {len(X)}")
            
            return X, y
            
        except Exception as e:
            raise Exception(f"Data preprocessing error: {str(e)}")
    
    def train_models(self):
        """Train multiple ML models"""
        try:
            print("Starting model training...")
            
            # Load and preprocess data
            if self.df is None:
                self.load_data()
            
            X, y = self.preprocess_data()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=None
            )
            
            print(f"Training set: {X_train.shape}")
            print(f"Test set: {X_test.shape}")
            
            # Scale features for linear models
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['standard'] = scaler
            
            # Define models with optimized parameters
            models_to_train = {
                'Random Forest': RandomForestRegressor(
                    n_estimators=100,  # Reduced for faster training
                    max_depth=15, 
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1  # Use all available cores
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=100,  # Reduced for faster training
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42
                ),
                'Linear Regression': LinearRegression(),
                'SVR': SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
            }
            
            # Train and evaluate models
            for name, model in models_to_train.items():
                print(f"Training {name}...")
                
                try:
                    if name in ['Linear Regression', 'SVR']:
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Ensure non-negative accuracy
                    accuracy = max(0, r2 * 100)
                    
                    self.models[name] = model
                    self.model_metrics[name] = {
                        'MSE': float(mse),
                        'RMSE': float(rmse),
                        'MAE': float(mae),
                        'R2': float(r2),
                        'Accuracy': float(accuracy)
                    }
                    
                    print(f"{name} - R²: {r2:.3f}, RMSE: {rmse:.2f}")
                    
                except Exception as e:
                    print(f"Error training {name}: {e}")
                    continue
            
            if not self.models:
                raise Exception("No models were successfully trained")
            
            # Save models
            self.save_models()
            self.models_trained = True
            
            print("Model training completed successfully!")
            return self.model_metrics
            
        except Exception as e:
            raise Exception(f"Model training error: {str(e)}")
    
    def predict_price(self, input_data, model_name='Random Forest'):
        """Make prediction using trained model"""
        try:
            if not self.models_trained or model_name not in self.models:
                raise ValueError(f"Model {model_name} not available. Please train models first.")
            
            model = self.models[model_name]
            
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical variables
            for col, encoder in self.encoders.items():
                if col in input_data:
                    try:
                        input_df[col + '_encoded'] = encoder.transform([str(input_data[col])])
                    except ValueError:
                        # Handle unseen categories by using the most frequent category
                        most_frequent = encoder.classes_[0]
                        input_df[col + '_encoded'] = encoder.transform([str(most_frequent)])
                        print(f"Warning: Unknown {col} '{input_data[col]}', using '{most_frequent}'")
            
            # Select features and fill missing values
            X_input = input_df[self.feature_columns].fillna(0)
            
            # Scale if needed
            if model_name in ['Linear Regression', 'SVR']:
                X_input = self.scalers['standard'].transform(X_input)
            
            prediction = model.predict(X_input)[0]
            return max(0, float(prediction))  # Ensure non-negative price
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
    
    def generate_visualizations(self):
        """Generate visualization plots"""
        try:
            plots = {}
            
            # Set style for better looking plots
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Price distribution
            plt.figure(figsize=(12, 8))
            plt.hist(self.df['price_lakhs'], bins=50, alpha=0.7, color='#667eea', edgecolor='black', linewidth=1.2)
            plt.title('Distribution of Property Prices in Telangana', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('Price (Lakhs ₹)', fontsize=14, fontweight='bold')
            plt.ylabel('Frequency', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, linestyle='--')
            
            # Add statistics text
            mean_price = self.df['price_lakhs'].mean()
            median_price = self.df['price_lakhs'].median()
            plt.axvline(mean_price, color='red', linestyle='--', linewidth=2, label=f'Mean: ₹{mean_price:.1f}L')
            plt.axvline(median_price, color='green', linestyle='--', linewidth=2, label=f'Median: ₹{median_price:.1f}L')
            plt.legend(fontsize=12)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300, facecolor='white')
            buffer.seek(0)
            plots['price_distribution'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Model comparison
            plt.figure(figsize=(14, 10))
            models = list(self.model_metrics.keys())
            accuracies = [max(0, self.model_metrics[model]['Accuracy']) for model in models]
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            bars = plt.bar(models, accuracies, color=colors[:len(models)], alpha=0.8, edgecolor='black', linewidth=2)
            plt.title('Machine Learning Model Performance Comparison', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('ML Models', fontsize=14, fontweight='bold')
            plt.ylabel('Accuracy (R² Score %)', fontsize=14, fontweight='bold')
            plt.xticks(rotation=15, fontsize=12)
            plt.yticks(fontsize=12)
            plt.grid(True, alpha=0.3, axis='y', linestyle='--')
            
            # Add value labels on bars
            for bar, acc in zip(bars, accuracies):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
            
            plt.tight_layout()
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300, facecolor='white')
            buffer.seek(0)
            plots['model_comparison'] = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            # Feature correlation heatmap
            plt.figure(figsize=(14, 12))
            numeric_cols = ['area_sqft', 'bhk', 'bath', 'age', 'price_lakhs']
            available_cols = [col for col in numeric_cols if col in self.df.columns]
            
            if len(available_cols) > 1:
                corr_matrix = self.df[available_cols].corr()
                
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0,
                           square=True, linewidths=1, cbar_kws={"shrink": .8}, 
                           fmt='.2f', annot_kws={'size': 12, 'weight': 'bold'})
                plt.title('Feature Correlation Matrix', fontsize=18, fontweight='bold', pad=20)
                plt.xticks(fontsize=12, fontweight='bold')
                plt.yticks(fontsize=12, fontweight='bold')
                plt.tight_layout()
                buffer = BytesIO()
                plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300, facecolor='white')
                buffer.seek(0)
                plots['correlation_heatmap'] = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
            
            # City analysis
            if 'city' in self.df.columns:
                plt.figure(figsize=(16, 10))
                city_prices = self.df.groupby('city')['price_lakhs'].agg(['mean', 'count']).reset_index()
                city_prices = city_prices[city_prices['count'] >= 5]  # Cities with 5+ properties
                city_prices = city_prices.sort_values('mean', ascending=False).head(15)
                
                if len(city_prices) > 0:
                    bars = plt.bar(city_prices['city'], city_prices['mean'], 
                                  color='#667eea', alpha=0.8, edgecolor='black', linewidth=1.5)
                    plt.title('Average Property Prices by City (Top 15)', fontsize=18, fontweight='bold', pad=20)
                    plt.xlabel('Cities', fontsize=14, fontweight='bold')
                    plt.ylabel('Average Price (Lakhs ₹)', fontsize=14, fontweight='bold')
                    plt.xticks(rotation=45, ha='right', fontsize=11)
                    plt.yticks(fontsize=12)
                    plt.grid(True, alpha=0.3, axis='y', linestyle='--')
                    
                    # Add value labels on bars
                    for bar, price in zip(bars, city_prices['mean']):
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                                f'₹{price:.1f}L', ha='center', va='bottom', fontweight='bold', fontsize=10)
                    
                    plt.tight_layout()
                    buffer = BytesIO()
                    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300, facecolor='white')
                    buffer.seek(0)
                    plots['city_analysis'] = base64.b64encode(buffer.getvalue()).decode()
                    plt.close()
            
            return plots
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return {}

# Initialize the predictor
predictor = RealEstateMLPredictor()

# Try to load existing models on startup
try:
    if predictor.load_models():
        print("Pre-trained models loaded successfully!")
        session_data = {
            'trained': True,
            'metrics': predictor.model_metrics,
            'plots': predictor.generate_visualizations() if predictor.df is not None else {},
            'training_timestamp': predictor.get_training_timestamp()
        }
    else:
        print("No pre-trained models found. Training will be required.")
        session_data = {'trained': False, 'metrics': {}, 'plots': {}, 'training_timestamp': 'Unknown'}
except Exception as e:
    print(f"Error loading models on startup: {e}")
    session_data = {'trained': False, 'metrics': {}, 'plots': {}, 'training_timestamp': 'Unknown'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train():
    if request.method == 'POST':
        try:
            # Train models
            metrics = predictor.train_models()
            plots = predictor.generate_visualizations()
            
            # Update session
            session['trained'] = True
            session['metrics'] = metrics
            session['plots'] = plots
            session['training_timestamp'] = predictor.get_training_timestamp()
            
            return jsonify({
                'success': True,
                'metrics': metrics,
                'plots': plots,
                'dataset_info': {
                    'shape': predictor.df.shape,
                    'features': len(predictor.feature_columns),
                    'cities': predictor.df['city'].nunique() if 'city' in predictor.df.columns else 0,
                    'locations': predictor.df['location'].nunique() if 'location' in predictor.df.columns else 0
                },
                'training_timestamp': predictor.get_training_timestamp()
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # Check if models are already trained
    trained = session.get('trained', session_data.get('trained', False))
    return render_template('train.html', 
                         already_trained=trained,
                         training_timestamp=session.get('training_timestamp', session_data.get('training_timestamp', 'Unknown')))

@app.route('/dashboard')
def dashboard():
    trained = session.get('trained', session_data.get('trained', False))
    
    if not trained:
        return render_template('dashboard.html', trained=False)
    
    metrics = session.get('metrics', session_data.get('metrics', {}))
    plots = session.get('plots', session_data.get('plots', {}))
    
    # Ensure metrics have valid values
    cleaned_metrics = {}
    for model_name, model_metrics in metrics.items():
        cleaned_metrics[model_name] = {
            'MSE': float(model_metrics.get('MSE', 0)),
            'RMSE': float(model_metrics.get('RMSE', 0)),
            'MAE': float(model_metrics.get('MAE', 0)),
            'R2': float(model_metrics.get('R2', 0)),
            'Accuracy': max(0, float(model_metrics.get('Accuracy', 0)))
        }
    
    return render_template('dashboard.html', 
                         trained=True, 
                         metrics=cleaned_metrics, 
                         plots=plots,
                         training_timestamp=session.get('training_timestamp', session_data.get('training_timestamp', 'Unknown')))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    trained = session.get('trained', session_data.get('trained', False))
    
    if not trained:
        return render_template('predict.html', trained=False)
    
    if request.method == 'POST':
        try:
            input_data = {
                'location': request.form['location'],
                'city': request.form['city'],
                'property_type': request.form['property_type'],
                'area_sqft': float(request.form['area_sqft']),
                'bhk': int(request.form['bhk']),
                'bath': int(request.form['bath']),
                'parking': request.form['parking'],
                'age': int(request.form['age']),
                'furnishing': request.form['furnishing'],
                'near_metro': request.form['near_metro']
            }
            
            model_name = request.form.get('model', 'Random Forest')
            prediction = predictor.predict_price(input_data, model_name)
            
            # Get model accuracy
            metrics = session.get('metrics', session_data.get('metrics', {}))
            model_accuracy = metrics.get(model_name, {}).get('Accuracy', 0)
            
            return jsonify({
                'success': True,
                'prediction': round(prediction, 2),
                'model_used': model_name,
                'model_accuracy': round(model_accuracy, 1),
                'input_data': input_data
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    
    # Get unique values for dropdowns
    unique_values = {}
    if predictor.df is not None:
        try:
            unique_values = {
                'locations': sorted(predictor.df['location'].unique().tolist()) if 'location' in predictor.df.columns else [],
                'cities': sorted(predictor.df['city'].unique().tolist()) if 'city' in predictor.df.columns else [],
                'property_types': sorted(predictor.df['property_type'].unique().tolist()) if 'property_type' in predictor.df.columns else [],
                'parking_options': ['Yes', 'No'],
                'furnishing_options': sorted(predictor.df['furnishing'].unique().tolist()) if 'furnishing' in predictor.df.columns else ['Furnished', 'Semi-Furnished', 'Unfurnished'],
                'metro_options': ['Yes', 'No']
            }
        except Exception as e:
            print(f"Error getting unique values: {e}")
            unique_values = {
                'locations': [],
                'cities': [],
                'property_types': [],
                'parking_options': ['Yes', 'No'],
                'furnishing_options': ['Furnished', 'Semi-Furnished', 'Unfurnished'],
                'metro_options': ['Yes', 'No']
            }
    
    return render_template('predict.html', 
                         trained=True, 
                         unique_values=unique_values)

@app.route('/api/model_info')
def model_info():
    """API endpoint to get model information"""
    trained = session.get('trained', session_data.get('trained', False))
    
    if not trained:
        return jsonify({'error': 'Models not trained yet'})
    
    return jsonify({
        'metrics': session.get('metrics', session_data.get('metrics', {})),
        'dataset_info': {
            'total_records': predictor.df.shape[0] if predictor.df is not None else 0,
            'features': len(predictor.feature_columns),
            'cities': predictor.df['city'].nunique() if predictor.df is not None and 'city' in predictor.df.columns else 0
        },
        'training_timestamp': session.get('training_timestamp', session_data.get('training_timestamp', 'Unknown'))
    })

@app.route('/retrain', methods=['POST'])
def retrain():
    """Force retrain models"""
    try:
        # Clear existing models
        predictor.models = {}
        predictor.model_metrics = {}
        predictor.models_trained = False
        
        # Train new models
        metrics = predictor.train_models()
        plots = predictor.generate_visualizations()
        
        # Update session
        session['trained'] = True
        session['metrics'] = metrics
        session['plots'] = plots
        session['training_timestamp'] = predictor.get_training_timestamp()
        
        return jsonify({
            'success': True,
            'message': 'Models retrained successfully!',
            'metrics': metrics,
            'training_timestamp': predictor.get_training_timestamp()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
