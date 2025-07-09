import os
import sys
import subprocess

def install_requirements():
    """Install required packages"""
    requirements = [
        'flask==3.0.0',
        'pandas==2.1.4',
        'numpy==1.24.3',
        'scikit-learn==1.3.2',
        'matplotlib==3.8.2',
        'seaborn==0.13.0',
        'requests==2.31.0'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    return True

def create_directories():
    """Create necessary directories"""
    directories = ['templates', 'static', 'models', 'data', 'static/plots']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created {directory} directory")

def main():
    print("🏠 Real Estate ML Predictor Setup")
    print("=" * 50)
    
    # Create directories
    create_directories()
    print("✅ All directories created successfully!")
    
    # Install requirements
    if install_requirements():
        print("✅ All packages installed successfully!")
    else:
        print("❌ Some packages failed to install. Please install manually:")
        print("   pip install flask pandas numpy scikit-learn matplotlib seaborn requests")
        return
    
    print("\n🚀 Setup completed successfully!")
    print("\nTo run the application:")
    print("1. Run: python app.py")
    print("2. Open browser: http://localhost:5000")
    print("\n📋 Features:")
    print("✅ Advanced ML models (Random Forest, Gradient Boosting, Linear Regression, SVR)")
    print("✅ Model persistence - train once, use forever!")
    print("✅ Real-time model training and evaluation")
    print("✅ Comprehensive dashboard with visualizations")
    print("✅ Interactive prediction interface")
    print("✅ Responsive, professional UI design")
    print("✅ Real implementation with actual Telangana real estate data")
    print("✅ High accuracy predictions with multiple model comparison")
    print("✅ Automatic data caching for faster loading")
    print("✅ Error handling and recovery")
    
    print("\n💡 Pro Tips:")
    print("• Models are saved automatically after training")
    print("• No need to retrain every time you restart the app")
    print("• Use 'Retrain Models' only when you want fresh data")
    print("• The app will load pre-trained models on startup if available")

if __name__ == "__main__":
    main()
