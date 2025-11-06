"""
Simple script to run the Flask application
"""
from app import app

if __name__ == '__main__':
    print("Starting Traffic Light Control System...")
    print("Open your browser and navigate to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)

