import os
import sys
import json
import http.server
import socketserver
import joblib
import pandas as pd

PORT = 8000

# Load model
model_path = "best_startup_model.joblib"
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found! Please run solve_50_startups.py first to train the model.")
    sys.exit(1)

model = joblib.load(model_path)
print(f"Successfully loaded model from {model_path}")

class StartupPredictHandler(http.server.SimpleHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        if self.path == '/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                # Extract parameters
                rd_spend = float(data.get('rd_spend', 73721.6))
                admin_spend = float(data.get('admin_spend', 121344.6))
                marketing_spend = float(data.get('marketing_spend', 211025.1))
                state = data.get('state', 'California')
                
                # Make prediction
                df = pd.DataFrame([{
                    'R&D Spend': rd_spend,
                    'Administration': admin_spend,
                    'Marketing Spend': marketing_spend,
                    'State': state
                }])
                
                prediction = model.predict(df)[0]
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'predicted_profit': prediction}).encode('utf-8'))
                print(f"Prediction: R&D={rd_spend}, Admin={admin_spend}, Mkt={marketing_spend}, State={state} -> Profit={prediction:.2f}")
                
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
                print(f"Error handling prediction request: {e}", file=sys.stderr)
        else:
            super().do_POST()

# Start HTTP server
handler = StartupPredictHandler

with socketserver.TCPServer(("", PORT), handler) as httpd:
    print(f"Server running at http://localhost:{PORT}")
    print("Press Ctrl+C to terminate the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
