import os
import sys
import json
import http.server
import socketserver
import socket

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))

# Load model
model_path = "best_startup_model.joblib"
model = None
pd = None
if os.path.exists(model_path):
    try:
        import joblib
        import pandas as pd

        model = joblib.load(model_path)
        print(f"Successfully loaded model from {model_path}")
    except ModuleNotFoundError as e:
        print(f"Warning: missing package {e.name}; /predict will use the built-in fallback formula.")
    except Exception as e:
        print(f"Warning: could not load {model_path}: {e}")
        print("/predict will use the built-in fallback formula.")
else:
    print(f"Warning: {model_path} not found. /predict will use the built-in fallback formula.")


def fallback_predict(rd_spend, admin_spend, marketing_spend, state):
    std_rd = (rd_spend - 73721.6156) / 45902.256482
    std_admin = (admin_spend - 121344.6396) / 28017.802755
    std_mkt = (marketing_spend - 211025.0978) / 122290.310726
    state_val = 0
    if state == "Florida":
        state_val = 83.9
    elif state == "New York":
        state_val = -95.1
    return 112012.6392 + (36633.27 * std_rd) + (-1675.24 * std_admin) + (3564.08 * std_mkt) + state_val

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
                
                if model is not None and pd is not None:
                    df = pd.DataFrame([{
                        'R&D Spend': rd_spend,
                        'Administration': admin_spend,
                        'Marketing Spend': marketing_spend,
                        'State': state
                    }])
                    prediction = float(model.predict(df)[0])
                    source = "model"
                else:
                    prediction = float(fallback_predict(rd_spend, admin_spend, marketing_spend, state))
                    source = "fallback"
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'predicted_profit': prediction,
                    'source': source
                }).encode('utf-8'))
                print(f"Prediction ({source}): R&D={rd_spend}, Admin={admin_spend}, Mkt={marketing_spend}, State={state} -> Profit={prediction:.2f}")
                
            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
                print(f"Error handling prediction request: {e}", file=sys.stderr)
        else:
            super().do_POST()

class ThreadingReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def get_lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "你的電腦區網 IP"


# Start HTTP server
handler = StartupPredictHandler

with ThreadingReusableTCPServer((HOST, PORT), handler) as httpd:
    lan_ip = get_lan_ip()
    print(f"Server running on {HOST}:{PORT}")
    print(f"Local:   http://localhost:{PORT}")
    print(f"Network: http://{lan_ip}:{PORT}")
    print("同一個 Wi-Fi / 區網內的裝置可用 Network 網址開啟。")
    print("若外部裝置連不上，請允許 Windows 防火牆開放 Python 或此 port。")
    print("Press Ctrl+C to terminate the server.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()
