import time
import json
import torch
import requests
import numpy as np
import firebase_admin
from datetime import datetime
from firebase_admin import credentials, db
from processing import hr_to_hrv_freq_features
from model import StressNet

model = torch.load("full_model.pt", weights_only=False)
model.eval()

cred = credentials.Certificate("autism-dba09-firebase-adminsdk-fbsvc-451ebc5eb8.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://autism-dba09-default-rtdb.firebaseio.com'
})

ref = db.reference('test_write_health')

class_names = {1: 'Amusement', 2: 'Calm', 0: 'Stress'}

# Define feature order
ordered_keys = ['HF', 'LF', 'VLF']  # adjust based on training order

while True:
    data = ref.get()

    hr_entries = []
    for entry in data.values():
        if 'heartRate' in entry and 'timestamp' in entry:
            hr_entries.append((entry['timestamp'], entry['heartRate']))
    
    hr_entries.sort(key=lambda x: x[0])
    last_entries = hr_entries[-120:]

    hr_values = [
        float(hr) for _, hr in last_entries
        if hr is not None and str(hr).lower() != "nan" and hr != 0 and isinstance(hr, (int, float, str))
    ]

    hr_array = np.nan_to_num(np.array(hr_values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    print("HR values before HRV processing:", hr_array)

    if len(hr_array) >= 60:
        try:
            features = hr_to_hrv_freq_features(hr_array, sampling_rate_hz=1/1.5)
            print("HRV Frequency Domain Features before predictions:", features)
            # Convert dict to numeric array
            numeric_features = np.array([features[k] for k in ordered_keys], dtype=np.float32)  

            with torch.no_grad():
                input_tensor = torch.from_numpy(numeric_features).unsqueeze(0)
                output = model(input_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
                print("Predicted index:", predicted_class)
            
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "hrv_features": {k: float(features[k]) for k in ordered_keys},
                "prediction": float(predicted_class),
                "label": class_names[predicted_class]
            }
            print(json.dumps(log_data, indent=2))

            
            # Send prediction to FastAPI endpoint
            requests.post("http://127.0.0.1:8000/store_prediction/", json={
                "prediction": float(predicted_class)
            })

        except Exception as e:
            error_log = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "hr_values": hr_array.tolist()
            }
            print(json.dumps(error_log, indent=2))

    else:
        waiting_log = {
            "timestamp": datetime.now().isoformat(),
            "status": "waiting",
            "current_length": len(hr_array),
            "required_length": 60
        }
        print(json.dumps(waiting_log, indent=2))
    
    time.sleep(300)
