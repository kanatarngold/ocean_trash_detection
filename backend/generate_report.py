import csv
import os
from pathlib import Path
from collections import Counter
import json

def generate_report():
    # Path to logs
    log_dir = Path.home() / "Desktop" / "OceanTrashLogs"
    if not log_dir.exists():
        print("No logs found!")
        return

    # Aggregate Data
    total_detections = 0
    label_counts = Counter()
    files_processed = 0

    print(f"Reading logs from {log_dir}...")
    
    for log_file in log_dir.glob("*.csv"):
        try:
            with open(log_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None) # Skip header
                if not header: continue
                
                for row in reader:
                    if len(row) >= 3:
                        label = row[2] # Label is the 3rd column
                        label_counts[label] += 1
                        total_detections += 1
            files_processed += 1
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    if total_detections == 0:
        print("No detections found in logs.")
        return

    # Prepare Data for Chart.js
    labels = list(label_counts.keys())
    data = list(label_counts.values())
    
    # Generate HTML
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Ocean Trash Mission Report</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; padding: 40px; }}
            .container {{ max_width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            h1 {{ color: #1a73e8; text-align: center; }}
            .stats {{ display: flex; justify-content: space-around; margin: 30px 0; }}
            .stat-box {{ text-align: center; padding: 20px; background: #e8f0fe; border-radius: 10px; width: 30%; }}
            .stat-number {{ font-size: 2em; font-weight: bold; color: #1a73e8; }}
            .chart-container {{ position: relative; height: 400px; width: 100%; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌊 Ocean Trash Mission Report</h1>
            
            <div class="stats">
                <div class="stat-box">
                    <div class="stat-number">{files_processed}</div>
                    <div>Missions Flown</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{total_detections}</div>
                    <div>Items Detected</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{len(labels)}</div>
                    <div>Trash Types</div>
                </div>
            </div>

            <div class="chart-container">
                <canvas id="trashChart"></canvas>
            </div>
        </div>

        <script>
            const ctx = document.getElementById('trashChart').getContext('2d');
            new Chart(ctx, {{
                type: 'doughnut',
                data: {{
                    labels: {json.dumps(labels)},
                    datasets: [{{
                        data: {json.dumps(data)},
                        backgroundColor: [
                            '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#C9CBCF'
                        ],
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'bottom' }},
                        title: {{ display: true, text: 'Trash Composition' }}
                    }}
                }}
            }});
        </script>
    </body>
    </html>
    """

    # Save Report
    report_path = Path.home() / "Desktop" / "MissionReport.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
        
    print(f"✓ Report generated: {report_path}")

if __name__ == "__main__":
    generate_report()
