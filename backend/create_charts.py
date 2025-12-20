import csv
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import sys
import subprocess

def install_matplotlib():
    try:
        import matplotlib
    except ImportError:
        print("Installing matplotlib for charts...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

def generate_charts():
    # Ensure matplotlib is installed
    install_matplotlib()
    
    # Path to logs
    log_dir = Path.home() / "Desktop" / "OceanTrashLogs"
    if not log_dir.exists():
        print(f"❌ No logs found at {log_dir}")
        print("Run the detector first to collect some data!")
        return

    # Aggregate Data
    label_counts = Counter()
    total_detections = 0
    
    print(f"Reading logs from {log_dir}...")
    for log_file in log_dir.glob("*.csv"):
        try:
            with open(log_file, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header: continue
                
                for row in reader:
                    if len(row) >= 3:
                        label = row[2]
                        # Clean up labels if needed (e.g. remove ID)
                        label_counts[label] += 1
                        total_detections += 1
        except Exception as e:
            print(f"Skipping {log_file}: {e}")

    if total_detections == 0:
        print("❌ No detections found in logs.")
        return

    # Prepare Data
    # Sort by count (descending)
    sorted_items = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_items]
    sizes = [x[1] for x in sorted_items]
    
    # Setup Style
    plt.style.use('ggplot')
    
    # 1. Pie Chart (Top 6 + Other)
    plt.figure(figsize=(12, 8))
    
    # Group small items
    top_n = 6
    if len(labels) > top_n:
        pie_labels = labels[:top_n] + ['Other']
        pie_sizes = sizes[:top_n] + [sum(sizes[top_n:])]
    else:
        pie_labels = labels
        pie_sizes = sizes
        
    colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99', '#c2c2f0', '#ffb3e6', '#c4c4c4']
    
    # Explode the 1st slice (biggest)
    explode = [0.1] + [0] * (len(pie_labels) - 1)
    
    plt.pie(pie_sizes, labels=pie_labels, colors=colors[:len(pie_labels)], 
            autopct='%1.1f%%', startangle=140, shadow=True, explode=explode)
    plt.axis('equal')
    plt.title(f'Ocean Trash Composition (Top {top_n})', fontsize=16)
    
    output_path = Path.home() / "Desktop" / "Trash_PieChart.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Pie Chart saved to: {output_path}")
    
    # 2. Horizontal Bar Chart (All Items)
    plt.figure(figsize=(12, len(labels) * 0.5 + 4)) # Dynamic height
    
    # Create horizontal bars
    y_pos = range(len(labels))
    plt.barh(y_pos, sizes, color='#1a73e8')
    plt.yticks(y_pos, labels)
    plt.gca().invert_yaxis() # Biggest at top
    
    plt.xlabel('Count')
    plt.title('Detected Items Count (All Types)')
    plt.tight_layout()
    
    output_path_bar = Path.home() / "Desktop" / "Trash_BarChart.png"
    plt.savefig(output_path_bar, dpi=300, bbox_inches='tight')
    print(f"✅ Bar Chart saved to: {output_path_bar}")

if __name__ == "__main__":
    generate_charts()
