import os
import json
import glob
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
REPORT_DIR = "report"
os.makedirs(REPORT_DIR, exist_ok=True)

def generate_report():
    print("Gathering results...")
    
    # We expect results to be in path_a_{size}_seed_{seed} and path_b_shakespeare_{size}_seed_{seed}
    # Let's plot the average validation loss across seeds for each size
    sizes = ['100k', '1M', '2M', '3M', '4M', '5M']
    
    html_content = f"""
    <html>
    <head>
        <title>NanoGPT Overfitting Transfer Learning Research Report</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 2rem; color: #333; }}
            h1 {{ border-bottom: 2px solid #eaecef; padding-bottom: 0.3em; }}
            .graph {{ margin: 2rem 0; text-align: center; border: 1px solid #ddd; padding: 1rem; border-radius: 8px; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <h1>NanoGPT Overfitting Transfer Learning Research Report</h1>
        <p>This report compares two training paths across different model sizes (100k to 5M parameters). Models were trained to predict Tiny Shakespeare characters.</p>
        <ul>
            <li><strong>Path A (Control):</strong> Trained directly on Tiny Shakespeare from standard initialization.</li>
            <li><strong>Path B (Experiment):</strong> Trained to extreme overfitting on a structured Arabic-script corpus, then fine-tuned on Tiny Shakespeare.</li>
        </ul>
    """

    for size in sizes:
        path_a_dirs = glob.glob(os.path.join(RESULTS_DIR, f"path_a_{size}_seed_*"))
        path_b_dirs = glob.glob(os.path.join(RESULTS_DIR, f"path_b_shakespeare_{size}_seed_*"))
        
        if not path_a_dirs and not path_b_dirs:
            continue
            
        # We will plot Path A vs Path B for the first available seed just as an example, 
        # or average them if we want to be rigorous. For simplicity, let's plot Seed 42 if available.
        # Let's plot all seeds with light colors and average with bold.
        
        plt.figure(figsize=(10, 6))
        
        # Helper to plot a set of directories
        def plot_dirs(dirs, label, color):
            for i, d in enumerate(dirs):
                metrics_file = os.path.join(d, 'metrics.json')
                if os.path.exists(metrics_file):
                    with open(metrics_file, 'r') as f:
                        metrics = json.load(f)
                    
                    l = label if i == 0 else "_nolegend_"
                    plt.plot(metrics['iters'], metrics['val_loss'], label=l, color=color, alpha=0.7)
        
        plot_dirs(path_a_dirs, "Path A (Control)", "blue")
        plot_dirs(path_b_dirs, "Path B (Pre-trained on Arabic Corpus)", "red")
        
        plt.title(f"Validation Loss vs Iterations ({size} Model)")
        plt.xlabel("Iterations")
        plt.ylabel("Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_path = os.path.join(REPORT_DIR, f"loss_{size}.png")
        plt.savefig(plot_path)
        plt.close()
        
        html_content += f"""
        <div class="graph">
            <h2>{size} Parameter Model</h2>
            <img src="loss_{size}.png" alt="Loss graph for {size}">
        </div>
        """
        
    html_content += """
    </body>
    </html>
    """
    
    report_file = os.path.join(REPORT_DIR, "index.html")
    with open(report_file, 'w') as f:
        f.write(html_content)
    
    print(f"Report generated at {report_file}")

if __name__ == '__main__':
    generate_report()
