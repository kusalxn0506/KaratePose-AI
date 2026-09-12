import os
import subprocess
import sys

# Auto-install python-docx so we can generate the Word file
try:
    import docx
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx", "-q"])
    import docx

from docx import Document
from docx.shared import Inches

doc = Document()
doc.add_heading("Today's Testing Session & Plot Explanations", 0)

doc.add_paragraph("Today I ran the full Phase A Python pipeline on a sample video to ensure the code architecture works before real data collection. The pipeline successfully ran YOLOv8 and extracted 4,784 joint coordinates.")
doc.add_paragraph("The plots below were generated using Synthetic (Mock) Data. This proves the plotting engine works. Once the real videos are filmed, the system will swap the fake data for the real coordinates.")

plots_dir = "output/plots"

# Plot 1
doc.add_heading('1. Statistical Summary Table', level=2)
doc.add_paragraph("Explanation: I built the statistical engine first using fake data. When we feed in the real video, this table will run Wilcoxon tests to mathematically prove exactly how much worse the AI gets when someone wears a Karate Gi compared to tight clothes.")
p1 = os.path.join(plots_dir, 'statistical_summary.png')
if os.path.exists(p1):
    doc.add_picture(p1, width=Inches(6.0))

# Plot 2
doc.add_heading('2. Spatial Jitter Boxplot', level=2)
doc.add_paragraph("Explanation: This is our main visual proof. Jitter is how much the AI's tracking shakes. Using fake data, I proved my code can graph Set A vs Set B side-by-side. For the real paper, we expect the orange boxes (the Gi) to be much higher, especially on the wrists and ankles.")
p2 = os.path.join(plots_dir, 'jitter_all_joints.png')
if os.path.exists(p2):
    doc.add_picture(p2, width=Inches(6.0))

# Plot 3
doc.add_heading('3. Confidence Heatmap', level=2)
doc.add_paragraph("Explanation: This heatmap tracks the AI's confidence. Dark colours mean the AI completely lost track of the limb. This graph will allow us to point to the exact millisecond during a punch where the Gi causes the AI to fail.")
p3 = os.path.join(plots_dir, 'confidence_heatmap.png')
if os.path.exists(p3):
    doc.add_picture(p3, width=Inches(6.0))

doc.save('Lecturer_Explanation.docx')
print("✅ Word Document generated successfully!")
