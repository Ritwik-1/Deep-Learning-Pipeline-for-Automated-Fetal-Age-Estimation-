import os
import re
from collections import defaultdict
import shutil

seg_folder = r'OUTPUT_2\SEG'
output_folder = r'OUTPUT_2\CLEANED_SEG'
os.makedirs(output_folder, exist_ok=True)

pattern = re.compile(r'^(Patient\d+)_.*Fetal (abdomen|femur|brain)_seg_(\d+\.\d+)\.png$', re.IGNORECASE)

best_files = defaultdict(lambda: ('', -1.0))  # (filename, probability)

for filename in os.listdir(seg_folder):
    if filename.endswith('.png'):
        match = pattern.match(filename)
        if match:
            patient_id, organ_class, prob_str = match.groups()
            prob = float(prob_str)
            key = (patient_id, organ_class.lower())
            
            if prob > best_files[key][1]:  
                best_files[key] = (filename, prob)

for (patient_id, organ_class), (filename, prob) in best_files.items():
    src_path = os.path.join(seg_folder, filename)
    dst_path = os.path.join(output_folder, filename)
    shutil.copy2(src_path, dst_path)

print(f"Cleaned files copied to: {output_folder}")
