"""
Inspect what metrics are available in TensorBoard logs.
"""

import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGS_DIR = '../logs/'

# Check one log file
log_dir = os.path.join(LOGS_DIR, 'mnist_0')
event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))

if event_files:
    event_file = event_files[0]
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    print(f"Inspecting: {log_dir}\n")
    print("Available scalar tags:")
    for tag in ea.Tags()['scalars']:
        print(f"  - {tag}")
        # Get last value
        events = ea.Scalars(tag)
        if events:
            print(f"    Last value: {events[-1].value:.4f}")
    
    print(f"\nTotal scalars found: {len(ea.Tags()['scalars'])}")
else:
    print(f"No event files found in {log_dir}")
