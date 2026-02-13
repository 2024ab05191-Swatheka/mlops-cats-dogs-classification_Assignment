"""
Example local configuration file for team members
Copy this file to 'local_config.py' and update paths for your local setup
NOTE: local_config.py is gitignored, so your personal paths won't be committed
"""

class LocalConfig:
    # TEAM MEMBERS: Update this path to your local dataset location
    
    # Example for Windows:
    RAW_DATA_PATH = r'C:\Users\YOUR_USERNAME\Downloads\PetImages'
    
    # Example for Linux/Mac:
    # RAW_DATA_PATH = '/home/YOUR_USERNAME/datasets/PetImages'
    
    # Example for different drive:
    # RAW_DATA_PATH = r'D:\Datasets\PetImages'
    
    # You can also override other settings if needed:
    # NUM_EPOCHS = 5  # Use fewer epochs for testing
    # BATCH_SIZE = 16  # Smaller batch size for limited GPU memory
