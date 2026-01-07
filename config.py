"""
Configuration file for the model
Centralized hyperparameters and settings
"""

# Data Configuration
DATA_CONFIG = {
    'train_csv': 'train.csv',
    'test_csv': 'test.csv',
    'novels_dir': 'novels',
    'val_split': 0.2,
}

# Text Processing Configuration
PROCESSING_CONFIG = {
    'chunk_size': 1000,          # Words per chunk
    'overlap': 200,              # Overlapping words
    'chunk_strategy': 'sliding', # 'sliding' or 'chapter'
    'max_chunks': 100,           # Maximum chunks per narrative
    'encoder_model': 'all-MiniLM-L6-v2',  # Sentence transformer model
}

# Model Architecture Configuration
MODEL_CONFIG = {
    'embedding_dim': 384,        # Will be auto-set by encoder
    'hidden_dim': 512,
    'num_layers': 3,
    'dropout': 0.3,
}

# Training Configuration
TRAINING_CONFIG = {
    'batch_size': 4,
    'epochs': 20,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'use_focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0,
    'gradient_clip': 1.0,
}

# Prediction Configuration
PREDICTION_CONFIG = {
    'batch_size': 4,
    'use_hybrid': False,         # Use hybrid reasoning (neural + rules)
    'rule_weight': 0.3,          # Weight for rule-based component
}

# Output Configuration
OUTPUT_CONFIG = {
    'checkpoint_dir': 'checkpoints',
    'results_file': 'results.csv',
    'save_every': 1,             # Save checkpoint every N epochs
}

# Device Configuration
DEVICE_CONFIG = {
    'device': 'auto',            # 'cuda', 'cpu', or 'auto'
    'num_workers': 0,            # DataLoader workers
    'pin_memory': True,
}

# Logging Configuration
LOGGING_CONFIG = {
    'verbose': True,
    'log_file': 'training.log',
    'save_history': True,
}
