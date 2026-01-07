"""
Quick test script to verify setup and data
"""

import pandas as pd
import os
import sys

def check_files():
    """Check if all required files exist"""
    print("=" * 60)
    print("CHECKING FILE SETUP")
    print("=" * 60)
    
    required_files = ['train.csv', 'test.csv', 'requirements.txt']
    missing = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"✗ {file} MISSING")
            missing.append(file)
    
    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False
    
    print("\n✓ All required files present")
    return True


def check_data():
    """Check data format and content"""
    print("\n" + "=" * 60)
    print("CHECKING DATA FORMAT")
    print("=" * 60)
    
    # Check train.csv
    print("\nTrain Data:")
    try:
        train_df = pd.read_csv('train.csv')
        print(f"  Rows: {len(train_df)}")
        print(f"  Columns: {list(train_df.columns)}")
        print(f"  Labels: {train_df['label'].value_counts().to_dict()}")
        
        # Show sample
        print("\n  Sample row:")
        sample = train_df.iloc[0]
        print(f"    ID: {sample['id']}")
        print(f"    Book: {sample['book_name']}")
        print(f"    Character: {sample['char']}")
        print(f"    Backstory (first 100 chars): {sample['content'][:100]}...")
        print(f"    Label: {sample['label']}")
        
    except Exception as e:
        print(f"  ✗ Error reading train.csv: {e}")
        return False
    
    # Check test.csv
    print("\nTest Data:")
    try:
        test_df = pd.read_csv('test.csv')
        print(f"  Rows: {len(test_df)}")
        print(f"  Columns: {list(test_df.columns)}")
        
    except Exception as e:
        print(f"  ✗ Error reading test.csv: {e}")
        return False
    
    print("\n✓ Data format looks good")
    return True


def check_novels():
    """Check if novels directory and files exist"""
    print("\n" + "=" * 60)
    print("CHECKING NOVELS DIRECTORY")
    print("=" * 60)
    
    if not os.path.exists('novels'):
        print("✗ 'novels' directory not found")
        print("\n⚠️  You need to create a 'novels' directory with .txt files")
        print("   Example structure:")
        print("   novels/")
        print("   ├── In Search of the Castaways.txt")
        print("   └── The Count of Monte Cristo.txt")
        return False
    
    novels = [f for f in os.listdir('novels') if f.endswith('.txt')]
    
    if not novels:
        print("✗ No .txt files found in 'novels' directory")
        return False
    
    print(f"✓ Found {len(novels)} novel files:")
    for novel in novels[:5]:  # Show first 5
        size = os.path.getsize(os.path.join('novels', novel))
        print(f"  - {novel} ({size:,} bytes)")
    
    if len(novels) > 5:
        print(f"  ... and {len(novels) - 5} more")
    
    return True


def check_dependencies():
    """Check if key dependencies are installed"""
    print("\n" + "=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)
    
    dependencies = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sentence_transformers': 'Sentence Transformers',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} NOT INSTALLED")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("\nRun: pip install -r requirements.txt")
        return False
    
    print("\n✓ All dependencies installed")
    return True


def show_next_steps():
    """Show next steps to user"""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    print("\n1. Ensure 'novels' directory contains all novel .txt files")
    print("\n2. Train the model:")
    print("   python train.py --epochs 10 --batch_size 2")
    print("\n3. Generate predictions:")
    print("   python predict.py --model_path checkpoints/best_model.pt")
    print("\n4. Submit results.csv")
    print("\nOr use the quick start script:")
    print("   chmod +x run.sh")
    print("   ./run.sh")


def main():
    """Run all checks"""
    print("\n")
    print("*" * 60)
    print("NARRATIVE CONSISTENCY MODEL - SETUP VERIFICATION")
    print("Track B - BDH-Inspired Reasoning")
    print("*" * 60)
    print("\n")
    
    checks = [
        check_files(),
        check_data(),
        check_novels(),
        check_dependencies()
    ]
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all(checks):
        print("\n✓ All checks passed! You're ready to train.")
    else:
        print("\n⚠️  Some checks failed. Please fix the issues above.")
    
    show_next_steps()
    
    print("\n" + "*" * 60)
    print("\n")


if __name__ == '__main__':
    main()
