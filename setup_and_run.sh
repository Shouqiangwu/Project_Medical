#!/bin/bash
# ============================================================
# AutoDL one-click setup & run script
# Upload the entire code_medical/ folder to your server, then:
#   cd code_medical && bash setup_and_run.sh
# ============================================================
set -e

echo "========== Step 1: Install dependencies =========="
pip install torch torchvision timm medvae pandas numpy matplotlib Pillow

echo "========== Step 2: Verify MedVAE =========="
python -c "from medvae import MVAE; print('MedVAE OK')"

echo "========== Step 3: Verify data =========="
# Adjust DATA_DIR to your actual data path on AutoDL
DATA_DIR="./data"

if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory $DATA_DIR not found!"
    echo "Please download Kaggle NIH CXR-14 (224x224) and extract to $DATA_DIR"
    echo "Expected: $DATA_DIR/images_224/*.png + Data_Entry_2017.csv + train_val_list.txt + test_list.txt"
    exit 1
fi

echo "Data directory found: $DATA_DIR"

echo "========== Step 4: Train baselines (fills baselines.json) =========="
python train_baselines.py --data_dir "$DATA_DIR" --epochs 50 --batch_size 64

echo "========== Step 5: Run evolution =========="
python main.py --data_dir "$DATA_DIR" --N 50 --IPC 50 --G 30 --seed 2025

echo "========== Step 6: Final evaluation =========="
python eval_final.py --z_path best_z.pt --data_dir "$DATA_DIR" --steps 1000

echo "========== All done! =========="
