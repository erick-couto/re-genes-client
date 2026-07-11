import os
import glob
import sys

def reset_population():
    print("🧹 Cleaning up NEAT population data...")
    
    # 1. Remove Checkpoints
    # Use explicit path join to be safe, though glob works relative to script if run from there
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints = glob.glob(os.path.join(base_dir, "neat-checkpoint-*"))
    
    deleted_count = 0
    for cp in checkpoints:
        try:
            os.remove(cp)
            print(f"   🗑️ Deleted checkpoint: {os.path.basename(cp)}")
            deleted_count += 1
        except OSError as e:
            print(f"   ❌ Error deleting {os.path.basename(cp)}: {e}")

    # 2. Remove Stats Logs (Optional but recommended for fresh stats)
    stats_files = ["neat_performance.csv", "NEAT_HALL_OF_FAME.md"]
    for sf in stats_files:
        full_path = os.path.join(base_dir, sf)
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
                print(f"   📉 Deleted stats log: {sf}")
                deleted_count += 1
            except OSError as e:
                print(f"   ❌ Error deleting {sf}: {e}")
                
    if deleted_count == 0:
        print("✨ Nothing to clean. Already fresh!")
    else:
        print(f"\n✨ Reset complete! {deleted_count} files removed.")
        print("🚀 Ready for a fresh start.")

if __name__ == "__main__":
    # If run directly, ask for confirmation unless --force is used
    if "--force" in sys.argv:
        reset_population()
    else:
        print("⚠️  WARNING: This will PERMANENTLY DELETE all genetic history, checkpoints, and stats.")
        confirm = input("are you sure you want to reset the population? (y/N): ")
        if confirm.lower() == 'y':
            reset_population()
        else:
            print("❌ Cancelled.")
