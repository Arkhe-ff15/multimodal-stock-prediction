# Run this to download ~2GB of real FNSPID data
from src.fnspid_integration import RealDataCollector

print("🚀 Starting FNSPID data download...")
collector = RealDataCollector()
real_data = collector.collect_all_data()

print(f"✅ Download complete!")
print(f"📊 Dataset shape: {real_data.shape}")
print(f"📅 Date range: {real_data['date'].min()} to {real_data['date'].max()}")
print(f"🏢 Symbols: {sorted(real_data['symbol'].unique())}")