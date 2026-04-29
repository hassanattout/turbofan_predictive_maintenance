import pandas as pd

# Load test data
# Adjust this path if your test files are in CMAPSSData/
test_file = "CMAPSSData/test_FD001.txt"  # you can loop through all FD001–FD004 if needed
rul_file = "CMAPSSData/RUL_FD001.txt"

# Column names based on NASA C-MAPSS format
columns = ['engine_id', 'cycle', 
           'setting_1', 'setting_2', 'setting_3'] + [f'sensor_{i}' for i in range(1,22)]

# Read test data
data = pd.read_csv(test_file, sep='\s+', header=None, names=columns)

# Read RUL file
rul = pd.read_csv(rul_file, sep='\s+', header=None, names=['RUL'])

# Each engine has multiple cycles, RUL is counted backwards
# Join RUL to last cycle per engine
data = data.sort_values(['engine_id','cycle']).copy()
engine_ids = data['engine_id'].unique()
rul_list = []

for eid in engine_ids:
    engine_data = data[data['engine_id']==eid].copy()
    engine_rul = rul.loc[eid-1,'RUL']  # index 0-based
    cycles = len(engine_data)
    # Compute RUL per cycle
    engine_data['RUL'] = [engine_rul + cycles - i for i in range(1, cycles+1)]
    rul_list.append(engine_data)

# Combine all engines
full_data = pd.concat(rul_list)

# Save full CSV with all sensors + settings + RUL
output_file = "test_engine_with_rul_full.csv"
full_data.to_csv(output_file, index=False)
print(f"Full test CSV created: {output_file}")
