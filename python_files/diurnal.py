import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Create output directory
output_dir = r"C:\Users\Karthik\Documents\Aditya_vs_code\Aditya_pa_work_on_diurnal\output"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Load CSV data (50 to 270 minutes)
# ============================================================
csv_path = r"C:\Users\Karthik\Documents\Aditya_vs_code\Aditya_pa_work_on_diurnal\data\csv_files\processed_temperature_data.csv"
csv_df = pd.read_csv(csv_path)

# Filter data from 50 to 270 minutes
csv_df = csv_df[(csv_df['time_minutes'] >= 50) & (csv_df['time_minutes'] <= 270)]
csv_df = csv_df.reset_index(drop=True)

# Calculate running mean with window size = 30
csv_df['running_mean'] = csv_df['ts'].rolling(window=30, center=True).mean()

# ============================================================
# Load mooring data
# ============================================================
cdf_path = r"C:\Users\Karthik\Documents\Aditya_vs_code\Aditya_pa_work_on_diurnal\data\nc_files\sst_2023_mooring.cdf"
ds = xr.open_dataset(cdf_path)

# Extract SST and time
sst = ds['T_25'].values.flatten()
time = pd.to_datetime(ds['time'].values)
lat = float(ds['lat'].values)
lon = float(ds['lon'].values)

# Create DataFrame
mooring_df = pd.DataFrame({'time': time, 'sst': sst})
mooring_df['month'] = mooring_df['time'].dt.month
mooring_df['date'] = mooring_df['time'].dt.date

# Get March and April data (2 consecutive days each from middle of month)
march_df = mooring_df[mooring_df['month'] == 3]
april_df = mooring_df[mooring_df['month'] == 4]

# March: pick 2 consecutive days from middle
march_dates = sorted(march_df['date'].unique())
march_mid = len(march_dates) // 2
march_date1 = march_dates[march_mid]
march_date2 = march_dates[march_mid + 1]
march_day1 = march_df[march_df['date'] == march_date1]['sst'].values
march_day2 = march_df[march_df['date'] == march_date2]['sst'].values
march_sst = np.concatenate([march_day1, march_day2])
# Calculate running mean with window size = 5 for mooring data
march_sst = pd.Series(march_sst).rolling(window=5,center=True).mean().values
march_hours = np.arange(len(march_sst))

# April: pick 2 consecutive days from middle
april_dates = sorted(april_df['date'].unique())
april_mid = len(april_dates) // 2
april_date1 = april_dates[april_mid]
april_date2 = april_dates[april_mid + 1]
april_day1 = april_df[april_df['date'] == april_date1]['sst'].values
april_day2 = april_df[april_df['date'] == april_date2]['sst'].values
april_sst = np.concatenate([april_day1, april_day2])
# Calculate running mean with window size = 5 for mooring data
april_sst = pd.Series(april_sst).rolling(window=5,center=True).mean().values
april_hours = np.arange(len(april_sst))

# ============================================================
# PLOT 1: CSV - Actual vs Running Mean
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(csv_df['time_minutes'], csv_df['ts'], 'b-', label='Actual', linewidth=1, alpha=0.7)
plt.plot(csv_df['time_minutes'], csv_df['running_mean'], 'r-', label='Running Mean (window=30)', linewidth=2)
plt.xlabel('Time (minutes)')
plt.ylabel('Temperature (°C)')
plt.title('Temperature vs Time: Actual vs Running Mean\n(50 to 270 minutes)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(50, 270)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot1_csv_actual_vs_running_mean.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot1_csv_actual_vs_running_mean.png")

# ============================================================
# PLOT 2: March and April Diurnal (Mooring Data)
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(march_hours, march_sst, 'b-', label=f'March ({march_date1} to {march_date2})', linewidth=1.5)
plt.plot(april_hours, april_sst, 'r-', label=f'April ({april_date1} to {april_date2})', linewidth=1.5)
plt.xlabel('Hour')
plt.ylabel('SST (°C)')
plt.title(f'Diurnal SST Variation - March vs April 2023\nLocation: {abs(lat)}°S, {lon}°E (RAMA Mooring)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 47)
plt.xticks(range(0, 48, 6))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot2_march_april_diurnal.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot2_march_april_diurnal.png")

# ============================================================
# PLOT 3: Normalized Comparison
# ============================================================

# Min-max normalization function
def min_max_normalize(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

# Normalize CSV data (time and temperature)
csv_time_norm = (csv_df['time_minutes'].values - csv_df['time_minutes'].min()) / (csv_df['time_minutes'].max() - csv_df['time_minutes'].min())
csv_temp_norm = min_max_normalize(csv_df['running_mean'].values)

# Normalize March data
march_time_norm = march_hours / (len(march_hours) - 1)
march_sst_norm = min_max_normalize(march_sst)

# Normalize April data
april_time_norm = april_hours / (len(april_hours) - 1)
april_sst_norm = min_max_normalize(april_sst)

plt.figure(figsize=(10, 5))
plt.plot(csv_time_norm, csv_temp_norm, 'g-', label='Experimnetal data', linewidth = 1.5)
plt.plot(march_time_norm, march_sst_norm, 'b-', label=f'March 2023',linewidth=1.5, alpha=0.8)
plt.plot(april_time_norm, april_sst_norm, 'r-', label=f'April 2023', linewidth=1.5, alpha=0.8)
plt.xlabel('Normalized Time (0-1)')
plt.ylabel('Normalized Temperature (0-1)')
plt.title('Normalized Comparison: experimental data vs observational data ')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot3_normalized_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot3_normalized_comparison.png")

# ============================================================
# PLOT 4: Overlay with Normalized Time, Actual Temperature
# ============================================================
plt.figure(figsize=(10, 5))
plt.plot(csv_time_norm, csv_df['running_mean'].values, 'g-', label='Experimental data', linewidth=1.5)
plt.plot(march_time_norm, march_sst, 'b-', label=f'March 2023', linewidth=1.5, alpha=0.8)
plt.plot(april_time_norm, april_sst, 'r-', label=f'April 2023', linewidth=1.5, alpha=0.8)
plt.xlabel('Normalized Time (0-1)')
plt.ylabel('Temperature (°C)')
plt.title('Comparison: Experimental vs Observational Data\n(Normalized Time, Actual Temperature)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'plot4_normalized_time_actual_temp.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: plot4_normalized_time_actual_temp.png")

print("\nDone! All 4 plots saved to output folder.")
