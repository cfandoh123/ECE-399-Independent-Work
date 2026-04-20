import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# =========================================================
# 1. SET UP THE WEB PAGE
# =========================================================
st.set_page_config(page_title="Radar DSP Simulator", layout="wide")
st.title("📡 Full Radar DSP Simulator")
st.markdown("Adjust the target's physical position and velocity in the sidebar to see how the Digital Signal Processing (DSP) pipeline responds.")

# =========================================================
# 2. CREATE NATIVE WEB SLIDERS (SIDEBAR)
# =========================================================
st.sidebar.header("Target Controls")
x = st.sidebar.slider("Left/Right Position (X) [m]", min_value=-10.0, max_value=10.0, value=2.0, step=0.1)
y = st.sidebar.slider("Forward Distance (Y) [m]", min_value=0.1, max_value=15.0, value=5.0, step=0.1)
v = st.sidebar.slider("Velocity [m/s]", min_value=-5.0, max_value=5.0, value=-1.5, step=0.1)

# =========================================================
# 3. CALCULATE THE RADAR MATH
# =========================================================
range_axis = np.linspace(0, 15, 200)
angle_axis = np.linspace(-90, 90, 200)
vel_axis = np.linspace(-5, 5, 200)
V_grid, R_grid = np.meshgrid(vel_axis, range_axis)

# Physics
R = np.sqrt(x**2 + y**2)
theta = np.degrees(np.arctan2(x, y))

# Simulated FFT Signals
range_signal = np.exp(-0.5 * ((range_axis - R) / 0.2)**2)
angle_signal = np.exp(-0.5 * ((angle_axis - theta) / 8.0)**2)
heatmap = np.exp(-0.5 * ((V_grid - v)/0.4)**2 - 0.5 * ((R_grid - R)/0.4)**2)

# =========================================================
# 4. DRAW THE GRAPHS (Standard Matplotlib)
# =========================================================
plt.style.use('dark_background')
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)

# Plot 1: Room
ax_room = fig.add_subplot(gs[0, 0])
ax_room.set_xlim(-10, 10); ax_room.set_ylim(0, 15)
ax_room.set_title("1. Physical Room Setup")
ax_room.grid(True, linestyle='--', alpha=0.3)
ax_room.plot(0, 0, 'o', color='#00ff00', markersize=10, label="Radar") 
ax_room.plot(x, y, 'o', color='#00ffff', markersize=8, label="Target")
ax_room.plot([0, x], [0, y], '--', color='#00ffff', alpha=0.5)

# Plot 2: Heatmap
ax_heatmap = fig.add_subplot(gs[0, 1])
ax_heatmap.set_title("2. Range-Doppler Heatmap (2D-FFT)")
ax_heatmap.pcolormesh(V_grid, R_grid, heatmap, shading='auto', cmap='jet', vmin=0, vmax=1)
ax_heatmap.axvline(0, color='white', linestyle=':', alpha=0.5)

# Plot 3: Range FFT
ax_range = fig.add_subplot(gs[1, 0])
ax_range.set_xlim(0, 15); ax_range.set_ylim(0, 1.1)
ax_range.set_title(f"3. Range-FFT (Distance: {R:.2f}m)")
ax_range.grid(True, alpha=0.2)
ax_range.plot(range_axis, range_signal, color='#00ffff', lw=2)

# Plot 4: Angle FFT
ax_angle = fig.add_subplot(gs[1, 1])
ax_angle.set_xlim(-90, 90); ax_angle.set_ylim(0, 1.1)
ax_angle.set_title(f"4. Angle-FFT (Angle: {theta:.1f}°)")
ax_angle.grid(True, alpha=0.2)
ax_angle.plot(angle_axis, angle_signal, color='#ff00ff', lw=2)

# =========================================================
# 5. RENDER TO THE BROWSER
# =========================================================
# This single command tells Streamlit to draw the figure!
st.pyplot(fig)