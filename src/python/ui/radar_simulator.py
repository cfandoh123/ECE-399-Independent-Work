import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# =========================================================
# 1. SETUP THE FIGURE & LAYOUT
# =========================================================
# Create a wide window with a dark theme (optional, but looks cool)
plt.style.use('dark_background')
fig = plt.figure(figsize=(14, 6))
plt.subplots_adjust(bottom=0.25, wspace=0.3)
fig.canvas.manager.set_window_title('Local Radar Spatial Simulator')

# Create the 3 subplots
ax_room = plt.subplot(1, 3, 1)
ax_range = plt.subplot(1, 3, 2)
ax_angle = plt.subplot(1, 3, 3)

# =========================================================
# 2. INITIALIZE THE DATA & MATH
# =========================================================
initial_x = 0.0  # Ball is dead center left/right
initial_y = 5.0  # Ball is 5 meters forward

range_axis = np.linspace(0, 15, 500)
angle_axis = np.linspace(-90, 90, 500)

def calculate_radar_data(x, y):
    """Calculates physical location and simulates the FFT peaks."""
    # Pythagoras for Range
    R = np.sqrt(x**2 + y**2)
    
    # Trigonometry for Angle (0 degrees is straight ahead at x=0)
    theta = np.degrees(np.arctan2(x, y))
    
    # Simulate the FFT peaks using Gaussian functions
    # Range peak is very sharp, Angle peak is wide (low resolution)
    range_signal = np.exp(-0.5 * ((range_axis - R) / 0.2)**2)
    angle_signal = np.exp(-0.5 * ((angle_axis - theta) / 8.0)**2)
    
    return R, theta, range_signal, angle_signal

R_init, theta_init, r_sig_init, a_sig_init = calculate_radar_data(initial_x, initial_y)

# =========================================================
# 3. DRAW THE INITIAL PLOTS
# =========================================================
# Plot 1: Top-Down Room View
ax_room.set_xlim(-10, 10)
ax_room.set_ylim(0, 15)
ax_room.set_title("Top-Down Room View")
ax_room.set_xlabel("X-Axis: Left/Right (m)")
ax_room.set_ylabel("Y-Axis: Forward Distance (m)")
ax_room.grid(True, linestyle='--', alpha=0.3)

radar_dot, = ax_room.plot(0, 0, 'o', color='#00ff00', markersize=10, label="Radar") 
ball_dot, = ax_room.plot(initial_x, initial_y, 'o', color='#00ffff', markersize=8, label="Target")
line_of_sight, = ax_room.plot([0, initial_x], [0, initial_y], '--', color='#00ffff', alpha=0.5)
ax_room.legend(loc='upper right')

# Plot 2: Range FFT Plot
ax_range.set_xlim(0, 15)
ax_range.set_ylim(0, 1.1)
ax_range.set_title("Simulated Range-FFT")
ax_range.set_xlabel("Distance (m)")
ax_range.set_ylabel("Amplitude")
ax_range.grid(True, alpha=0.2)
range_line, = ax_range.plot(range_axis, r_sig_init, color='#00ffff', lw=2)
range_text = ax_range.text(0.4, 0.9, f"Range: {R_init:.2f}m", transform=ax_range.transAxes, color='white')

# Plot 3: Angle FFT Plot
ax_angle.set_xlim(-90, 90)
ax_angle.set_ylim(0, 1.1)
ax_angle.set_title("Simulated Angle-FFT")
ax_angle.set_xlabel("Angle (Degrees)")
ax_angle.set_ylabel("Amplitude")
ax_angle.grid(True, alpha=0.2)
angle_line, = ax_angle.plot(angle_axis, a_sig_init, color='#ff00ff', lw=2)
angle_text = ax_angle.text(0.4, 0.9, f"Angle: {theta_init:.1f}°", transform=ax_angle.transAxes, color='white')

# =========================================================
# 4. ADD INTERACTIVE SLIDERS
# =========================================================
# Define where the sliders sit on the window [left, bottom, width, height]
ax_slider_x = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_slider_y = plt.axes([0.15, 0.05, 0.65, 0.03])

slider_x = Slider(ax_slider_x, 'Move Left/Right (X)', -10.0, 10.0, valinit=initial_x, color='#00ffff')
slider_y = Slider(ax_slider_y, 'Move Forward (Y)', 0.1, 15.0, valinit=initial_y, color='#ff00ff')

# =========================================================
# 5. THE UPDATE FUNCTION (Runs when a slider moves)
# =========================================================
def update(val):
    x = slider_x.val
    y = slider_y.val
    
    # Recalculate the math
    R, theta, r_sig, a_sig = calculate_radar_data(x, y)
    
    # Update the Room View graphics
    ball_dot.set_data([x], [y])
    line_of_sight.set_data([0, x], [0, y])
    
    # Update the Range Plot graphics
    range_line.set_ydata(r_sig)
    range_text.set_text(f"Range: {R:.2f}m")
    
    # Update the Angle Plot graphics
    angle_line.set_ydata(a_sig)
    angle_text.set_text(f"Angle: {theta:.1f}°")
    
    # Redraw the canvas
    fig.canvas.draw_idle()

# Link the sliders to the update function
slider_x.on_changed(update)
slider_y.on_changed(update)

# Launch the interactive window!
plt.show()