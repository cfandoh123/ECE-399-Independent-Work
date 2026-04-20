%% =========================================================
%  FMCW Radar Simulation Basics
%  Range FFT, Doppler FFT, and Angle FFT of a Single Object
%
%  Project:  Radar Object Detection + ML (Junior IW)
%% =========================================================

clear; clc; close all;

%% =========================================================
%  SECTION 1: RADAR PARAMETERS
%% =========================================================

c       = 3e8;          % Speed of light (m/s)
fc      = 77e9;         % Carrier frequency (77 GHz)
lambda  = c / fc;       % Wavelength (m)

B       = 4e9;          % Chirp bandwidth (4 GHz)
Tc      = 40e-6;        % Chirp duration (40 us)
S       = B / Tc;       % Chirp slope (Hz/s)

Fs      = 4e6;          % ADC sampling rate (Hz)
N_s     = Fs * Tc;      % Number of ADC samples per chirp

N_chirps = 128;         % Number of chirps per frame
N_ant    = 4;           % Number of RX antennas

D        = lambda / 2;  % Antenna spacing (half-wavelength)

%% --- Derived performance metrics ---
d_res    = c / (2 * B);  
d_max    = (Fs * c) / (2 * S);
v_max    = lambda / (4 * Tc);
v_res    = lambda / (2 * N_chirps * Tc);
aoa_res  = rad2deg(2 / N_ant);   % at theta=0, D=lambda/2 => res=2/N (radians)

fprintf('===== Radar Performance Summary =====\n');
fprintf('Range resolution   : %.4f m (%.2f cm)\n', d_res, d_res*100);
fprintf('Max range          : %.2f m\n', d_max);
fprintf('Max velocity       : %.2f m/s\n', v_max);
fprintf('Velocity resolution: %.4f m/s\n', v_res);
fprintf('AoA resolution     : %.2f deg (at boresight)\n', aoa_res);
fprintf('=====================================\n\n');

%% =========================================================
%  SECTION 2: SINGLE OBJECT DEFINITION
%% =========================================================

% Object parameters — these can be changed to simulate different scenarios
obj_range    = 4;      % Object range (meters)
obj_velocity = 3;       % Object radial velocity (m/s), positive = moving away, negative = moving towards
obj_angle    = 20;      % Object angle of arrival (degrees)

fprintf('===== Object Parameters =====\n');
fprintf('Range    : %.1f m\n', obj_range);
fprintf('Velocity : %.1f m/s\n', obj_velocity);
fprintf('Angle    : %.1f deg\n', obj_angle);
fprintf('=============================\n\n');

%% =========================================================
%  SECTION 3: IF SIGNAL GENERATION
%
%  For each chirp k and each RX antenna n, we generate the
%  IF signal: A * exp(j * 2*pi*(f_beat*t + phi_doppler + phi_angle))
%
%  Where:
%   f_beat       = S * 2*d/c          (range encodes as frequency)
%   phi_doppler  = (4*pi/lambda) * v * k * Tc  (velocity encodes as phase across chirps)
%   phi_angle    = (2*pi/lambda) * D * n * sin(theta)  (AoA encodes as phase across antennas)
%% =========================================================

t = (0 : N_s-1) / Fs;    % Fast-time axis (within one chirp)

% Pre-allocate IF data cube: [samples x chirps x antennas]
IF_cube = zeros(N_s, N_chirps, N_ant);

f_beat  = S * (2 * obj_range) / c;                    % Beat frequency from range
phi0    = (4 * pi * fc * obj_range) / c;               % Initial phase from range

for n = 1 : N_ant
    phi_angle = (2*pi/lambda) * D * (n-1) * sind(obj_angle);  % AoA phase per antenna

    for k = 1 : N_chirps
        % Doppler phase accumulates chirp-to-chirp
        phi_doppler = (4*pi/lambda) * obj_velocity * (k-1) * Tc;

        % Total phase of IF signal for this chirp & antenna
        total_phase = 2*pi*f_beat*t + phi0 + phi_doppler + phi_angle;

        % Add small Gaussian noise to simulate real conditions
        signal_power = 1;  % since signal amplitude is 1
        SNR_dB = 0.001;       % desired SNR in dB — lower this to increase noise
        SNR_linear = 10^(SNR_dB / 10);
        noise_std = sqrt(signal_power / (2 * SNR_linear));  % factor of 2 for complex noise

        noise = noise_std * (randn(1, N_s) + 1j*randn(1, N_s));

        IF_cube(:, k, n) = exp(1j * total_phase) + noise;
    end
end

fprintf('IF data cube generated: [%d samples x %d chirps x %d antennas]\n\n', ...
        N_s, N_chirps, N_ant);

%% =========================================================
%  SECTION 4: RANGE FFT  (FFT along fast-time / samples axis)
%
%  Each row of IF_cube is one chirp's time-domain samples.
%  FFT across samples -> frequency axis -> range axis.
%  Peak location in frequency maps directly to object range.
%% =========================================================

NFFT_range = 512;   % Zero-pad for finer frequency resolution

win = hanning(N_s);
win = reshape(win, [N_s, 1, 1]);
IF_cube = IF_cube .* win; 

% Take Range FFT over first dimension (samples), for all chirps & antennas
range_fft = fft(IF_cube, NFFT_range, 1);

% Build range axis
f_axis     = (0 : NFFT_range-1) * (Fs / NFFT_range);   % Frequency bins (Hz)
range_axis = (f_axis * c) / (2 * S);                     % Convert freq -> range (m)

% Use only first half (one-sided spectrum)
half       = 1 : NFFT_range;
range_axis = range_axis(half);
range_fft_plot = range_fft(half, :, :);

% Average across chirps & antennas for a clean single plot
range_profile = mean(abs(range_fft_plot), [2 3]);

figure('Name', 'Range FFT', 'NumberTitle', 'off', 'Color', 'w');
plot(range_axis, 20*log10(range_profile / max(range_profile)), 'b', 'LineWidth', 1.8);
xlabel('Range (m)'); ylabel('Normalized Magnitude (dB)');
title(sprintf('Range FFT — Object at %.1f m', obj_range));
xline(obj_range, 'r--', 'LineWidth', 1.5, 'Label', sprintf('Truth: %.1fm', obj_range));
grid on; xlim([0 d_max]);
annotation('textbox', [0.55 0.75 0.35 0.15], 'String', ...
    sprintf('d_{res} = %.2f cm\nd_{max} = %.1f m', d_res*100, d_max), ...
    'FitBoxToText','on', 'BackgroundColor','yellow', 'FontSize', 9);

%% =========================================================
%  SECTION 5: DOPPLER FFT  (FFT along slow-time / chirp axis)
%
%  For a fixed range bin, phase shifts across consecutive chirps
%  encode the radial velocity of the object.
%  FFT across chirps -> Doppler frequency -> velocity axis.
%% =========================================================

% Hanning window along the chirp dimension (dim 2)
win_doppler = hanning(N_chirps);          % [128 x 1] column vector

% Reshape to [1 x N_chirps x 1] so it broadcasts across
% range bins (dim 1) and antennas (dim 3) automatically
win_doppler = reshape(win_doppler, [1, N_chirps, 1]);

% Apply window — multiplies each chirp sample by the window value
range_fft_windowed = range_fft .* win_doppler;
% ------------------------------------------

NFFT_doppler = 256;  % Zero-pad for finer velocity resolution

% Take Doppler FFT over second dimension (chirps)
doppler_fft = fft(range_fft_windowed, NFFT_doppler, 2);

% fftshift to center zero-velocity
doppler_fft = fftshift(doppler_fft, 2);

% Build velocity axis
omega_axis   = linspace(-pi, pi, NFFT_doppler);        % Phase per chirp
vel_axis     = (lambda * omega_axis) / (4 * pi * Tc);  % Convert to m/s

% Build Range-Doppler map: magnitude averaged across antennas
range_doppler_map = mean(abs(doppler_fft(half, :, :)), 3);

figure('Name', 'Range-Doppler Map', 'NumberTitle', 'off', 'Color', 'w');
imagesc(vel_axis, range_axis, 20*log10(range_doppler_map / max(range_doppler_map(:))));
colormap('jet'); colorbar;
xlabel('Velocity (m/s)'); ylabel('Range (m)');
title('Range-Doppler Heatmap');
hold on;
plot(obj_velocity, obj_range, 'w+', 'MarkerSize', 14, 'LineWidth', 2.5);
legend(sprintf('Truth: (%.1f m/s, %.1f m)', obj_velocity, obj_range), ...
       'Location','northeast', 'TextColor','w');
set(gca, 'YDir', 'normal');
clim([-40 0]);   % dB color scale
annotation('textbox', [0.55 0.02 0.4 0.1], 'String', ...
    sprintf('v_{max} = %.2f m/s | v_{res} = %.4f m/s', v_max, v_res), ...
    'FitBoxToText','on','BackgroundColor','yellow','FontSize',8);

%% =========================================================
%  SECTION 6: ANGLE FFT  (FFT along spatial / antenna axis)
%
%  For a fixed range-Doppler bin, phase shifts across antennas
%  encode the angle of arrival of the object.
%  FFT across antennas -> spatial frequency -> angle axis.
%% =========================================================

NFFT_angle = 256;   % Zero-pad heavily — we only have 4 physical antennas

% Find the peak range bin index (closest to true object range)
[~, range_bin] = min(abs(range_axis - obj_range));

% Find the peak Doppler bin index
[~, doppler_bin] = min(abs(vel_axis - obj_velocity));

% Extract the steering vector across antennas at the peak range-Doppler bin
steering_vec = squeeze(doppler_fft(range_bin, doppler_bin, :));  % [N_ant x 1]

% Take FFT across antennas (spatial FFT)
angle_fft = fftshift(fft(steering_vec, NFFT_angle));

% Build sin(theta) axis, then convert to degrees
omega_angle = linspace(-pi, pi, NFFT_angle);
sin_theta   = (lambda * omega_angle) / (2 * pi * D);  % sin(theta)

% Clip to valid range [-1, 1] to avoid asin errors
valid       = abs(sin_theta) <= 1;
angle_axis  = zeros(1, NFFT_angle);
angle_axis(valid)  = rad2deg(asin(sin_theta(valid)));
angle_axis(~valid) = NaN;

angle_profile = abs(angle_fft);

figure('Name', 'Angle FFT', 'NumberTitle', 'off', 'Color', 'w');
plot(angle_axis, 20*log10(angle_profile / max(angle_profile)), ...
     'm', 'LineWidth', 1.8);
xlabel('Angle of Arrival (degrees)'); ylabel('Normalized Magnitude (dB)');
title(sprintf('Angle FFT — Object at %.1f°', obj_angle));
xline(obj_angle, 'r--', 'LineWidth', 1.5, 'Label', sprintf('Truth: %.1f°', obj_angle));
grid on; xlim([-90 90]);
annotation('textbox', [0.13 0.15 0.38 0.12], 'String', ...
    sprintf('N_{ant} = %d | AoA res ≈ %.1f° (boresight)\nFoV = ±90° (D = λ/2)', ...
    N_ant, aoa_res), ...
    'FitBoxToText','on','BackgroundColor','yellow','FontSize',9);

%% =========================================================
%  SECTION 7: SUMMARY FIGURE — All 3 FFTs in one view
%% =========================================================

figure('Name', 'Full Radar Processing Pipeline', 'NumberTitle', 'off', ...
       'Color', 'w', 'Position', [100 100 1200 400]);

% --- Range ---
subplot(1,3,1);
plot(range_axis, 20*log10(range_profile / max(range_profile)), 'b', 'LineWidth', 1.5);
xlabel('Range (m)'); ylabel('Magnitude (dB)');
title('1) Range FFT');
xline(obj_range, 'r--', 'LineWidth', 1.2);
grid on; xlim([0 min(d_max, obj_range*3)]);
ylim([-40 2]);

% --- Range-Doppler ---
subplot(1,3,2);
imagesc(vel_axis, range_axis, 20*log10(range_doppler_map / max(range_doppler_map(:))));
colormap('jet'); colorbar;
xlabel('Velocity (m/s)'); ylabel('Range (m)');
title('2) Range-Doppler Heatmap');
hold on;
plot(obj_velocity, obj_range, 'w+', 'MarkerSize', 12, 'LineWidth', 2);
set(gca, 'YDir', 'normal'); clim([-40 0]);

% --- Angle ---
subplot(1,3,3);
plot(angle_axis, 20*log10(angle_profile / max(angle_profile)), 'm', 'LineWidth', 1.5);
xlabel('Angle (degrees)'); ylabel('Magnitude (dB)');
title('3) Angle FFT');
xline(obj_angle, 'r--', 'LineWidth', 1.2);
grid on; xlim([-90 90]); ylim([-40 2]);

sgtitle(sprintf('FMCW Radar Pipeline | Object: %.1fm, %.1fm/s, %.1f°', ...
        obj_range, obj_velocity, obj_angle), 'FontSize', 13, 'FontWeight', 'bold');

fprintf('Done! All plots generated.\n');
fprintf('Try changing obj_range, obj_velocity, obj_angle in Section 2\n');
fprintf('to see how each FFT responds.\n');
