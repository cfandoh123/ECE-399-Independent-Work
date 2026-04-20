%% =========================================================
%  FMCW Radar Simulation 
%  Multi-Object + Moving Object Scene
%
%  Key upgrade from Milestone 1:
%  - Object range now updates chirp-by-chirp: d(k) = d0 + v*k*Tc
%  - Multiple objects with different range, velocity, angle
%  - Range-Doppler map now shows physically correct coupling
%
%  Project:  Radar Object Detection + ML (Junior IW)
%% =========================================================

clear; clc; close all;

%% =========================================================
%  SECTION 1: RADAR PARAMETERS  (same as Milestone 1)
%% =========================================================

c        = 3e8;
fc       = 77e9;
lambda   = c / fc;

B        = 4e9;
Tc       = 40e-6;
S        = B / Tc;

Fs       = 4e6;
N_s      = round(Fs * Tc);   % samples per chirp

N_chirps = 128;
N_ant    = 4;
D        = lambda / 2;

% Performance metrics
d_res  = c / (2 * B);
d_max  = (Fs * c) / (2 * S);
v_max  = lambda / (4 * Tc);
v_res  = lambda / (2 * N_chirps * Tc);

fprintf('===== Radar Performance =====\n');
fprintf('Range res : %.2f cm\n', d_res*100);
fprintf('Max range : %.2f m\n',  d_max);
fprintf('Max vel   : %.2f m/s\n', v_max);
fprintf('Vel res   : %.4f m/s\n', v_res);
fprintf('=============================\n\n');

%% =========================================================
%  SECTION 2: SCENE DEFINITION — Multiple Moving Objects
%
%  Each object is defined by:
%    [range (m), velocity (m/s), angle (deg), RCS amplitude]
%
%  Positive velocity = moving away from radar
%  Negative velocity = moving toward radar
%
%  Think of this as a simple room scene:
%    Object 1 — Moving toward radar  (strong reflector)
%    Object 2 — Still         (medium reflector)
%    Object 3 — Moving away from radar        (strong reflector)
%% =========================================================

%         range  velocity  angle   amplitude
objects = [1.5,   -2.0,    -20,    1.0;   % walking toward, left
           3.0,    0.0,     0,     0.7;   % stationary,     center
           5.0,    3.0,     30,    0.9];  % walking away,   right

N_obj = size(objects, 1);

fprintf('===== Scene: %d Objects =====\n', N_obj);
for i = 1:N_obj
    fprintf('  Obj %d | range=%.1fm  vel=%.1fm/s  angle=%.1f°  amp=%.1f\n', ...
        i, objects(i,1), objects(i,2), objects(i,3), objects(i,4));
end
fprintf('==============================\n\n');

%% =========================================================
%  SECTION 3: IF SIGNAL GENERATION — Time-Varying Range
%
%  KEY UPGRADE from Milestone 1:
%  Range is no longer fixed. For chirp k, object i is at:
%
%       d_i(k) = d0_i + v_i * k * Tc
%
%  This means:
%  - The beat frequency slowly drifts across chirps (range migration)
%  - The Doppler phase accumulates correctly from real physics
%  - Range and velocity are now physically coupled
%
%  The IF signal for object i at chirp k, antenna n:
%
%    s = A * exp(j * [2π*f_beat(k)*t  +  φ_range(k)  +  φ_doppler(k)  +  φ_angle(n)])
%
%  where all phase terms use the instantaneous range d_i(k)
%% =========================================================

t = (0 : N_s-1).' / Fs;   % Column vector — fast time axis

% Pre-allocate IF data cube [samples x chirps x antennas]
IF_cube = zeros(N_s, N_chirps, N_ant);

for i = 1 : N_obj
    d0    = objects(i, 1);   % Initial range
    v     = objects(i, 2);   % Radial velocity
    theta = objects(i, 3);   % Angle of arrival (degrees)
    A     = objects(i, 4);   % Amplitude (RCS proxy)

    phi_angle_base = (2*pi/lambda) * D * sind(theta);  % Phase per antenna

    for k = 1 : N_chirps
        % --- Time-varying range (the key upgrade) ---
        d_k = d0 + v * (k-1) * Tc;

        % Beat frequency for this chirp (range encodes as frequency)
        f_beat = S * (2 * d_k) / c;

        % Initial phase of IF signal (from round-trip path length)
        phi_range = (4 * pi * fc * d_k) / c;

        for n = 1 : N_ant
            phi_angle = phi_angle_base * (n-1);

            % Full IF signal for this object, chirp, antenna
            signal = A * exp(1j * (2*pi*f_beat*t + phi_range + phi_angle));

            IF_cube(:, k, n) = IF_cube(:, k, n) + signal;
        end
    end
end

% Add complex Gaussian noise (SNR ~ 20 dB)
signal_power = mean(abs(IF_cube(:)).^2);  % measure actual signal power
SNR_dB = 5;                              % lower this to increase noise
SNR_linear = 10^(SNR_dB / 10);
noise_std = sqrt(signal_power / (2 * SNR_linear));

IF_cube = IF_cube + noise_std * (randn(size(IF_cube)) + 1j*randn(size(IF_cube)));

fprintf('IF cube ready: [%d samples × %d chirps × %d antennas]\n\n', ...
        N_s, N_chirps, N_ant);

%% =========================================================
%  SECTION 4: RANGE FFT
%% =========================================================

NFFT_r    = 512;

% Hanning window along fast-time (samples) dimension
win_range   = hanning(N_s);
win_range   = reshape(win_range, [N_s, 1, 1]);  % broadcast across chirps & antennas
IF_windowed = IF_cube .* win_range;

range_fft = fft(IF_windowed, NFFT_r, 1);

% Full spectrum — complex IF signal uses all NFFT_r bins (not half)
half       = 1 : NFFT_r;
f_axis     = (0:NFFT_r-1) * (Fs/NFFT_r);
range_axis = (f_axis * c) / (2*S);
range_axis = range_axis(half);

range_fft_half = range_fft(half, :, :);

% Range profile: average magnitude across all chirps & antennas
range_profile = mean(abs(range_fft_half), [2 3]);

figure('Name','Range FFT — Multi-Object','NumberTitle','off','Color','w');
plot(range_axis, 20*log10(range_profile/max(range_profile)), 'b', 'LineWidth', 1.8);
hold on;
colors = {'r','g','m'};
for i = 1:N_obj
    xline(objects(i,1), '--', 'Color', colors{i}, 'LineWidth', 1.4, ...
          'Label', sprintf('Obj%d: %.0fm', i, objects(i,1)), ...
          'LabelVerticalAlignment','bottom');
end
xlabel('Range (m)'); ylabel('Magnitude (dB)');
title('Range FFT — Multiple Objects');
grid on; xlim([0 d_max]); ylim([-40 2]);
legend('Range Profile', 'Location','northeast');

%% =========================================================
%  SECTION 5: RANGE-DOPPLER MAP
%% =========================================================

NFFT_d      = 256;

% Hanning window along slow-time (chirps) dimension
win_doppler = hanning(N_chirps);
win_doppler = reshape(win_doppler, [1, N_chirps, 1]);  % broadcast across range & antennas
range_fft_windowed = range_fft_half .* win_doppler;

doppler_fft = fft(range_fft_windowed, NFFT_d, 2);
doppler_fft = fftshift(doppler_fft, 2);

omega_axis = linspace(-pi, pi, NFFT_d);
vel_axis   = (lambda * omega_axis) / (4*pi*Tc);

% Average magnitude across antennas
RD_map = mean(abs(doppler_fft), 3);

figure('Name','Range-Doppler Map — Multi-Object','NumberTitle','off','Color','w');
imagesc(vel_axis, range_axis, 20*log10(RD_map / max(RD_map(:))));
colormap('jet'); colorbar;
set(gca, 'YDir', 'normal');
clim([-40 0]);
xlabel('Velocity (m/s)'); ylabel('Range (m)');
title('Range-Doppler Heatmap — Multiple Objects');
hold on;
for i = 1:N_obj
    plot(objects(i,2), objects(i,1), 'w+', 'MarkerSize', 14, 'LineWidth', 2.5);
    text(objects(i,2)+0.15, objects(i,1)+0.5, sprintf('Obj%d',i), ...
         'Color','white','FontSize',9,'FontWeight','bold');
end
legend('Object truth positions','Location','northeast','TextColor','w');

%% =========================================================
%  SECTION 6: ANGLE FFT — Per Object
%
%  For each object, we extract the range-Doppler bin closest
%  to the object's true (range, velocity) and run the angle
%  FFT to recover its AoA.
%% =========================================================

NFFT_a   = 512;
fig_aoa  = figure('Name','Angle FFT — Per Object','NumberTitle','off','Color','w');

for i = 1:N_obj
    % Find the closest range and Doppler bin for this object
    [~, rb] = min(abs(range_axis - objects(i,1)));
    [~, db] = min(abs(vel_axis   - objects(i,2)));

    % Steering vector across antennas
    steer = squeeze(doppler_fft(rb, db, :));

    % Spatial FFT
    angle_fft_i = fftshift(fft(steer, NFFT_a));

    omega_a  = linspace(-pi, pi, NFFT_a);
    sin_th   = (lambda * omega_a) / (2*pi*D);
    valid    = abs(sin_th) <= 1;
    ang_axis = NaN(1, NFFT_a);
    ang_axis(valid) = rad2deg(asin(sin_th(valid)));

    subplot(1, N_obj, i);
    plot(ang_axis, 20*log10(abs(angle_fft_i)/max(abs(angle_fft_i))), ...
         colors{i}, 'LineWidth', 1.8);
    hold on;
    xline(objects(i,3), 'k--', 'LineWidth', 1.4, ...
          'Label', sprintf('Truth: %.0f°', objects(i,3)));
    xlabel('Angle (degrees)'); ylabel('Magnitude (dB)');
    title(sprintf('Obj %d | v=%.1fm/s, d=%.0fm', i, objects(i,2), objects(i,1)));
    grid on; xlim([-90 90]); ylim([-40 2]);
end
sgtitle('Angle FFT — One Per Object', 'FontSize', 12, 'FontWeight', 'bold');

%% =========================================================
%  SECTION 7: RANGE MIGRATION VISUALIZATION
%
%  This plot shows WHY the moving-range physics matters.
%  It displays the raw IF spectrogram for one antenna,
%  showing how the beat frequency (= range) drifts across
%  chirps for a moving object vs. a stationary one.
% %% =========================================================
% 
% figure('Name','Range Migration — IF Spectrogram','NumberTitle','off','Color','w');
% 
% % Compute short-time magnitude of range FFT across chirps (antenna 1)
% range_fft_ant1 = abs(range_fft_half(:, :, 1));   % [range_bins x chirps]
% 
% imagesc(1:N_chirps, range_axis, 20*log10(range_fft_ant1 / max(range_fft_ant1(:))));
% colormap('hot'); colorbar;
% set(gca, 'YDir', 'normal');
% clim([-30 0]);
% xlabel('Chirp Index'); ylabel('Range (m)');
% title('Range Migration: Beat Frequency Drift Across Chirps');
% 
% % Overlay theoretical range trajectory for each moving object
% hold on;
% for i = 1:N_obj
%     k_axis  = 0 : N_chirps-1;
%     d_traj  = objects(i,1) + objects(i,2) * k_axis * Tc;
%     plot(k_axis+1, d_traj, '--', 'Color', colors{i}, 'LineWidth', 2.0, ...
%          'DisplayName', sprintf('Obj%d trajectory (v=%.1fm/s)', i, objects(i,2)));
% end
% legend('Location','northeast','TextColor','k','FontSize',8);

%% =========================================================
%  SECTION 8: SUMMARY CONSOLE OUTPUT
%% =========================================================

fprintf('===== Detection Summary =====\n');
fprintf('Objects should appear at:\n');
for i = 1:N_obj
    fprintf('  Obj %d → Range: %.1fm | Vel: %.1fm/s | Angle: %.1f°\n', ...
        i, objects(i,1), objects(i,2), objects(i,3));
end
fprintf('\nTry these experiments:\n');
fprintf('  1. Set two objects to same range → see them split in Doppler\n');
fprintf('  2. Set two objects to same velocity → see them split in Range\n');
fprintf('  3. Set two objects to same range+velocity → only Angle FFT separates them\n');
fprintf('  4. Push velocity beyond v_max (%.2fm/s) → observe aliasing\n', v_max);
fprintf('=============================\n');
