%% =========================================================
%  process_real_data.m
%  Convert Real Radar IF Data → Range-Angle Heatmap
%  → Save for Python ML inference
%
%  Usage:
%    Option A: Load raw IF cube from TI radar SDK
%    Option B: Load from saved .mat file
%    Option C: Simulate with added realism (no hardware yet)
%
%  Output:
%    real_heatmap.mat  — same format as training data
%                        ready for radar_multitask_cnn.py
%
%  Project: Radar Object Detection + ML (Junior IW)
%% =========================================================

clear; clc; close all;

%% =========================================================
%  SECTION 1: RADAR PARAMETERS
%  MUST match radar_shape_classification.m exactly
%% =========================================================

c        = 3e8;
fc       = 77e9;
lambda   = c / fc;

B        = 4e9;
Tc       = 40e-6;
S        = B / Tc;

Fs       = 4e6;
N_s      = round(Fs * Tc);

N_chirps = 128;
N_ant    = 4;
D        = lambda / 2;

d_max    = (Fs*c) / (2*S);

fprintf('===== Radar Parameters =====\n');
fprintf('Max range  : %.2f m\n', d_max);
fprintf('Wavelength : %.2f mm\n', lambda*1000);
fprintf('============================\n\n');

%% =========================================================
%  SECTION 2: LOAD IF DATA
%
%  Choose ONE of the three options below.
%  Comment out the others.
%% =========================================================

DATA_SOURCE = 'simulate';   % 'ti_radar' | 'mat_file' | 'simulate'

switch DATA_SOURCE

    case 'ti_radar'
        %% ── Option A: TI mmWave Radar (Real Hardware) ────
        % Requires TI mmWave SDK and MATLAB support package
        % Uncomment and fill in your device port

        % port = 'COM3';   % Windows example — check Device Manager
        % port = '/dev/ttyUSB0';   % Linux example

        % cfg.numSamples = N_s;
        % cfg.numChirps  = N_chirps;
        % cfg.numRx      = N_ant;
        % cfg.bandwidth  = B;
        % cfg.startFreq  = fc;
        % cfg.chirpTime  = Tc;

        % Connect and collect one frame
        % radarObj  = mmWaveDevice(port, cfg);
        % IF_cube   = radarObj.captureFrame();
        % delete(radarObj);

        % If using TI Demo binary output:
        % IF_cube = parse_ti_bin('adc_data.bin', N_s, N_chirps, N_ant);

        fprintf('TI radar option selected — uncomment hardware code\n');
        IF_cube = zeros(N_s, N_chirps, N_ant);   % placeholder

    case 'mat_file'
        %% ── Option B: Load Previously Saved .mat File ────
        % Use this if you captured data earlier and saved it

        filename = 'captured_radar_data.mat';
        fprintf('Loading IF cube from: %s\n', filename);

        loaded  = load(filename);

        % Try common field names from TI SDK output
        if isfield(loaded, 'IF_cube')
            IF_cube = loaded.IF_cube;
        elseif isfield(loaded, 'adcData')
            IF_cube = loaded.adcData;
        elseif isfield(loaded, 'rawData')
            IF_cube = loaded.rawData;
        else
            error(['Cannot find IF data in .mat file. ' ...
                   'Available fields: %s'], ...
                   strjoin(fieldnames(loaded), ', '));
        end

        fprintf('Loaded IF cube: [%d × %d × %d]\n', size(IF_cube));

    case 'simulate'
        %% ── Option C: Realistic Simulation (No Hardware) ──
        % Simulates a real object with hardware imperfections.
        % Use this to test the full pipeline before having hardware.

        fprintf('Simulating realistic radar return...\n');

        % Define test object — change this to simulate different objects
        % Try matching one of your training shapes:
        %   Circle:    radius=0.25m, 20 uniform points, amp=0.35
        %   Square:    0.5m side, 4 corners amp=1.0, edges amp=0.6
        %   Rectangle: 0.7×0.3m, same corner/edge structure
        %   Triangle:  0.5m side equilateral, 3 corners + edges

        TEST_OBJECT = 'square';   % change to test different shapes

        switch TEST_OBJECT
            case 'circle'
                angles   = linspace(0, 2*pi, 21); angles = angles(1:end-1);
                obj_sc   = [0.25*cos(angles(:)), 0.25*sin(angles(:)), ...
                            0.35*ones(20,1)];
            case 'square'
                h = 0.25;
                obj_sc = [-h,-h,1.0; h,-h,1.0; h,h,1.0; -h,h,1.0;
                           0,-h,0.6;  0, h,0.6; -h,0,0.6; h,0,0.6];
            case 'rectangle'
                hw=0.35; hh=0.15;
                obj_sc = [-hw,-hh,1.0; hw,-hh,1.0; hw,hh,1.0; -hw,hh,1.0;
                            0,-hh,0.6;  0,  hh,0.6;
                          -hw,   0,0.6; hw,   0,0.6];
            case 'triangle'
                s=0.5; ht=s*sqrt(3)/2;
                v1=[0, 2*ht/3]; v2=[-s/2,-ht/3]; v3=[s/2,-ht/3];
                obj_sc = [v1,1.0; v2,1.0; v3,1.0];
        end

        % Place object at 4m range, 0° angle
        obj_range = 4.0;
        obj_angle = 0.0;
        rx = obj_range * sind(obj_angle);
        ry = obj_range * cosd(obj_angle);

        % Translate scatterers to world coordinates
        sc_world = obj_sc;
        sc_world(:,1) = sc_world(:,1) + rx;
        sc_world(:,2) = sc_world(:,2) + ry;

        % Generate IF cube
        t       = (0:N_s-1).' / Fs;
        IF_cube = zeros(N_s, N_chirps, N_ant);

        for s = 1:size(sc_world,1)
            sx  = sc_world(s,1);
            sy  = sc_world(s,2);
            amp = sc_world(s,3);

            d_s     = sqrt(sx^2 + sy^2);
            theta_s = atan2d(sx, sy);

            if d_s < 0.1 || d_s > d_max; continue; end

            f_beat     = S*(2*d_s)/c;
            phi_r      = (4*pi*fc*d_s)/c;
            phi_a_base = (2*pi/lambda)*D*sind(theta_s);

            for k = 1:N_chirps
                for n = 1:N_ant
                    phi_a = phi_a_base*(n-1);
                    sig   = amp*exp(1j*(2*pi*f_beat*t + phi_r + phi_a));
                    IF_cube(:,k,n) = IF_cube(:,k,n) + sig;
                end
            end
        end

        % ── Add realistic hardware imperfections ──────────
        % These make the simulated data more similar to real hardware

        % 1. Thermal noise (SNR ~ 20dB — realistic for short range)
        snr_db    = 20;
        noise_amp = 10^(-snr_db/20);
        IF_cube   = IF_cube + noise_amp * ...
                    (randn(size(IF_cube)) + 1j*randn(size(IF_cube)));

        % 2. Phase noise (random phase jitter per chirp)
        % Models oscillator imperfections in real hardware
        phase_noise_std = 0.05;   % radians
        for k = 1:N_chirps
            phase_jitter = phase_noise_std * randn(1,1,N_ant);
            IF_cube(:,k,:) = IF_cube(:,k,:) .* exp(1j*phase_jitter);
        end

        % 3. DC offset (LO leakage — very common in real FMCW hardware)
        % Shows up as a strong peak at range = 0
        dc_offset = 0.02 * (1 + 1j);
        IF_cube   = IF_cube + dc_offset;

        % 4. Inter-antenna gain imbalance
        % Real antennas have slightly different gains
        gain_imbalance = 1 + 0.05*randn(1,1,N_ant);
        IF_cube        = IF_cube .* gain_imbalance;

        fprintf('Simulated %s at %.1fm with hardware imperfections\n', ...
                TEST_OBJECT, obj_range);
end

fprintf('IF cube shape: [%d × %d × %d]\n\n', size(IF_cube));

%% =========================================================
%  SECTION 3: DC REMOVAL
%  Real radars always have a DC offset from LO leakage.
%  Subtract the mean across chirps to remove it.
%  This is standard preprocessing for real FMCW data.
%% =========================================================

% Remove mean across chirp dimension (DC component)
IF_cube = IF_cube - mean(IF_cube, 2);
fprintf('DC offset removed\n');

%% =========================================================
%  SECTION 4: RANGE FFT
%  Identical to training pipeline — must match exactly
%% =========================================================

NFFT_r = 512;

% Hanning window — same as training
win_range = hanning(N_s);
win_range = reshape(win_range, [N_s,1,1]);
IF_win    = IF_cube .* win_range;

% Full spectrum FFT (complex signal)
range_fft  = fft(IF_win, NFFT_r, 1);   % [512 × 128 × 4]

% Range axis
f_ax       = (0:NFFT_r-1)*(Fs/NFFT_r);
range_axis = (f_ax*c)/(2*S);

%% =========================================================
%  SECTION 5: CLUTTER REMOVAL (Static Object Subtraction)
%
%  In a real room, walls and static furniture produce strong
%  reflections that appear at fixed ranges across all chirps.
%  This step removes them by subtracting the mean across chirps.
%
%  This is critical for real-world deployment —
%  without it, the room itself dominates the heatmap.
%% =========================================================

% Subtract mean across chirps (removes static clutter)
clutter       = mean(range_fft, 2);
range_fft_dc  = range_fft - clutter;
fprintf('Static clutter removed\n');

%% =========================================================
%  SECTION 6: ANGLE FFT → RANGE-ANGLE HEATMAP
%  Identical to training pipeline — must match exactly
%% =========================================================

NFFT_a    = 256;

% Angle FFT across antennas
ang_fft   = fftshift(fft(range_fft_dc, NFFT_a, 3), 3);
RA_map    = mean(abs(ang_fft), 2);     % average across chirps
RA_map    = squeeze(RA_map);           % [512 × 256]

% Angle axis
omega_a   = linspace(-pi, pi, NFFT_a);
sin_theta = (lambda * omega_a) / (2*pi*D);
valid_a   = abs(sin_theta) <= 1;
ang_axis  = NaN(1, NFFT_a);
ang_axis(valid_a) = rad2deg(asin(sin_theta(valid_a)));

%% =========================================================
%  SECTION 7: NORMALISE
%  Must match training normalisation exactly.
%  Training used: RA = RA / max(RA(:))
%  Apply the same here.
%% =========================================================

RA_normalised = RA_map / (max(RA_map(:)) + eps);

fprintf('Heatmap shape: [%d × %d]\n', size(RA_normalised));
fprintf('Value range  : [%.4f, %.4f]\n', ...
        min(RA_normalised(:)), max(RA_normalised(:)));

%% =========================================================
%  SECTION 8: VISUALISE THE REAL HEATMAP
%  Inspect before sending to Python.
%  Should look similar to training heatmaps.
%% =========================================================

zoom_half = 1.5;
obj_approx_range = range_axis(find(mean(RA_map,2) == max(mean(RA_map,2)), 1));
y_lo = max(0,         obj_approx_range - zoom_half);
y_hi = min(d_max,     obj_approx_range + zoom_half);

figure('Name','Real/Realistic Radar Heatmap', ...
       'NumberTitle','off','Color','w','Position',[200 200 900 420]);

subplot(1,2,1);
imagesc(ang_axis, range_axis, 20*log10(RA_normalised + eps));
colormap('jet'); clim([-40 0]); colorbar;
set(gca,'YDir','normal');
xlabel('Angle (degrees)'); ylabel('Range (m)');
title('Full Range-Angle Map');
xlim([-90 90]);

subplot(1,2,2);
imagesc(ang_axis, range_axis, 20*log10(RA_normalised + eps));
colormap('jet'); clim([-40 0]); colorbar;
set(gca,'YDir','normal');
xlabel('Angle (degrees)'); ylabel('Range (m)');
title('Zoomed — Object Region');
xlim([-60 60]);
ylim([y_lo y_hi]);

sgtitle(sprintf('Real Radar Heatmap | Object ~%.1fm | %s source', ...
        obj_approx_range, DATA_SOURCE), ...
        'FontSize',12,'FontWeight','bold');

%% =========================================================
%  SECTION 9: SAVE FOR PYTHON INFERENCE
%  Save in the same format as the training dataset.
%  Python loads this and calls predict_object_multitask().
%% =========================================================

save('real_heatmap.mat', ...
     'RA_normalised', ...   % [512 × 256] normalised heatmap
     'RA_map', ...          % [512 × 256] raw heatmap
     'range_axis', ...      % [512] range values in metres
     'ang_axis', ...        % [256] angle values in degrees
     'DATA_SOURCE');

fprintf('\nSaved: real_heatmap.mat\n');
fprintf('Now run predict_real.py to classify this object.\n');
