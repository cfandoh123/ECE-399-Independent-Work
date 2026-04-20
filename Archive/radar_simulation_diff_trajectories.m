%% =========================================================
%  FMCW Radar Simulation — Milestone 2
%  4-Antenna Array | Moving Object | Frame-by-Frame Heatmaps
%
%  Data Hierarchy:
%    Sample  → fast time  (within chirp)  → Range
%    Chirp   → slow time  (within frame)  → Velocity
%    Frame   → time evolution             → Trajectory
%    Antenna → spatial    (across array)  → Angle
%
%  Output:
%    RD_cube  [range_bins × doppler_bins × N_frames]  ← ML input later
%    RA_cube  [range_bins × angle_bins  × N_frames]   ← alternative view
%
%  Based on: ECE/COS 368 Lecture 32 (Ghasempour, Princeton)
%  Project:  Radar Object Detection + ML (Junior IW)
%% =========================================================

clear; clc; close all;

%% =========================================================
%  SECTION 1: RADAR PARAMETERS
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

% Frame timing
T_frame  = N_chirps * Tc;    % Duration of one frame (seconds)
N_frames = 50;               % Total number of frames to simulate

% Performance metrics
d_res  = c / (2*B);
d_max  = (Fs*c) / (2*S);
v_max  = lambda / (4*Tc);
v_res  = lambda / (2*N_chirps*Tc);

fprintf('===== Radar Parameters =====\n');
fprintf('Frame duration : %.2f ms\n', T_frame*1e3);
fprintf('Total sim time : %.2f s\n',  N_frames*T_frame);
fprintf('Range res      : %.2f cm\n', d_res*100);
fprintf('Max range      : %.2f m\n',  d_max);
fprintf('Velocity res   : %.4f m/s\n', v_res);
fprintf('Max velocity   : %.2f m/s\n', v_max);
fprintf('============================\n\n');

%% =========================================================
%  SECTION 2: OBJECT TRAJECTORY DEFINITION
%
%  We define a smooth trajectory for a single person.
%  Position is (x, y) in meters where:
%    x = lateral position (left/right)
%    y = depth from radar  (range)
%
%  The radar sits at origin (0,0) facing +y direction.
%
%  Available trajectories — set TRAJECTORY to one of:
%    'walk_across'   — person walks left to right at fixed depth
%    'walk_toward'   — person walks straight toward radar
%    'arc'           — person walks in a curved arc
%    'L_shape'       — person walks toward then turns right
%
%  Range  = sqrt(x^2 + y^2)    (distance from radar)
%  Angle  = atan2d(x, y)       (angle from boresight)
%  Radial velocity = d(range)/dt
%% =========================================================

TRAJECTORY = 'arc';   % <-- Change this to try different paths

% Frame time axis
frame_times = (0 : N_frames-1) * T_frame;

switch TRAJECTORY

    case 'walk_across'
        % Person walks from left to right at 10m depth
        x_traj = linspace(-8, 8, N_frames);
        y_traj = 10 * ones(1, N_frames);

    case 'walk_toward'
        % Person walks straight toward radar from 20m to 5m
        x_traj = zeros(1, N_frames);
        y_traj = linspace(20, 5, N_frames);

    case 'arc'
        % Person walks in a curved arc — changes both range and angle
        angle_traj = linspace(-40, 40, N_frames);   % degrees swept
        radius     = 12;                             % arc radius (m)
        x_traj     = radius * sind(angle_traj);
        y_traj     = radius * cosd(angle_traj);

    case 'L_shape'
        % Person walks toward radar then turns right
        half = round(N_frames/2);
        x_traj = [zeros(1,half),     linspace(0, 8, N_frames-half)];
        y_traj = [linspace(20,10,half), 10*ones(1, N_frames-half)];
end

% Derived trajectory quantities
range_traj = sqrt(x_traj.^2 + y_traj.^2);          % range per frame (m)
angle_traj = atan2d(x_traj, y_traj);                % AoA per frame (deg)

% Radial velocity per frame (finite difference of range)
radvel_traj        = zeros(1, N_frames);
radvel_traj(2:end) = diff(range_traj) / T_frame;    % m/s
radvel_traj(1)     = radvel_traj(2);                % edge case

fprintf('===== Trajectory: %s =====\n', TRAJECTORY);
fprintf('Range  : %.1f m  →  %.1f m\n', range_traj(1),  range_traj(end));
fprintf('Angle  : %.1f°  →  %.1f°\n',  angle_traj(1),  angle_traj(end));
fprintf('Radial vel range: [%.2f, %.2f] m/s\n', min(radvel_traj), max(radvel_traj));
fprintf('===========================\n\n');

%% =========================================================
%  SECTION 3: FFT AXES  (computed once, reused every frame)
%% =========================================================

NFFT_r = 512;
NFFT_d = 256;
NFFT_a = 256;

% Range axis — full spectrum (complex IF signal, no half-spectrum cut)
f_axis     = (0:NFFT_r-1) * (Fs/NFFT_r);
range_axis = (f_axis*c) / (2*S);
half_r     = 1:NFFT_r;            % use ALL bins, not just half
range_axis = range_axis(half_r);

% Velocity axis
omega_d  = linspace(-pi, pi, NFFT_d);
vel_axis = (lambda * omega_d) / (4*pi*Tc);

% Angle axis
omega_a   = linspace(-pi, pi, NFFT_a);
sin_theta = (lambda * omega_a) / (2*pi*D);
valid_a   = abs(sin_theta) <= 1;
ang_axis  = NaN(1, NFFT_a);
ang_axis(valid_a) = rad2deg(asin(sin_theta(valid_a)));

%% =========================================================
%  SECTION 4: FRAME-BY-FRAME SIMULATION
%
%  For each frame f:
%    1. Get object position (range, velocity, angle) at frame f
%    2. Generate IF cube [samples × chirps × antennas]
%       with chirp-level time-varying range (from Milestone 1b)
%    3. Range FFT  → [range_bins × chirps × antennas]
%    4. Doppler FFT → Range-Doppler map [range_bins × doppler_bins]
%    5. Angle FFT   → Range-Angle map   [range_bins × angle_bins]
%    6. Store in output cubes
%% =========================================================

% Output cubes for ML — first dimension is NFFT_r (full spectrum)
RD_cube = zeros(NFFT_r, NFFT_d, N_frames);   % Range-Doppler
RA_cube = zeros(NFFT_r, NFFT_a, N_frames);   % Range-Angle

t = (0:N_s-1).' / Fs;   % Fast-time column vector

fprintf('Simulating %d frames', N_frames);

for f = 1:N_frames

    if mod(f, 10) == 0
        fprintf('.');
    end

    % Object state at this frame
    d0    = range_traj(f);
    v     = radvel_traj(f);
    theta = angle_traj(f);
    A     = 1.0;

    phi_angle_base = (2*pi/lambda) * D * sind(theta);

    % Build IF cube for this frame
    IF_cube = zeros(N_s, N_chirps, N_ant);

    for k = 1:N_chirps
        % Time-varying range within the frame
        d_k    = d0 + v*(k-1)*Tc;
        f_beat = S*(2*d_k)/c;
        phi_r  = (4*pi*fc*d_k)/c;

        for n = 1:N_ant
            phi_a  = phi_angle_base * (n-1);
            signal = A * exp(1j*(2*pi*f_beat*t + phi_r + phi_a));
            IF_cube(:,k,n) = IF_cube(:,k,n) + signal;
        end
    end

    % Noise
    IF_cube = IF_cube + 0.1*(randn(size(IF_cube)) + 1j*randn(size(IF_cube)));

    % --- Hanning window along fast-time (range) dimension ---
    win_range = hanning(N_s);
    win_range = reshape(win_range, [N_s, 1, 1]);
    IF_cube   = IF_cube .* win_range;

    % --- Range FFT ---
    range_fft = fft(IF_cube, NFFT_r, 1);
    range_fft = range_fft(half_r, :, :);   % full spectrum

    % --- Hanning window along slow-time (Doppler) dimension ---
    win_doppler = hanning(N_chirps);
    win_doppler = reshape(win_doppler, [1, N_chirps, 1]);
    range_fft_w = range_fft .* win_doppler;

    % --- Doppler FFT → Range-Doppler map ---
    dop_fft = fftshift(fft(range_fft_w, NFFT_d, 2), 2);
    RD_map  = mean(abs(dop_fft), 3);       % average across antennas
    RD_cube(:,:,f) = RD_map;

    % --- Angle FFT → Range-Angle map ---
    % For each range bin, take FFT across antennas
    ang_fft = fftshift(fft(range_fft, NFFT_a, 3), 3);
    RA_map  = mean(abs(ang_fft), 2);       % average across chirps
    RA_cube(:,:,f) = squeeze(RA_map);

end

fprintf(' done!\n\n');

%% =========================================================
%  SECTION 5: TRAJECTORY VISUALIZATION
%% =========================================================

figure('Name','Object Trajectory (Top-Down View)', ...
       'NumberTitle','off','Color','w','Position',[50 500 500 450]);

scatter(x_traj, y_traj, 40, 1:N_frames, 'filled');
colormap(gca, 'cool'); cb = colorbar;
cb.Label.String = 'Frame index';
hold on;
plot(0, 0, 'r^', 'MarkerSize', 12, 'MarkerFaceColor','r');
text(0.3, -0.5, 'Radar', 'Color','r','FontWeight','bold');
plot(x_traj(1),   y_traj(1),   'gs', 'MarkerSize', 10, 'MarkerFaceColor','g');
plot(x_traj(end), y_traj(end), 'rs', 'MarkerSize', 10, 'MarkerFaceColor','r');
xlabel('Lateral position x (m)');
ylabel('Depth y (m)');
title(sprintf('Trajectory: %s', TRAJECTORY));
legend('Path','Radar','Start','End','Location','best');
grid on; axis equal;
xlim([min(x_traj)-2, max(x_traj)+2]);
ylim([-2, max(y_traj)+3]);

%% =========================================================
%  SECTION 6: TRAJECTORY METRICS OVER TIME
%% =========================================================

figure('Name','Trajectory Metrics','NumberTitle','off','Color','w', ...
       'Position',[50 50 900 280]);

subplot(1,3,1);
plot(1:N_frames, range_traj, 'b', 'LineWidth', 1.8);
xlabel('Frame'); ylabel('Range (m)');
title('Range vs Frame'); grid on;

subplot(1,3,2);
plot(1:N_frames, radvel_traj, 'r', 'LineWidth', 1.8);
yline(v_max,  'k--', 'v_{max}',  'LabelHorizontalAlignment','left');
yline(-v_max, 'k--', '-v_{max}', 'LabelHorizontalAlignment','left');
xlabel('Frame'); ylabel('Radial Velocity (m/s)');
title('Radial Velocity vs Frame'); grid on;

subplot(1,3,3);
plot(1:N_frames, angle_traj, 'm', 'LineWidth', 1.8);
yline(90,  'k--'); yline(-90, 'k--');
xlabel('Frame'); ylabel('Angle (degrees)');
title('AoA vs Frame'); grid on;

sgtitle('Object State Across Frames','FontSize',12,'FontWeight','bold');

%% =========================================================
%  SECTION 7: RANGE-DOPPLER HEATMAP — Selected Frames
%
%  Show the RD heatmap at 4 evenly-spaced frames to see
%  how the object moves in range-velocity space over time
%% =========================================================

sel_frames = round(linspace(1, N_frames, 4));

figure('Name','Range-Doppler Heatmaps — Selected Frames', ...
       'NumberTitle','off','Color','w','Position',[600 500 1000 400]);

for idx = 1:4
    f   = sel_frames(idx);
    map = 20*log10(RD_cube(:,:,f) / max(RD_cube(:,:,f), [], 'all') + eps);

    subplot(1,4,idx);
    imagesc(vel_axis, range_axis, map);
    colormap('jet'); clim([-40 0]);
    set(gca,'YDir','normal');
    xlabel('Velocity (m/s)'); ylabel('Range (m)');
    title(sprintf('Frame %d\n(%.1fm, %.1fm/s, %.1f°)', ...
          f, range_traj(f), radvel_traj(f), angle_traj(f)));
    hold on;
    plot(radvel_traj(f), range_traj(f), 'w+', ...
         'MarkerSize',12,'LineWidth',2.5);
end
sgtitle('Range-Doppler Heatmap Evolution','FontSize',12,'FontWeight','bold');

%% =========================================================
%  SECTION 8: RANGE-ANGLE HEATMAP — Selected Frames
%% =========================================================

figure('Name','Range-Angle Heatmaps — Selected Frames', ...
       'NumberTitle','off','Color','w','Position',[600 50 1000 400]);

for idx = 1:4
    f   = sel_frames(idx);
    map = 20*log10(RA_cube(:,:,f) / max(RA_cube(:,:,f), [], 'all') + eps);

    subplot(1,4,idx);
    imagesc(ang_axis, range_axis, map);
    colormap('jet'); clim([-40 0]);
    set(gca,'YDir','normal');
    xlabel('Angle (degrees)'); ylabel('Range (m)');
    title(sprintf('Frame %d\n(%.1fm, %.1f°)', ...
          f, range_traj(f), angle_traj(f)));
    hold on;
    plot(angle_traj(f), range_traj(f), 'w+', ...
         'MarkerSize',12,'LineWidth',2.5);
    xlim([-90 90]);
end
sgtitle('Range-Angle Heatmap Evolution','FontSize',12,'FontWeight','bold');

%% =========================================================
%  SECTION 9: ANIMATED HEATMAP — Watch the Object Move
%% =========================================================

fprintf('Rendering animation (close window to stop)...\n');

fig_anim = figure('Name','Live Range-Doppler Animation', ...
                  'NumberTitle','off','Color','w','Position',[200 200 900 420]);

for f = 1:N_frames
    if ~ishandle(fig_anim), break; end

    RD_map = 20*log10(RD_cube(:,:,f) / max(RD_cube(:,:,f),[],'all') + eps);
    RA_map = 20*log10(RA_cube(:,:,f) / max(RA_cube(:,:,f),[],'all') + eps);

    subplot(1,2,1);
    imagesc(vel_axis, range_axis, RD_map);
    colormap('jet'); clim([-40 0]);
    set(gca,'YDir','normal');
    xlabel('Velocity (m/s)'); ylabel('Range (m)');
    title(sprintf('Range-Doppler | Frame %d/%d', f, N_frames));
    hold on;
    plot(radvel_traj(f), range_traj(f), 'w+','MarkerSize',14,'LineWidth',2.5);
    hold off;

    subplot(1,2,2);
    imagesc(ang_axis, range_axis, RA_map);
    colormap('jet'); clim([-40 0]);
    set(gca,'YDir','normal');
    xlabel('Angle (degrees)'); ylabel('Range (m)');
    title(sprintf('Range-Angle   | Frame %d/%d', f, N_frames));
    xlim([-90 90]);
    hold on;
    plot(angle_traj(f), range_traj(f), 'w+','MarkerSize',14,'LineWidth',2.5);
    hold off;

    sgtitle(sprintf('%.1fm | %.2fm/s | %.1f°', ...
            range_traj(f), radvel_traj(f), angle_traj(f)), ...
            'FontSize',11,'FontWeight','bold');

    drawnow;
    pause(0.08);
end

%% =========================================================
%  SECTION 10: SAVE OUTPUT CUBES FOR ML
%
%  RD_cube  [range_bins × doppler_bins × N_frames]
%  RA_cube  [range_bins × angle_bins   × N_frames]
%
%  These are saved as .mat files ready to be loaded
%  directly as training data in the ML pipeline.
%% =========================================================

save('radar_data_milestone2.mat', ...
     'RD_cube', 'RA_cube', ...
     'range_axis', 'vel_axis', 'ang_axis', ...
     'range_traj', 'radvel_traj', 'angle_traj', ...
     'TRAJECTORY', 'N_frames', 'T_frame', ...
     'lambda', 'fc', 'B', 'Tc', 'N_chirps', 'N_ant');

fprintf('\n===== Output Saved =====\n');
fprintf('RD_cube : [%d × %d × %d]  (range × doppler × frames)\n', ...
        size(RD_cube,1), size(RD_cube,2), size(RD_cube,3));
fprintf('RA_cube : [%d × %d × %d]  (range × angle   × frames)\n', ...
        size(RA_cube,1), size(RA_cube,2), size(RA_cube,3));
fprintf('Saved to: radar_data_milestone2.mat\n');
fprintf('\nNext step: run multiple trajectories and label them\n');
fprintf('           → that .mat file becomes your ML training set\n');
fprintf('========================\n');
