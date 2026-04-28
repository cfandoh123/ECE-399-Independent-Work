%% =========================================================
%  radar_shape_simple.m
%  Simplified Radar Shape Classification Dataset Generator
%
%  Task:   Classify shape only — Circle, Square, Rectangle,
%          Triangle, Oval
%  Material: All metal (high amplitude scatterers)
%  Output: radar_shapes_simple.mat → feed into Python CNN
%
%  Dataset per shape:
%    36 orientations × 3 sizes × 4 ranges = 432 samples
%    5 shapes × 432 = 2,160 total samples
%
%  Project: Radar Object Detection + ML (Junior IW)
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
antenna_counts    = [4, 8, 10, 12, 16, 32];
N_ant          = antenna_counts(3);   % default for visualisation sections
D = lambda / 2;
d_max    = (Fs * c) / (2 * S);
d_res    = c / (2 * B);

fprintf('===== Radar Parameters =====\n');
fprintf('Range res  : %.2f cm\n', d_res * 100);
fprintf('Max range  : %.2f m\n',  d_max);
fprintf('============================\n\n');

%% =========================================================
%  SECTION 2: SHAPE DEFINITIONS — All Metal
%
%  All shapes use:
%    Corner amplitude : 1.0  (metal corner reflector)
%    Edge amplitude   : 0.70 (flat metal surface)
%    Curve amplitude  : 0.50 (curved metal surface)
%
%  Each builder function returns [x, y, amplitude] in LOCAL
%  coordinates centred on the shape centroid.
%
%  Size parameter controls the scale. We will generate
%  each shape at 3 sizes: small, medium, large.
%% =========================================================

% ── Corner and edge amplitudes (all metal) ───────────────
AMP_CORNER = 1.00;
AMP_EDGE   = 0.70;
AMP_CURVE  = 0.50;

% ── Shape 1: Circle ──────────────────────────────────────
% Smooth ring of points — no corners
% Amplitude is uniform around the circumference
function sc = make_circle(radius)
    N   = 24;   % points around circumference
    ang = linspace(0, 2*pi, N+1); ang = ang(1:end-1);
    x   = radius * cos(ang(:));
    y   = radius * sin(ang(:));
    sc  = [x, y, 0.50 * ones(N,1)];
end

% ── Shape 2: Square ──────────────────────────────────────
% Equal width and height
% 4 strong corners + edge points on all 4 sides
function sc = make_square(side)
    h      = side / 2;
    N_edge = 5;
    e      = linspace(-h, h, N_edge+2); e = e(2:end-1);

    corners = [-h,-h,1.0; h,-h,1.0; h,h,1.0; -h,h,1.0];
    bottom  = [e(:), -h*ones(N_edge,1), 0.70*ones(N_edge,1)];
    top     = [e(:),  h*ones(N_edge,1), 0.70*ones(N_edge,1)];
    left    = [-h*ones(N_edge,1), e(:), 0.70*ones(N_edge,1)];
    right   = [ h*ones(N_edge,1), e(:), 0.70*ones(N_edge,1)];
    sc      = [corners; bottom; top; left; right];
end

% ── Shape 3: Rectangle ───────────────────────────────────
% Width > Height (aspect ratio 2:1)
% 4 corners + more edge points along longer dimension
function sc = make_rectangle(width, height)
    hw     = width  / 2;
    hh     = height / 2;
    N_w    = 8;   % more points along width
    N_h    = 4;   % fewer along height

    ew = linspace(-hw, hw, N_w+2); ew = ew(2:end-1);
    eh = linspace(-hh, hh, N_h+2); eh = eh(2:end-1);

    corners = [-hw,-hh,1.0; hw,-hh,1.0; hw,hh,1.0; -hw,hh,1.0];
    bottom  = [ew(:), -hh*ones(N_w,1), 0.70*ones(N_w,1)];
    top     = [ew(:),  hh*ones(N_w,1), 0.70*ones(N_w,1)];
    left    = [-hw*ones(N_h,1), eh(:), 0.70*ones(N_h,1)];
    right   = [ hw*ones(N_h,1), eh(:), 0.70*ones(N_h,1)];
    sc      = [corners; bottom; top; left; right];
end

% ── Shape 4: Triangle ────────────────────────────────────
% Equilateral triangle — 3 strong corners + 3 edges
function sc = make_triangle(side)
    ht  = side * sqrt(3) / 2;
    v1  = [0,       2*ht/3];    % apex
    v2  = [-side/2, -ht/3];    % bottom-left
    v3  = [ side/2, -ht/3];    % bottom-right

    corners = [v1,1.0; v2,1.0; v3,1.0];

    N_e = 6;
    t   = linspace(0,1,N_e+2); t = t(2:end-1);

    e1 = [(1-t')*v1(1)+t'*v2(1), (1-t')*v1(2)+t'*v2(2), 0.70*ones(N_e,1)];
    e2 = [(1-t')*v2(1)+t'*v3(1), (1-t')*v2(2)+t'*v3(2), 0.70*ones(N_e,1)];
    e3 = [(1-t')*v3(1)+t'*v1(1), (1-t')*v3(2)+t'*v1(2), 0.70*ones(N_e,1)];
    sc = [corners; e1; e2; e3];
end

% ── Shape 5: Oval (Ellipse) ──────────────────────────────
% Width > Height (different from circle — unequal axes)
% Key distinction from circle: non-uniform curvature means
% ends (high curvature) return more energy than sides
function sc = make_oval(width, height)
    a  = width  / 2;   % semi-major axis
    b  = height / 2;   % semi-minor axis
    N  = 24;
    ang = linspace(0, 2*pi, N+1); ang = ang(1:end-1);
    x   = a * cos(ang(:));
    y   = b * sin(ang(:));

    % Amplitude varies with local curvature
    % High curvature at ends (±a, 0) → stronger return
    % Low curvature at top/bottom (0, ±b) → weaker return
    kappa = (a*b) ./ ((b*cos(ang(:))).^2 + (a*sin(ang(:))).^2).^1.5;
    kappa_norm = kappa / max(kappa);
    amp   = 0.35 + 0.40 * kappa_norm;   % range [0.35, 0.75]

    sc = [x, y, amp];
end

%% =========================================================
%  SECTION 3: SIZE VARIANTS
%
%  Each shape is generated at 3 sizes.
%  Size is defined by the primary dimension (metres).
%
%  Small  : 0.20m  (fits inside 20cm cube)
%  Medium : 0.40m  (fits inside 40cm cube)
%  Large  : 0.60m  (fits inside 60cm cube)
%
%  Rectangle and oval have fixed aspect ratios:
%    Rectangle: width = 2 × height
%    Oval:      width = 1.8 × height
%% =========================================================

sizes_label = {'Small (0.20m)', 'Medium (0.40m)', 'Large (0.60m)'};
sizes_val   = [0.20, 0.40, 0.60];   % primary dimension

% Build scatterer sets for all shapes at all sizes
shape_names = {'Circle', 'Square', 'Rectangle', 'Triangle', 'Oval'};
N_classes   = 5;

% Cell array: shape_scatterers{shape_idx}{size_idx}
shape_scatterers = cell(N_classes, length(sizes_val));

for si = 1:length(sizes_val)
    s = sizes_val(si);
    shape_scatterers{1,si} = make_circle(s/2);             % radius = s/2
    shape_scatterers{2,si} = make_square(s);                % side = s
    shape_scatterers{3,si} = make_rectangle(s, s/2);        % 2:1 ratio
    % With a loop over multiple aspect ratios:
% aspect_ratios = [2.0, 3.0, 4.0];   % all are "Rectangle"
% for ar = aspect_ratios
    % sc = make_rectangle(s, s/ar);
    % generate heatmap and store with label = Rectangle
%end
    shape_scatterers{4,si} = make_triangle(s);              % side = s
    shape_scatterers{5,si} = make_oval(s, s/1.8);           % 1.8:1 ratio
end

fprintf('===== Shape + Size Definitions =====\n');
for i = 1:N_classes
    fprintf('  %d. %s\n', i, shape_names{i});
    for si = 1:length(sizes_val)
        fprintf('       %s — %d scatterers\n', ...
                sizes_label{si}, size(shape_scatterers{i,si},1));
    end
end
fprintf('=====================================\n\n');

%% =========================================================
%  SECTION 4: FFT AXES
%% =========================================================

NFFT_r = 512;
NFFT_a = 256;

f_ax       = (0:NFFT_r-1) * (Fs/NFFT_r);
range_axis = (f_ax * c) / (2*S);

omega_a    = linspace(-pi, pi, NFFT_a);
sin_theta  = (lambda * omega_a) / (2*pi*D);
valid_a    = abs(sin_theta) <= 1;
ang_axis   = NaN(1, NFFT_a);
ang_axis(valid_a) = rad2deg(asin(sin_theta(valid_a)));

%% =========================================================
%  SECTION 5: HELPER — Rotate and Place Scatterers
%% =========================================================

function world_sc = place_shape(sc_local, rx, ry, angle_deg)
    R        = [cosd(angle_deg), -sind(angle_deg);
                sind(angle_deg),  cosd(angle_deg)];
    xy_rot   = (R * sc_local(:,1:2).').' ;
    world_xy = xy_rot + repmat([rx, ry], size(sc_local,1), 1);
    world_sc = [world_xy, sc_local(:,3)];
end

%% =========================================================
%  SECTION 6: HELPER — Generate Range-Angle Map
%% =========================================================

function RA = generate_RA(world_sc, rp)
    t  = (0:rp.N_s-1).' / rp.Fs;
    IF = zeros(rp.N_s, rp.N_ant);

    for s = 1:size(world_sc,1)
        sx  = world_sc(s,1);
        sy  = world_sc(s,2);
        amp = world_sc(s,3);
        d   = sqrt(sx^2 + sy^2);
        th  = atan2d(sx, sy);

        if d < 0.1 || d > rp.d_max; continue; end

        f_b    = rp.S * 2*d / rp.c;
        phi_r  = 4*pi*rp.fc*d / rp.c;
        phi_ab = (2*pi/rp.lambda) * rp.D * sind(th);

        for n = 1:rp.N_ant
            sig = amp * exp(1j*(2*pi*f_b*t + phi_r + phi_ab*(n-1)));
            IF(:,n) = IF(:,n) + sig;
        end
    end

    % Noise (SNR = 25 dB)
    noise_amp = 10^(-rp.snr_db / 20)
    IF = IF + noise_amp * (randn(size(IF)) + 1j*randn(size(IF)));

    % Hanning window + Range FFT (full spectrum)
    win     = hanning(rp.N_s);
    rng_fft = fft(IF .* win, rp.NFFT_r, 1);

    % Angle FFT
    ang_fft = fftshift(fft(rng_fft, rp.NFFT_a, 2), 2);

    % Average magnitude across chirp dimension
    RA = abs(mean(ang_fft, 1));   % actually no chirp dim here
    RA = abs(ang_fft);            % [NFFT_r × NFFT_a]
end

%% =========================================================
%  SECTION 7: CANONICAL HEATMAPS — Visual Inspection
%  One per shape at medium size, 4m, 0° orientation
%  Run this first to confirm signatures look correct
%  BEFORE generating the full dataset
%% =========================================================

rp.c=c; rp.fc=fc; rp.lambda=lambda; rp.S=S; rp.Fs=Fs;
rp.N_s=N_s; rp.N_ant=N_ant; rp.D=D; rp.d_max=d_max;
rp.NFFT_r=NFFT_r; rp.NFFT_a=NFFT_a; rp.snr_db = 20;

obj_range = 4.0;
zoom_half = 0.8;
y_lo      = obj_range - zoom_half;
y_hi      = obj_range + zoom_half;

shape_colors = {[0.2 0.6 1.0],   % Circle   — blue
                [1.0 0.4 0.1],   % Square   — orange
                [0.2 0.8 0.2],   % Rectangle— green
                [0.8 0.2 0.8],   % Triangle — purple
                [1.0 0.8 0.0]};  % Oval     — yellow

figure('Name','Canonical Heatmaps — 5 Shapes (Metal, 4m, 0°, Medium)', ...
       'NumberTitle','off','Color','w','Position',[50 200 1500 360]);

for i = 1:N_classes
    sc_local = shape_scatterers{i, 2};   % medium size
    world_sc = place_shape(sc_local, 0, obj_range, 0);
    RA       = generate_RA(world_sc, rp);
    RA_dB    = 20*log10(RA / max(RA(:)) + eps);

    subplot(1, N_classes, i);
    imagesc(ang_axis, range_axis, RA_dB);
    colormap('jet'); clim([-40 0]); colorbar;
    set(gca, 'YDir', 'normal');
    xlim([-60 60]); ylim([y_lo y_hi]);
    xlabel('Angle (°)', 'FontSize', 10);
    ylabel('Range (m)', 'FontSize', 10);
    title(shape_names{i}, 'FontSize', 13, 'FontWeight', 'bold', ...
          'Color', shape_colors{i});
    hold on;
    plot(0, obj_range, 'w+', 'MarkerSize', 12, 'LineWidth', 2);
end

sgtitle('Canonical Range-Angle Heatmaps | All Metal | 4m | 0° | Medium Size', ...
        'FontSize', 12, 'FontWeight', 'bold');

fprintf('Canonical heatmaps plotted. Inspect before generating dataset.\n');
fprintf('Press any key to continue to dataset generation...\n');
pause;

%% =========================================================
%  SECTION 8: ORIENTATION COMPARISON
%  Show each shape at 4 orientations to confirm the model
%  will see sufficient variation during training
%% =========================================================

figure('Name','Orientation Variation — All Shapes', ...
       'NumberTitle','off','Color','w','Position',[50 50 1500 700]);

orientations_show = [0, 45, 90, 135];

for i = 1:N_classes
    for j = 1:4
        subplot(N_classes, 4, (i-1)*4 + j);

        sc_local = shape_scatterers{i, 2};   % medium size
        world_sc = place_shape(sc_local, 0, obj_range, orientations_show(j));
        RA       = generate_RA(world_sc, rp);
        RA_dB    = 20*log10(RA / max(RA(:)) + eps);

        imagesc(ang_axis, range_axis, RA_dB);
        colormap('jet'); clim([-40 0]);
        set(gca, 'YDir', 'normal');
        xlim([-60 60]); ylim([y_lo y_hi]);

        if j == 1
            ylabel(sprintf('%s\nRange(m)', shape_names{i}), ...
                   'FontSize', 8, 'Color', shape_colors{i}, ...
                   'FontWeight', 'bold');
        end
        if i == N_classes
            xlabel('Angle (°)', 'FontSize', 8);
        end
        title(sprintf('%d°', orientations_show(j)), 'FontSize', 9);
        hold on;
        plot(0, obj_range, 'w+', 'MarkerSize', 6, 'LineWidth', 1.5);
    end
end

sgtitle('Orientation Variation | Rows = Shapes | Cols = 0°, 45°, 90°, 135°', ...
        'FontSize', 12, 'FontWeight', 'bold');

%% =========================================================
%  SECTION 9: FULL DATASET GENERATION
%
%  For each shape:
%    36 orientations (0° to 350°, step 10°)
%    3 sizes         (small, medium, large)
%    4 ranges        (2, 3, 4, 5 m)
%    = 432 samples per shape
%    = 2,160 total samples
%
%  Each sample: normalised Range-Angle map [512 × 256]
%  Each label:  integer 1-5
%
%  Label map:
%    1 = Circle
%    2 = Square
%    3 = Rectangle
%    4 = Triangle
%    5 = Oval
%% =========================================================

for ant_exp = 1:length(antenna_counts)

    N_ant     = antenna_counts(ant_exp);
    rp.N_ant  = N_ant;

    % Recompute angle axis for this antenna count
    % (range axis does not change)
    omega_a   = linspace(-pi, pi, NFFT_a);
    sin_theta = (lambda * omega_a) / (2*pi*D);
    valid_a   = abs(sin_theta) <= 1;
    ang_axis  = NaN(1, NFFT_a);
    ang_axis(valid_a) = rad2deg(asin(sin_theta(valid_a)));

    fprintf('\n========================================\n');
    fprintf('  ANTENNA EXPERIMENT: N_ant = %d\n', N_ant);
    fprintf('========================================\n');


snr_levels = [5, 10, 20, 30];  %db - low, medium, high SNR
orientations_ds = 0 : 10 : 350;     % 36 values
ranges_ds       = [2.0, 3.0, 4.0, 5.0];   % 4 values
% sizes_val already defined: [0.20, 0.40, 0.60]  % 3 values

samples_per_shape = length(orientations_ds) * ...
                    length(ranges_ds)       * ...
                    length(sizes_val)       * ...
                    length(snr_levels);
total_samples     = samples_per_shape * N_classes;

fprintf('\n===== Dataset Generation =====\n');
fprintf('Orientations : %d (0 to 350, step 10)\n', length(orientations_ds));
fprintf('Sizes        : %d (%.2f, %.2f, %.2f m)\n', ...
        length(sizes_val), sizes_val(1), sizes_val(2), sizes_val(3));
fprintf('Ranges       : %d (%.0f, %.0f, %.0f, %.0f m)\n', ...
        length(ranges_ds), ranges_ds(1), ranges_ds(2), ...
        ranges_ds(3), ranges_ds(4));
fprintf('Per shape    : %d samples\n', samples_per_shape);
fprintf('Total        : %d samples\n', total_samples);
fprintf('==============================\n\n');

% Pre-allocate
X = zeros(total_samples, NFFT_r, NFFT_a, 'single');
Y = zeros(total_samples, 1, 'uint8');

sample_idx = 0;
tic;

for shape_i = 1:N_classes
    fprintf('Generating %s', shape_names{shape_i});

    for size_i = 1:length(sizes_val)
        sc_local = shape_scatterers{shape_i, size_i};

        for ori = orientations_ds
            % Pre-rotate scatterers once per orientation
            R_mat    = [cosd(ori), -sind(ori);
                        sind(ori),  cosd(ori)];
            xy_rot   = (R_mat * sc_local(:,1:2).').' ;

            for rng = ranges_ds
                for snr = snr_levels
                % Place at boresight (0 lateral) at this range
                world_sc = [xy_rot(:,1) + 0, ...
                            xy_rot(:,2) + rng, ...
                            sc_local(:,3)];
                rp.snr_db = snr;

                RA = generate_RA(world_sc, rp);

                % Normalise to [0,1]
                RA = single(RA / (max(RA(:)) + eps));

                sample_idx = sample_idx + 1;
                X(sample_idx, :, :) = RA;
                Y(sample_idx)       = uint8(shape_i);
                end
            end
        end
    end

    fprintf(' — %d samples done\n', samples_per_shape);
end
elapsed = toc;
fprintf('\nTotal: %d samples in %.1f seconds\n\n', sample_idx, elapsed);



%% =========================================================
%  SECTION 10: TRAIN / VAL / TEST SPLIT
%  Stratified: 70% train / 20% val / 10% test
%  Uses RandStream to avoid rng variable conflict
%% =========================================================

fprintf('===== Splitting Dataset =====\n');

train_ratio = 0.60;
val_ratio   = 0.30;

train_idx = [];
val_idx   = [];
test_idx  = [];

rs = RandStream('mt19937ar', 'Seed', 42);

for cls = 1:N_classes
    idx  = find(Y == cls);
    idx  = idx(randperm(rs, length(idx)));
    n    = length(idx);
    n_tr = round(n * train_ratio);
    n_vl = round(n * val_ratio);

    train_idx = [train_idx; idx(1:n_tr)];
    val_idx   = [val_idx;   idx(n_tr+1 : n_tr+n_vl)];
    test_idx  = [test_idx;  idx(n_tr+n_vl+1 : end)];
end

X_train = X(train_idx, :, :);  Y_train = Y(train_idx);
X_val   = X(val_idx,   :, :);  Y_val   = Y(val_idx);
X_test  = X(test_idx,  :, :);  Y_test  = Y(test_idx);

fprintf('Train : %d\n', length(train_idx));
fprintf('Val   : %d\n', length(val_idx));
fprintf('Test  : %d\n', length(test_idx));
fprintf('=============================\n\n');

%% =========================================================
%  SECTION 11: DATASET STATISTICS PLOT
%% =========================================================

figure('Name','Dataset Statistics','NumberTitle','off', ...
       'Color','w','Position',[400 300 650 380]);

counts = histcounts(Y, 1:N_classes+1);
clrs   = vertcat(shape_colors{:});
b      = bar(1:N_classes, counts, 'FaceColor', 'flat');
b.CData = clrs;
xticks(1:N_classes);
xticklabels(shape_names);
ylabel('Number of Samples', 'FontSize', 11);
title(sprintf('Dataset: %d Total Samples | %d Shapes | All Metal', ...
              total_samples, N_classes), 'FontSize', 12);
grid on;
for i = 1:N_classes
    text(i, counts(i)+5, num2str(counts(i)), ...
         'HorizontalAlignment','center','FontWeight','bold','FontSize',11);
end

%% =========================================================
%  SECTION 12: SAVE
%% =========================================================

class_names = shape_names;

filename = sprintf('radar_shapes_N%d.mat', N_ant);
save(filename, ...
     'X_train', 'Y_train', ...
     'X_val',   'Y_val',   ...
     'X_test',  'Y_test',  ...
     'range_axis', 'ang_axis', ...
     'class_names', 'N_classes', ...
     'NFFT_r', 'NFFT_a', 'N_ant', ...
     'sizes_val', 'orientations_ds', 'ranges_ds', '-v7.3');

fprintf('Saved: %s\n', filename);
fprintf('X : [N × %d × %d]  Range-Angle maps\n', NFFT_r, NFFT_a);
fprintf('Y : [N × 1]  Labels\n\n');
fprintf('Label map:\n');
for i = 1:N_classes
    fprintf('  %d = %s\n', i, shape_names{i});
end

end

fprintf('\nAll antenna experiments complete.\n');
fprintf('Files saved: radar_shapes_N4.mat, N8.mat, N16.mat, N32.mat\n');

fprintf('\nNext: run radar_shape_simple_cnn.py\n');
fprintf('==========================================\n');
