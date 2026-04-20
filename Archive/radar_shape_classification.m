%% =========================================================
%  Radar Shape Classification
%  Circle | Square | Rectangle | Triangle
%
%  Each shape is modelled as a collection of point scatterers
%  placed along its edges and corners.
%
%  Physics:
%    Corners → strongest reflectors (corner reflector effect)
%    Straight edges → medium, distributed along edge
%    Curved edges  → weaker, uniformly spread around curve
%
%  Pipeline:
%    Shape geometry → IF signal → Range FFT → Angle FFT
%    → Range-Angle heatmap → labelled ML dataset
%
%  Output:
%    radar_shapes_dataset.mat  [X_train, X_val, X_test, Y_*]
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

N_ant    = 4;
D        = lambda / 2;

d_res  = c / (2*B);
d_max  = (Fs*c) / (2*S);

fprintf('===== Radar Parameters =====\n');
fprintf('Range res : %.2f cm\n', d_res*100);
fprintf('Max range : %.2f m\n',  d_max);
fprintf('============================\n\n');

%% =========================================================
%  SECTION 2: FFT PARAMETERS AND AXES
%% =========================================================

NFFT_r = 512;
NFFT_a = 256;

% Full range axis — complex signal, no half-cut
f_ax       = (0:NFFT_r-1) * (Fs/NFFT_r);
range_axis = (f_ax * c) / (2*S);

% Angle axis
omega_a    = linspace(-pi, pi, NFFT_a);
sin_theta  = (lambda * omega_a) / (2*pi*D);
valid_a    = abs(sin_theta) <= 1;
ang_axis   = NaN(1, NFFT_a);
ang_axis(valid_a) = rad2deg(asin(sin_theta(valid_a)));

%% =========================================================
%  SECTION 3: SHAPE DEFINITIONS
%
%  Each shape defined as [x_local, y_local, amplitude]
%  in local coordinates centred on the shape's centroid.
%  Size parameter controls the overall scale (metres).
%
%  Physical justification for amplitudes:
%    Corner reflector (90-deg angle) → amp = 1.0
%    Flat edge (specular return)     → amp = 0.60
%    Curved surface                  → amp = 0.35
%
%  All shapes sized at ~0.5m — visible at 4GHz bandwidth
%  (range res = 3.75cm, so features > 5cm are resolvable)
%% =========================================================

function sc = make_circle(radius, N_points)
    % Uniformly distributed points around circumference
    % No corners → uniform amplitude around the ring
    angles = linspace(0, 2*pi, N_points+1);
    angles = angles(1:end-1);   % remove duplicate endpoint
    x   = radius * cos(angles);
    y   = radius * sin(angles);
    amp = 0.35 * ones(N_points, 1);   % curved surface, moderate
    sc  = [x(:), y(:), amp];
end

function sc = make_square(side)
    % 4 strong corner reflectors + edge points
    h = side/2;
    N_edge = 6;   % points per edge (excluding corners)

    % Corners — very strong (corner reflector geometry)
    corners = [-h, -h, 1.00;
                h, -h, 1.00;
                h,  h, 1.00;
               -h,  h, 1.00];

    % Edge points — moderate amplitude
    e = linspace(-h, h, N_edge+2);
    e = e(2:end-1);   % exclude corners already defined

    bottom = [e(:),          -h*ones(N_edge,1), 0.60*ones(N_edge,1)];
    top    = [e(:),           h*ones(N_edge,1), 0.60*ones(N_edge,1)];
    left   = [-h*ones(N_edge,1), e(:),          0.60*ones(N_edge,1)];
    right  = [ h*ones(N_edge,1), e(:),          0.60*ones(N_edge,1)];

    sc = [corners; bottom; top; left; right];
end

function sc = make_rectangle(width, height)
    % Like square but different width and height
    % Asymmetry in range-angle signature encodes aspect ratio
    hw = width/2;
    hh = height/2;
    N_edge_w = 8;   % more points along longer dimension
    N_edge_h = 4;

    % Corners
    corners = [-hw, -hh, 1.00;
                hw, -hh, 1.00;
                hw,  hh, 1.00;
               -hw,  hh, 1.00];

    % Edge points
    ew = linspace(-hw, hw, N_edge_w+2); ew = ew(2:end-1);
    eh = linspace(-hh, hh, N_edge_h+2); eh = eh(2:end-1);

    bottom = [ew(:), -hh*ones(N_edge_w,1), 0.60*ones(N_edge_w,1)];
    top    = [ew(:),  hh*ones(N_edge_w,1), 0.60*ones(N_edge_w,1)];
    left   = [-hw*ones(N_edge_h,1), eh(:), 0.60*ones(N_edge_h,1)];
    right  = [ hw*ones(N_edge_h,1), eh(:), 0.60*ones(N_edge_h,1)];

    sc = [corners; bottom; top; left; right];
end

function sc = make_triangle(side)
    % Equilateral triangle — 3 corners + 3 edges
    % 3-peak pattern is uniquely triangular
    h_tri = side * sqrt(3)/2;
    cx    = 0;
    cy    = -h_tri/3;   % centroid offset

    % Vertices
    v1 = [  0,             2*h_tri/3 + cy];
    v2 = [ -side/2,       -h_tri/3  + cy];
    v3 = [  side/2,       -h_tri/3  + cy];

    corners = [v1, 1.00;
               v2, 1.00;
               v3, 1.00];

    N_edge = 6;
    t = linspace(0, 1, N_edge+2); t = t(2:end-1);

    % Interpolate points along each edge
    edge1 = [(1-t')*v1(1) + t'*v2(1),  (1-t')*v1(2) + t'*v2(2),  0.60*ones(N_edge,1)];
    edge2 = [(1-t')*v2(1) + t'*v3(1),  (1-t')*v2(2) + t'*v3(2),  0.60*ones(N_edge,1)];
    edge3 = [(1-t')*v3(1) + t'*v1(1),  (1-t')*v3(2) + t'*v1(2),  0.60*ones(N_edge,1)];

    sc = [corners; edge1; edge2; edge3];
end

% Instantiate shapes — all ~0.5m scale, well within radar resolution
shapes(1).name       = 'Circle';
shapes(1).label      = 1;
shapes(1).color      = [0.20 0.60 1.00];
shapes(1).scatterers = make_circle(0.25, 20);   % radius=0.25m, 20 points

shapes(2).name       = 'Square';
shapes(2).label      = 2;
shapes(2).color      = [1.00 0.40 0.10];
shapes(2).scatterers = make_square(0.50);        % 0.5m × 0.5m

shapes(3).name       = 'Rectangle';
shapes(3).label      = 3;
shapes(3).color      = [0.20 0.80 0.20];
shapes(3).scatterers = make_rectangle(0.70, 0.30); % 0.7m wide × 0.3m tall

shapes(4).name       = 'Triangle';
shapes(4).label      = 4;
shapes(4).color      = [0.80 0.20 0.80];
shapes(4).scatterers = make_triangle(0.50);      % equilateral, 0.5m side

N_classes = length(shapes);

fprintf('===== Shape Definitions =====\n');
for i = 1:N_classes
    fprintf('  %d. %-12s — %d scatterers\n', ...
            i, shapes(i).name, size(shapes(i).scatterers,1));
end
fprintf('=============================\n\n');

%% =========================================================
%  SECTION 4: HELPER — Place and Rotate Scatterers
%% =========================================================

function world_sc = place_shape(scatterers, rx, ry, angle_deg)
    R        = [cosd(angle_deg), -sind(angle_deg);
                sind(angle_deg),  cosd(angle_deg)];
    xy       = scatterers(:,1:2);
    amp      = scatterers(:,3);
    xy_rot   = (R * xy.').' ;
    world_xy = xy_rot + repmat([rx, ry], size(xy,1), 1);
    world_sc = [world_xy, amp];
end

%% =========================================================
%  SECTION 5: HELPER — Generate Range-Angle Map
%% =========================================================

function RA_map = generate_RA_map(world_sc, rp)
    t  = (0:rp.N_s-1).' / rp.Fs;
    IF = zeros(rp.N_s, rp.N_ant);

    for s = 1:size(world_sc,1)
        sx  = world_sc(s,1);
        sy  = world_sc(s,2);
        amp = world_sc(s,3);

        d     = sqrt(sx^2 + sy^2);
        theta = atan2d(sx, sy);

        if d < 0.1 || d > rp.d_max
            continue;
        end

        f_beat     = rp.S * (2*d) / rp.c;
        phi_r      = (4*pi*rp.fc*d) / rp.c;
        phi_a_base = (2*pi/rp.lambda) * rp.D * sind(theta);

        for n = 1:rp.N_ant
            phi_a  = phi_a_base * (n-1);
            signal = amp * exp(1j*(2*pi*f_beat*t + phi_r + phi_a));
            IF(:,n) = IF(:,n) + signal;
        end
    end

    % Noise
    noise_amp = 10^(-rp.snr_db/20);
    IF = IF + noise_amp*(randn(size(IF)) + 1j*randn(size(IF)));

    % Hanning window — suppresses range sidelobes
    win = hanning(rp.N_s);
    IF  = IF .* win;

    % Range FFT — full spectrum (complex signal)
    rng_fft = fft(IF, rp.NFFT_r, 1);

    % Angle FFT across antennas
    ang_fft = fftshift(fft(rng_fft, rp.NFFT_a, 2), 2);
    RA_map  = abs(ang_fft);   % [NFFT_r × NFFT_a]
end

%% =========================================================
%  SECTION 6: VISUALISE SHAPE GEOMETRIES
%  Show the actual scatterer layout of each shape
%  so you can see the physical model before the radar sees it
%% =========================================================

figure('Name','Shape Geometries — Scatterer Layouts', ...
       'NumberTitle','off','Color','w','Position',[50 500 1100 280]);

for i = 1:N_classes
    subplot(1, N_classes, i);
    sc = shapes(i).scatterers;
    % scatter(sc(:,1), sc(:,2), 80*sc(:,3)*200, ...
    %         sc(:,3), 'filled');
    for j = 1:size(sc,1)
    plot(sc(j,1), sc(j,2), 'o', ...
         'MarkerSize',  8, ...
         'MarkerFaceColor', [sc(j,3), 0.3, 1-sc(j,3)], ...
         'MarkerEdgeColor', 'k');
    hold on;
    end
    colormap(gca, 'hot');
    clim([0 1]);
    axis equal; grid on;
    xlabel('x (m)'); ylabel('y (m)');
    title(shapes(i).name, 'FontSize', 12, 'FontWeight', 'bold', ...
          'Color', shapes(i).color);
    xlim([-0.5 0.5]); ylim([-0.5 0.5]);
    cb = colorbar; cb.Label.String = 'Amplitude';

    % Annotate corner vs edge
    for j = 1:size(sc,1)
        if sc(j,3) >= 0.9
            text(sc(j,1)+0.03, sc(j,2)+0.03, 'C', ...
                 'FontSize',7,'Color','white','FontWeight','bold');
        end
    end
end
sgtitle('Physical Scatterer Models  |  C = Corner Reflector (strongest)', ...
        'FontSize',12,'FontWeight','bold');

%% =========================================================
%  SECTION 7: CANONICAL RANGE-ANGLE HEATMAPS
%  One per shape — 0° orientation, 4m range, 30dB SNR
%  Y-axis zoomed to object region for clarity
%% =========================================================

% Pack radar params
rp.c      = c;       rp.fc     = fc;     rp.lambda = lambda;
rp.S      = S;       rp.Fs     = Fs;     rp.N_s    = N_s;
rp.N_ant  = N_ant;   rp.D      = D;      rp.d_max  = d_max;
rp.NFFT_r = NFFT_r;  rp.NFFT_a = NFFT_a;
rp.snr_db = 10;

obj_range = 4.0;   % metres — safely within d_max = 6m
zoom_half = 0.8;   % metres to show above/below object
y_lo      = obj_range - zoom_half;
y_hi      = obj_range + zoom_half;

figure('Name','Canonical Range-Angle Heatmaps — All Shapes', ...
       'NumberTitle','off','Color','w','Position',[50 150 1300 380]);

for i = 1:N_classes
    world_sc = place_shape(shapes(i).scatterers, 0, obj_range, 0);
    RA       = generate_RA_map(world_sc, rp);
    RA_dB    = 20*log10(RA / max(RA(:)) + eps);

    subplot(1, N_classes, i);
    imagesc(ang_axis, range_axis, RA_dB);
    colormap('jet'); clim([-40 0]);
    set(gca,'YDir','normal');
    xlim([-60 60]);
    ylim([y_lo y_hi]);

    cb = colorbar; cb.Label.String = 'dB';
    xlabel('Angle (degrees)', 'FontSize', 10);
    ylabel('Range (m)',       'FontSize', 10);
    title(shapes(i).name, 'FontSize', 13, 'FontWeight', 'bold', ...
          'Color', shapes(i).color);

    hold on;
    plot(0, obj_range, 'w+', 'MarkerSize', 12, 'LineWidth', 2);
end

sgtitle(sprintf('Range-Angle Signatures  |  Range=%.0fm  |  0°  |  30dB SNR', obj_range), ...
        'FontSize', 12, 'FontWeight','bold');

% %% =========================================================
% %  SECTION 8: ORIENTATION EFFECT
% %  Rotate each shape through 0, 45, 90, 135 degrees
% %  Circle should be invariant. Others change dramatically.
% %% =========================================================
% 
% orientations = [0, 45, 90, 135];
% 
% figure('Name','Orientation Effect — All Shapes', ...
%        'NumberTitle','off','Color','w','Position',[50 50 1300 780]);
% 
% for i = 1:N_classes
%     for j = 1:4
%         subplot(N_classes, 4, (i-1)*4 + j);
% 
%         world_sc = place_shape(shapes(i).scatterers, 0, obj_range, orientations(j));
%         RA       = generate_RA_map(world_sc, rp);
%         RA_dB    = 20*log10(RA / max(RA(:)) + eps);
% 
%         imagesc(ang_axis, range_axis, RA_dB);
%         colormap('jet'); clim([-40 0]);
%         set(gca,'YDir','normal');
%         xlim([-60 60]);
%         ylim([y_lo y_hi]);
% 
%         if j == 1
%             ylabel(sprintf('%s\nRange(m)', shapes(i).name), ...
%                    'FontSize',9,'Color',shapes(i).color,'FontWeight','bold');
%         else
%             ylabel('');
%         end
%         if i == N_classes
%             xlabel('Angle (deg)','FontSize',9);
%         end
%         title(sprintf('%d°', orientations(j)),'FontSize',10);
%         hold on;
%         plot(0, obj_range, 'w+','MarkerSize',8,'LineWidth',1.8);
%     end
% end
% 
% sgtitle('How Shape Signatures Change with Orientation', ...
%         'FontSize',13,'FontWeight','bold');

%% =========================================================
%  SECTION 9: DATASET GENERATION
%
%  Variation axes:
%    Orientations : 0° to 350° in 10° steps  → 36
%    Ranges       : 2, 3, 4, 5 m             → 4
%    Lateral pos  : -1, 0, +1 m              → 3
%    SNR levels   : 15, 25, 35 dB            → 3
%
%  Total: 36 × 4 × 3 × 3 = 1,296 per class
%         1,296 × 4 classes = 5,184 total
%% =========================================================

orientations_ds = 0 : 10 : 350;
ranges_ds       = [2.0, 3.0, 4.0, 5.0];
lateral_ds      = [-1.0, 0.0, 1.0];
snr_ds          = [5, 10, 15];

samples_per_class = length(orientations_ds) * length(ranges_ds) * ...
                    length(lateral_ds) * length(snr_ds);
total_samples     = samples_per_class * N_classes;

fprintf('===== Dataset Generation =====\n');
fprintf('Orientations : %d  (0 to 350 deg, step 10)\n', length(orientations_ds));
fprintf('Ranges       : %d  (2, 3, 4, 5 m)\n', length(ranges_ds));
fprintf('Lateral pos  : %d  (-1, 0, +1 m)\n', length(lateral_ds));
fprintf('SNR levels   : %d  (15, 25, 35 dB)\n', length(snr_ds));
fprintf('Per class    : %d samples\n', samples_per_class);
fprintf('Total        : %d samples\n', total_samples);
fprintf('==============================\n\n');

% Pre-allocate
X = zeros(total_samples, NFFT_r, NFFT_a, 'single');
Y = zeros(total_samples, 1, 'uint8');

sample_idx = 0;
tic;

for i = 1:N_classes
    shape = shapes(i);
    fprintf('Generating class %d (%s)...', shape.label, shape.name);

    for ori = orientations_ds
        for rng = ranges_ds
            for lat = lateral_ds
                for snr = snr_ds

                    % World position
                    rx = lat;
                    ry = sqrt(max(rng^2 - lat^2, 0.01));

                    rp.snr_db = snr;
                    world_sc  = place_shape(shape.scatterers, rx, ry, ori);
                    RA        = generate_RA_map(world_sc, rp);

                    % Normalise to [0,1]
                    RA = single(RA / (max(RA(:)) + eps));

                    sample_idx = sample_idx + 1;
                    X(sample_idx,:,:) = RA;
                    Y(sample_idx)     = uint8(shape.label);
                end
            end
        end
    end
    fprintf(' done (%d samples)\n', samples_per_class);
end

elapsed = toc;
fprintf('\nGenerated %d samples in %.1f seconds\n\n', sample_idx, elapsed);

%% =========================================================
%  SECTION 10: TRAIN / VAL / TEST SPLIT
%  Stratified 70 / 15 / 15
%% =========================================================

fprintf('===== Splitting Dataset =====\n');

train_ratio = 0.70;
val_ratio   = 0.15;

train_idx = [];
val_idx   = [];
test_idx  = [];

randstate = RandStream('mt19937ar','Seed',42);

for cls = 1:N_classes
    cls_idx = find(Y == cls);
    cls_idx = cls_idx(randperm(randstate, length(cls_idx)));

    n       = length(cls_idx);
    n_train = round(n * train_ratio);
    n_val   = round(n * val_ratio);

    train_idx = [train_idx; cls_idx(1:n_train)];
    val_idx   = [val_idx;   cls_idx(n_train+1 : n_train+n_val)];
    test_idx  = [test_idx;  cls_idx(n_train+n_val+1 : end)];
end

X_train = X(train_idx,:,:);  Y_train = Y(train_idx);
X_val   = X(val_idx,  :,:);  Y_val   = Y(val_idx);
X_test  = X(test_idx, :,:);  Y_test  = Y(test_idx);

fprintf('Train : %d samples\n', length(train_idx));
fprintf('Val   : %d samples\n', length(val_idx));
fprintf('Test  : %d samples\n', length(test_idx));

%% =========================================================
%  SECTION 11: SAVE DATASET
%% =========================================================

% Class name map — for Python loading
class_names = {shapes.name};

save('radar_shapes_dataset.mat', ...
     'X_train','Y_train', ...
     'X_val',  'Y_val',   ...
     'X_test', 'Y_test',  ...
     'range_axis','ang_axis', ...
     'class_names','N_classes', ...
     'NFFT_r','NFFT_a');

fprintf('\n===== Saved: radar_shapes_dataset.mat =====\n');
fprintf('X shape : [N x %d x %d]  (Range-Angle maps)\n', NFFT_r, NFFT_a);
fprintf('Y shape : [N x 1]         (labels: 1=Circle 2=Square 3=Rectangle 4=Triangle)\n');
fprintf('\nClass map:\n');
for i = 1:N_classes
    fprintf('  %d = %s\n', shapes(i).label, shapes(i).name);
end
fprintf('===========================================\n\n');

% %% =========================================================
% %  SECTION 12: DATASET STATISTICS PLOT
% %% =========================================================
% 
% figure('Name','Dataset Statistics','NumberTitle','off', ...
%        'Color','w','Position',[400 300 600 380]);
% 
% counts = histcounts(Y, 1:N_classes+1);
% clrs   = vertcat(shapes.color);
% b = bar(1:N_classes, counts, 'FaceColor','flat');
% b.CData = clrs;
% xticks(1:N_classes);
% xticklabels({shapes.name});
% ylabel('Samples'); title(sprintf('Dataset: %d Total | %d Classes', total_samples, N_classes));
% grid on;
% for i = 1:N_classes
%     text(i, counts(i)+5, num2str(counts(i)), ...
%          'HorizontalAlignment','center','FontWeight','bold','FontSize',10);
% end

%% =========================================================
%  SECTION 13: PYTHON LOADING INSTRUCTIONS (printed)
%% =========================================================

fprintf('===== Loading in Python / PyTorch =====\n');
fprintf('\nimport scipy.io\n');
fprintf('import numpy as np\n');
fprintf('import torch\n\n');
fprintf('data    = scipy.io.loadmat(''radar_shapes_dataset.mat'')\n');
fprintf('X_train = data[''X_train'']   # [N x %d x %d]\n', NFFT_r, NFFT_a);
fprintf('Y_train = data[''Y_train''].flatten() - 1  # 0-indexed for PyTorch\n\n');
fprintf('# Add channel dim for CNN: [N x 1 x %d x %d]\n', NFFT_r, NFFT_a);
fprintf('X_train = np.expand_dims(X_train, axis=1)\n');
fprintf('X_train_t = torch.tensor(X_train, dtype=torch.float32)\n');
fprintf('Y_train_t = torch.tensor(Y_train, dtype=torch.long)\n');
fprintf('=======================================\n');
