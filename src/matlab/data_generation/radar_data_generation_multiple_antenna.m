%% =========================================================
%  radar_data_generation_multiple_antenna.m
%  FINALIZED Radar Shape Classification Dataset Generator
%  (Includes Sim-to-Real Domain Randomization, Jitter per Sample,
%   Reproducibility, and Per-Sample AGC Normalization)
%% =========================================================

clear; clc; close all;

INTERACTIVE_MODE = false; 

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
antenna_counts = [4, 8, 10, 12, 16, 32];
N_ant    = antenna_counts(3);   
D        = lambda / 2;
d_max    = (Fs * c) / (2 * S);
d_res    = c / (2 * B);

fprintf('===== Radar Parameters =====\n');
fprintf('Range res  : %.2f cm\n', d_res * 100);
fprintf('Max range  : %.2f m\n',  d_max);
fprintf('============================\n\n');

%% =========================================================
%  SECTION 2: SHAPE DEFINITIONS (Fixed Scoping & Density)
%  Base amplitudes only. Jitter is applied per-sample later.
%% =========================================================

function sc = make_circle(radius)
    SPACING = 0.015; % 1.5 cm spacing to avoid spatial aliasing
    N   = max(24, round(2*pi*radius / SPACING)); 
    ang = linspace(0, 2*pi, N+1); ang = ang(1:end-1);
    x   = radius * cos(ang(:));
    y   = radius * sin(ang(:));
    sc  = [x, y, 0.50 * ones(N,1)];
end

function sc = make_square(side)
    SPACING = 0.015;
    h      = side / 2;
    N_edge = max(5, round(side / SPACING));
    e      = linspace(-h, h, N_edge+2); e = e(2:end-1);

    corners = [-h,-h,1.0; h,-h,1.0; h,h,1.0; -h,h,1.0];
    bottom  = [e(:), -h*ones(N_edge,1), 0.70*ones(N_edge,1)];
    top     = [e(:),  h*ones(N_edge,1), 0.70*ones(N_edge,1)];
    left    = [-h*ones(N_edge,1), e(:), 0.70*ones(N_edge,1)];
    right   = [ h*ones(N_edge,1), e(:), 0.70*ones(N_edge,1)];
    sc      = [corners; bottom; top; left; right];
end

function sc = make_rectangle(width, height)
    SPACING = 0.015;
    hw     = width  / 2;
    hh     = height / 2;
    N_w    = max(8, round(width / SPACING));
    N_h    = max(4, round(height / SPACING));

    ew = linspace(-hw, hw, N_w+2); ew = ew(2:end-1);
    eh = linspace(-hh, hh, N_h+2); eh = eh(2:end-1);

    corners = [-hw,-hh,1.0; hw,-hh,1.0; hw,hh,1.0; -hw,hh,1.0];
    bottom  = [ew(:), -hh*ones(N_w,1), 0.70*ones(N_w,1)];
    top     = [ew(:),  hh*ones(N_w,1), 0.70*ones(N_w,1)];
    left    = [-hw*ones(N_h,1), eh(:), 0.70*ones(N_h,1)];
    right   = [ hw*ones(N_h,1), eh(:), 0.70*ones(N_h,1)];
    sc      = [corners; bottom; top; left; right];
end

function sc = make_triangle(side)
    SPACING = 0.015;
    ht  = side * sqrt(3) / 2;
    v1  = [0,       2*ht/3];    
    v2  = [-side/2, -ht/3];    
    v3  = [ side/2, -ht/3];    

    corners = [v1,1.0; v2,1.0; v3,1.0];

    N_e = max(6, round(side / SPACING));
    t   = linspace(0,1,N_e+2); t = t(2:end-1);

    e1 = [(1-t')*v1(1)+t'*v2(1), (1-t')*v1(2)+t'*v2(2), 0.70*ones(N_e,1)];
    e2 = [(1-t')*v2(1)+t'*v3(1), (1-t')*v2(2)+t'*v3(2), 0.70*ones(N_e,1)];
    e3 = [(1-t')*v3(1)+t'*v1(1), (1-t')*v3(2)+t'*v1(2), 0.70*ones(N_e,1)];
    sc = [corners; e1; e2; e3];
end

function sc = make_oval(width, height)
    SPACING = 0.015;
    a  = width  / 2;   
    b  = height / 2;   
    perim = pi * (3*(a+b) - sqrt((3*a + b)*(a + 3*b)));
    N   = max(24, round(perim / SPACING)); 
    
    ang = linspace(0, 2*pi, N+1); ang = ang(1:end-1);
    x   = a * cos(ang(:));
    y   = b * sin(ang(:));

    kappa = (a*b) ./ ((b*cos(ang(:))).^2 + (a*sin(ang(:))).^2).^1.5;
    kappa_norm = kappa / max(kappa);
    amp   = 0.35 + 0.40 * kappa_norm;   

    sc = [x, y, amp];
end

%% =========================================================
%  SECTION 3: SIZE VARIANTS
%% =========================================================
sizes_label = {'Small (0.20m)', 'Medium (0.40m)', 'Large (0.60m)'};
sizes_val   = [0.20, 0.40, 0.60];   
shape_names = {'Circle', 'Square', 'Rectangle', 'Triangle', 'Oval'};
N_classes   = 5;

shape_scatterers = cell(N_classes, length(sizes_val));

for si = 1:length(sizes_val)
    s = sizes_val(si);
    shape_scatterers{1,si} = make_circle(s/2);             
    shape_scatterers{2,si} = make_square(s);                
    shape_scatterers{3,si} = make_rectangle(s, s/2);        
    shape_scatterers{4,si} = make_triangle(s);              
    shape_scatterers{5,si} = make_oval(s, s/1.8);           
end

%% =========================================================
%  SECTION 4: FFT AXES
%% =========================================================
NFFT_r = 512;
NFFT_a = 256;
f_ax       = (0:NFFT_r-1) * (Fs/NFFT_r);
range_axis = (f_ax * c) / (2*S);

%% =========================================================
%  SECTION 5: GENERATE RANGE-ANGLE MAP 
%  (Physics Fixes: Clutter, Phase Noise)
%% =========================================================
function RA = generate_RA(world_sc, rp)
    t  = (0:rp.N_s-1).' / rp.Fs;
    IF = zeros(rp.N_s, rp.N_ant);

    N_clutter = 8;
    clutter_sc = [(rand(N_clutter,1)-0.5)*4, (rand(N_clutter,1)*rp.d_max*0.8), rand(N_clutter,1)*0.05];
    all_sc = [world_sc; clutter_sc];

    for s = 1:size(all_sc,1)
        sx  = all_sc(s,1);
        sy  = all_sc(s,2);
        d   = sqrt(sx^2 + sy^2);
        th  = atan2d(sx, sy);

        if d < 0.1 || d > rp.d_max; continue; end

        % Simulated Hardware Range-Gain Compensation (No 1/R^2 applied)
        amp    = all_sc(s,3);
        f_b    = rp.S * 2*d / rp.c;
        phi_r  = 4*pi*rp.fc*d / rp.c;
        phi_ab = (2*pi/rp.lambda) * rp.D * sind(th);
        
        phase_err = 0.05 * randn(); 

        for n = 1:rp.N_ant
            sig = amp * exp(1j*(2*pi*f_b*t + phi_r + phi_ab*(n-1) + phase_err));
            IF(:,n) = IF(:,n) + sig;
        end
    end

    sig_power = var(IF(:));
    if sig_power == 0; sig_power = 1e-12; end
    req_snr_lin = 10^(rp.snr_db / 10);
    noise_power = sig_power / req_snr_lin;
    noise_amp = sqrt(noise_power / 2); 
    
    IF = IF + noise_amp * (randn(size(IF)) + 1j*randn(size(IF)));

    win     = hanning(rp.N_s);
    rng_fft = fft(IF .* win, rp.NFFT_r, 1);
    ang_fft = fftshift(fft(rng_fft, rp.NFFT_a, 2), 2);

    RA = abs(ang_fft);            
end

%% =========================================================
%  SECTION 6: CANONICAL HEATMAP VISUALIZATION
%% =========================================================
if INTERACTIVE_MODE
    fprintf('Interactive mode on. Inspecting heatmaps...\n');
    pause;
end

%% =========================================================
%  SECTION 7: FULL DATASET GENERATION
%% =========================================================
rp.c=c; rp.fc=fc; rp.lambda=lambda; rp.S=S; rp.Fs=Fs;
rp.N_s=N_s; rp.D=D; rp.d_max=d_max;
rp.NFFT_r=NFFT_r; rp.NFFT_a=NFFT_a;

for ant_exp = 1:length(antenna_counts)

    N_ant     = antenna_counts(ant_exp);
    rp.N_ant  = N_ant;

    omega_a   = linspace(-pi, pi, NFFT_a);
    sin_theta = (lambda * omega_a) / (2*pi*D);
    valid_a   = abs(sin_theta) <= 1;
    ang_axis  = NaN(1, NFFT_a);
    ang_axis(valid_a) = rad2deg(asin(sin_theta(valid_a)));

    fprintf('\n========================================\n');
    fprintf('  ANTENNA EXPERIMENT: N_ant = %d\n', N_ant);
    fprintf('========================================\n');

    snr_levels      = [5, 10, 20, 30];  
    orientations_ds = 0 : 10 : 350;     % Reverted to 10-degree steps for diversity
    ranges_ds       = [2.0, 3.0, 4.0, 5.0];   
    lateral_ds      = [-0.5, 0.0, 0.5]; 

    samples_per_shape = length(orientations_ds) * length(ranges_ds) * ...
                        length(sizes_val) * length(snr_levels) * length(lateral_ds);
    total_samples     = samples_per_shape * N_classes;

    X = zeros(total_samples, NFFT_r, NFFT_a, 'single');
    Y = zeros(total_samples, 1, 'uint8');
    
    config_labels = zeros(total_samples, 1); 
    
    sample_idx = 0;
    config_idx = 0;
    
    rng(42); % Fix reproducibility across generation runs
    tic;

    for shape_i = 1:N_classes
        fprintf('Generating %s', shape_names{shape_i});

        for size_i = 1:length(sizes_val)
            sc_local = shape_scatterers{shape_i, size_i};

            for ori = orientations_ds
                for lat = lateral_ds
                    for rng_idx = ranges_ds
                        % New Physical Geometry = New Configuration Group
                        config_idx = config_idx + 1; 

                        R_mat  = [cosd(ori), -sind(ori); sind(ori), cosd(ori)];
                        xy_rot = (R_mat * sc_local(:,1:2).').' ;

                        for snr = snr_levels
                            world_sc = [xy_rot(:,1) + lat, xy_rot(:,2) + rng_idx, sc_local(:,3)];
                            
                            % Apply Unique Per-Sample Amplitude Jitter Here
                            world_sc(:,3) = max(0.01, world_sc(:,3) + 0.1 * randn(size(world_sc(:,3))));
                            
                            rp.snr_db = snr;

                            RA = generate_RA(world_sc, rp);
                            
                            % Per-Sample AGC Normalization
                            RA = single(RA / (max(RA(:)) + eps));

                            sample_idx = sample_idx + 1;
                            X(sample_idx, :, :) = RA;
                            Y(sample_idx)       = uint8(shape_i);
                            config_labels(sample_idx) = config_idx; 
                        end
                    end
                end
            end
        end
        fprintf(' — %d samples done\n', samples_per_shape);
    end

    % Trim Arrays to avoid Zero-Entry Splitting Bug
    X = X(1:sample_idx, :, :);
    Y = Y(1:sample_idx);
    config_labels = config_labels(1:sample_idx);

    elapsed = toc;
    fprintf('Total: %d samples in %.1f seconds\n\n', sample_idx, elapsed);

    %% =========================================================
    %  SECTION 8: LEAKAGE-FREE TRAIN / VAL / TEST SPLIT
    %% =========================================================
    fprintf('===== Splitting Dataset (Grouped by Geometry) =====\n');

    rs = RandStream('mt19937ar', 'Seed', 42);
    train_idx = []; val_idx = []; test_idx = [];

    for cls = 1:N_classes
        cls_configs = unique(config_labels(Y == cls));
        cls_configs = cls_configs(randperm(rs, length(cls_configs)));
        
        n_cfg = length(cls_configs);
        n_tr = round(n_cfg * 0.70);
        n_vl = round(n_cfg * 0.15);
        
        tr_cfgs = cls_configs(1:n_tr);
        vl_cfgs = cls_configs(n_tr+1 : n_tr+n_vl);
        ts_cfgs = cls_configs(n_tr+n_vl+1 : end);

        train_idx = [train_idx; find(ismember(config_labels, tr_cfgs))];
        val_idx   = [val_idx;   find(ismember(config_labels, vl_cfgs))];
        test_idx  = [test_idx;  find(ismember(config_labels, ts_cfgs))];
    end

    X_train = X(train_idx, :, :);  Y_train = Y(train_idx);
    X_val   = X(val_idx,   :, :);  Y_val   = Y(val_idx);
    X_test  = X(test_idx,  :, :);  Y_test  = Y(test_idx);

    %% =========================================================
    %  SECTION 9: SAVE
    %% =========================================================
    class_names = shape_names;
    filename = sprintf('radar_shapes_N%d.mat', N_ant);
    save(filename, ...
         'X_train', 'Y_train', 'X_val', 'Y_val', 'X_test', 'Y_test', ...
         'range_axis', 'ang_axis', 'class_names', 'N_classes', ...
         'NFFT_r', 'NFFT_a', 'N_ant', 'sizes_val', 'orientations_ds', ...
         'ranges_ds', 'lateral_ds', '-v7.3');
    
    fprintf('Saved: %s\n', filename);
end

fprintf('\nAll antenna experiments complete.\n');
fprintf('Files saved: radar_shapes_N4.mat, N8.mat, N10.mat, N12.mat, N16.mat, N32.mat\n');