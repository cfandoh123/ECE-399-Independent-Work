%% =========================================================
%  radar_stl_to_heatmap.m
%  Load STL shape from Tinkercad → Generate Radar Heatmap
%
%  Prerequisites:
%    generate_RA.m must be in the same folder as this script
%
%  Usage:
%    1. Export your shape from Tinkercad as .stl
%    2. Set STL_FILE and TRUE_SHAPE below
%    3. Run this script
%    4. Run: python predict_toolbox_sample.py --file stl_heatmap.mat
%
%  Project: Radar Object Detection + ML (Junior IW)
%% =========================================================

clear; clc; close all;

%% =========================================================
%  SECTION 1: SETTINGS — change these for each shape
%% =========================================================

STL_FILE        = 'circle_400mm.stl';   % your exported STL file
TRUE_SHAPE      = 'circle';             % for labelling only
OBJ_RANGE       = 4.0;                  % metres from radar
OBJ_ORIENTATION = 0;                    % degrees rotation
MIN_SPACING     = 0.065;                 % min 2cm between scatterers
AMPLITUDE       = 0.50;                  % all metal

%% =========================================================
%  SECTION 2: RADAR PARAMETERS
%  Must match radar_shape_simple.m exactly
%% =========================================================

c      = 3e8;
fc     = 77e9;
lambda = c / fc;
B      = 4e9;
Tc     = 40e-6;
S      = B / Tc;
Fs     = 4e6;
N_s    = round(Fs * Tc);
N_ant  = 16;
D      = lambda / 2;
d_max  = (Fs * c) / (2 * S);
NFFT_r = 512;
NFFT_a = 256;

rp.c=c; rp.fc=fc; rp.lambda=lambda; rp.S=S; rp.Fs=Fs;
rp.N_s=N_s; rp.N_ant=N_ant; rp.D=D; rp.d_max=d_max;
rp.NFFT_r=NFFT_r; rp.NFFT_a=NFFT_a; rp.snr_db=20;

%% =========================================================
%  SECTION 3: LOAD STL AND EXTRACT SCATTERERS
%% =========================================================

if ~exist(STL_FILE, 'file')
    error(['STL file not found: %s\n' ...
           'Export your shape from Tinkercad as .stl\n' ...
           'and place it in the same folder as this script.'], ...
           STL_FILE);
end

TR    = stlread(STL_FILE);
verts = TR.Points / 1000;   % mm → metres
faces = TR.ConnectivityList;

fprintf('STL loaded: %s\n', STL_FILE);
fprintf('  Raw vertices : %d\n', size(verts,1));
fprintf('  Faces        : %d\n', size(faces,1));
fprintf('  X range      : %.3f to %.3f m\n', min(verts(:,1)), max(verts(:,1)));
fprintf('  Y range      : %.3f to %.3f m\n', min(verts(:,2)), max(verts(:,2)));
fprintf('  Z range      : %.3f to %.3f m\n\n', min(verts(:,3)), max(verts(:,3)));

%% =========================================================
%  SECTION 4: DENSIFY THE MESH
%
%  A simple box STL has only 8 corner vertices.
%  We add midpoints along every triangle edge to get
%  a dense set of surface points that represent the
%  shape boundary properly.
%
%  The densification loop runs multiple passes until
%  we have enough points to cover the shape outline.
%% =========================================================

fprintf('Densifying mesh...\n');

dense_pts = verts;

% Run 4 densification passes — each pass adds edge midpoints
for pass = 1:4
    new_pts = dense_pts;
    for f = 1:size(faces,1)
        v1 = dense_pts(faces(f,1),:);
        v2 = dense_pts(faces(f,2),:);
        v3 = dense_pts(faces(f,3),:);
        new_pts = [new_pts;
                   (v1+v2)/2;
                   (v2+v3)/2;
                   (v1+v3)/2];
    end
    % Remove duplicates after each pass
    dense_pts = unique(round(new_pts, 4), 'rows');
end

fprintf('  After densification: %d points\n', size(dense_pts,1));

%% SECTION 5 (REPLACE ENTIRELY): EXTRACT BOUNDARY VIA CONVEX HULL

% Project all points to 2D top face
z_vals  = dense_pts(:,3);
z_top   = max(z_vals);
top_pts = dense_pts(abs(z_vals - z_top) < 0.002, 1:2);

% Compute convex hull — gives indices of boundary points
try
    k = convhull(top_pts(:,1), top_pts(:,2));
    boundary_pts = top_pts(k, :);
    fprintf('  Convex hull boundary: %d points\n', size(boundary_pts,1));
catch
    warning('convhull failed — using all top points');
    boundary_pts = top_pts;
end

% Densify the boundary by interpolating along each edge
% This ensures enough points regardless of original mesh density
N_per_edge = 8;   % interpolation points per hull edge
xy_pts     = [];

for i = 1:size(boundary_pts,1)-1
    p1 = boundary_pts(i,   :);
    p2 = boundary_pts(i+1, :);
    t  = linspace(0, 1, N_per_edge)';
    segment = (1-t)*p1 + t*p2;
    xy_pts  = [xy_pts; segment];
end

xy_pts = unique(round(xy_pts, 4), 'rows');
fprintf('  After boundary interpolation: %d points\n', size(xy_pts,1));
%% =========================================================
%  SECTION 6: FILTER BY MINIMUM SPACING
%
%  Remove points that are too close together.
%  This prevents oversampling the flat interior
%  and keeps scatterers at meaningful separations
%  relative to radar range resolution (3.75 cm).
%% =========================================================

filtered = xy_pts(1,:);
for i = 2:size(xy_pts,1)
    dists = sqrt(sum((filtered - xy_pts(i,:)).^2, 2));
    if min(dists) > MIN_SPACING
        filtered = [filtered; xy_pts(i,:)];
    end
end

% Assign uniform metal amplitude to all scatterers
sc_local = [filtered, AMPLITUDE * ones(size(filtered,1),1)];

fprintf('  After spacing filter: %d scatterers\n\n', size(sc_local,1));

if size(sc_local,1) < 4
    warning(['Very few scatterers (%d). The heatmap may be sparse.\n' ...
             'Try reducing MIN_SPACING to 0.01.'], size(sc_local,1));
end

%% =========================================================
%  SECTION 7: VISUALISE SCATTERER LAYOUT
%  Inspect this before generating the heatmap.
%  Should look like the outline of your shape.
%% =========================================================

figure('Name','STL Scatterer Layout','NumberTitle','off', ...
       'Color','w','Position',[100 400 500 450]);

scatter(sc_local(:,1), sc_local(:,2), ...
        60, sc_local(:,3)*200, 'filled');
colormap('hot'); colorbar;
axis equal; grid on;
xlabel('x (m)'); ylabel('y (m)');
title(sprintf('Scatterers from STL: %s  (%d points)', ...
              TRUE_SHAPE, size(sc_local,1)));
xlim([min(sc_local(:,1))-0.05, max(sc_local(:,1))+0.05]);
ylim([min(sc_local(:,2))-0.05, max(sc_local(:,2))+0.05]);

%% =========================================================
%  SECTION 8: ROTATE AND PLACE IN WORLD COORDINATES
%% =========================================================

R_mat    = [cosd(OBJ_ORIENTATION), -sind(OBJ_ORIENTATION);
            sind(OBJ_ORIENTATION),  cosd(OBJ_ORIENTATION)];
xy_rot   = (R_mat * sc_local(:,1:2).').' ;
sc_world = [xy_rot(:,1) + 0, ...
            xy_rot(:,2) + OBJ_RANGE, ...
            sc_local(:,3)];

fprintf('Placed at: range=%.1fm  orientation=%ddeg\n\n', ...
        OBJ_RANGE, OBJ_ORIENTATION);

%% =========================================================
%  SECTION 9: GENERATE RANGE-ANGLE HEATMAP
%  Calls generate_RA.m — must be in the same folder
%% =========================================================

fprintf('Generating heatmap...\n');
RA            = generate_RA(sc_world, rp);
RA_normalised = RA / (max(RA(:)) + eps);
fprintf('Done. Heatmap shape: [%d x %d]\n\n', size(RA_normalised));

%% =========================================================
%  SECTION 10: BUILD AXES AND VISUALISE HEATMAP
%% =========================================================

f_ax       = (0:NFFT_r-1)*(Fs/NFFT_r);
range_axis = (f_ax*c)/(2*S);

omega_a   = linspace(-pi, pi, NFFT_a);
sin_theta = (lambda * omega_a)/(2*pi*D);
valid_a   = abs(sin_theta) <= 1;
ang_axis  = NaN(1, NFFT_a);
ang_axis(valid_a) = rad2deg(asin(sin_theta(valid_a)));

zoom_half = 0.8;
y_lo      = OBJ_RANGE - zoom_half;
y_hi      = OBJ_RANGE + zoom_half;

figure('Name',sprintf('STL Heatmap: %s',TRUE_SHAPE), ...
       'NumberTitle','off','Color','w','Position',[100 100 950 420]);

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
title(sprintf('Zoomed | %s | %ddeg | %.1fm', ...
              TRUE_SHAPE, OBJ_ORIENTATION, OBJ_RANGE));
xlim([-60 60]); ylim([y_lo y_hi]);
hold on;
plot(0, OBJ_RANGE, 'w+', 'MarkerSize', 14, 'LineWidth', 2.5);

sgtitle(sprintf('STL → Radar Heatmap  |  %s  |  %d scatterers', ...
        upper(TRUE_SHAPE), size(sc_local,1)), ...
        'FontSize',12,'FontWeight','bold');

%% =========================================================
%  SECTION 11: SAVE FOR PYTHON PREDICTION
%% =========================================================

output_file = sprintf('stl_heatmap_%s_%ddeg.mat', ...
                       TRUE_SHAPE, OBJ_ORIENTATION);

RA_map = RA_normalised;

save(output_file, ...
     'RA_normalised', ...
     'RA_map',        ...  % alias
     'range_axis',    ...
     'ang_axis',      ...
     'TRUE_SHAPE',    ...
     'OBJ_RANGE',     ...
     'OBJ_ORIENTATION', ...
     'N_ant');

% Also save RA_map for compatibility

save(output_file, 'RA_map', '-append');

fprintf('===== Saved =====\n');
fprintf('File : %s\n', output_file);
fprintf('\nPredict with:\n');
fprintf('  python predict_toolbox_sample.py --file %s\n', output_file);
fprintf('=================\n');
