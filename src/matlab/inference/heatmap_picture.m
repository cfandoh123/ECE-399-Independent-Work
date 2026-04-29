%% =========================================================
%  Generate Canonical Heatmaps — All 5 Shapes × 6 Antenna
%  Counts in a Single Figure (6 rows × 5 columns)
%% =========================================================

clear; clc; close all;

%% RADAR PARAMETERS
c      = 3e8;       fc     = 77e9;    lambda = c/fc;
B      = 4e9;       Tc     = 40e-6;   S      = B/Tc;
Fs     = 4e6;       N_s    = round(Fs*Tc);
D      = lambda/2;  d_max  = (Fs*c)/(2*S);
NFFT_r = 512;       NFFT_a = 256;

%% SHAPE DEFINITIONS (medium size, all metal)
s = 0.40;

ang_c  = linspace(0,2*pi,25); ang_c=ang_c(1:end-1);
circle = [s/2*cos(ang_c(:)), s/2*sin(ang_c(:)), 0.50*ones(24,1)];

h=s/2; e=linspace(-h,h,7); e=e(2:end-1);
square = [-h,-h,1.0; h,-h,1.0; h,h,1.0; -h,h,1.0;
           e(:),-h*ones(5,1),0.70*ones(5,1);
           e(:), h*ones(5,1),0.70*ones(5,1);
          -h*ones(5,1),e(:),0.70*ones(5,1);
           h*ones(5,1),e(:),0.70*ones(5,1)];

hw=s/2; hh=s/4;
ew=linspace(-hw,hw,10); ew=ew(2:end-1);
eh=linspace(-hh,hh,6);  eh=eh(2:end-1);
rect = [-hw,-hh,1.0; hw,-hh,1.0; hw,hh,1.0; -hw,hh,1.0;
         ew(:),-hh*ones(8,1),0.70*ones(8,1);
         ew(:), hh*ones(8,1),0.70*ones(8,1);
        -hw*ones(4,1),eh(:),0.70*ones(4,1);
         hw*ones(4,1),eh(:),0.70*ones(4,1)];

ht=s*sqrt(3)/2;
v1=[0,2*ht/3]; v2=[-s/2,-ht/3]; v3=[s/2,-ht/3];
t=linspace(0,1,8); t=t(2:end-1);
tri = [v1,1.0; v2,1.0; v3,1.0;
       (1-t')*v1(1)+t'*v2(1),(1-t')*v1(2)+t'*v2(2),0.70*ones(6,1);
       (1-t')*v2(1)+t'*v3(1),(1-t')*v2(2)+t'*v3(2),0.70*ones(6,1);
       (1-t')*v3(1)+t'*v1(1),(1-t')*v3(2)+t'*v1(2),0.70*ones(6,1)];

a=s/2; b=s/3.6;
ang_o=linspace(0,2*pi,25); ang_o=ang_o(1:end-1);
kappa=(a*b)./((b*cos(ang_o(:))).^2+(a*sin(ang_o(:))).^2).^1.5;
oval=[a*cos(ang_o(:)),b*sin(ang_o(:)),0.35+0.40*(kappa/max(kappa))];

shapes       = {circle, square, rect, tri, oval};
shape_names  = {'Circle','Square','Rectangle','Triangle','Oval'};
shape_colors = {[0.2 0.6 1.0],[1.0 0.4 0.1],[0.2 0.8 0.2],...
                [0.8 0.2 0.8],[1.0 0.8 0.0]};
N_shapes     = 5;

%% SETTINGS
antenna_counts = [4, 8, 10, 12, 16, 32];
N_ant_list     = length(antenna_counts);
obj_range      = 4.0;
zoom_half      = 0.8;
y_lo           = obj_range - zoom_half;
y_hi           = obj_range + zoom_half;
snr_db         = 25;

%% SINGLE FIGURE — rows = antenna counts, cols = shapes
% Each cell is one heatmap
% Size: wide enough for 5 columns, tall enough for 6 rows
figure('Name','All Heatmaps — Shapes vs Antenna Count', ...
       'NumberTitle','off','Color','w', ...
       'Position',[10 10 1800 1100]);

for ant_idx = 1:N_ant_list

    N_ant = antenna_counts(ant_idx);

    rp.c=c; rp.fc=fc; rp.lambda=lambda; rp.S=S; rp.Fs=Fs;
    rp.N_s=N_s; rp.N_ant=N_ant; rp.D=D; rp.d_max=d_max;
    rp.NFFT_r=NFFT_r; rp.NFFT_a=NFFT_a; rp.snr_db=snr_db;

    omega_a   = linspace(-pi,pi,NFFT_a);
    sin_theta = (lambda*omega_a)/(2*pi*D);
    valid_a   = abs(sin_theta)<=1;
    ang_axis  = NaN(1,NFFT_a);
    ang_axis(valid_a) = rad2deg(asin(sin_theta(valid_a)));

    f_ax       = (0:NFFT_r-1)*(Fs/NFFT_r);
    range_axis = (f_ax*c)/(2*S);

    for si = 1:N_shapes

        % subplot index: row = ant_idx, col = si
        subplot_idx = (ant_idx-1)*N_shapes + si;
        subplot(N_ant_list, N_shapes, subplot_idx);

        sc_world = [shapes{si}(:,1)+0, ...
                    shapes{si}(:,2)+obj_range, ...
                    shapes{si}(:,3)];

        RA    = generate_RA(sc_world, rp);
        RA_dB = 20*log10(RA/max(RA(:)) + eps);

        imagesc(ang_axis, range_axis, RA_dB);
        colormap('jet'); clim([-40 0]);
        set(gca,'YDir','normal');
        xlim([-60 60]); ylim([y_lo y_hi]);

        % Column headers — shape names on top row only
        if ant_idx == 1
            title(shape_names{si}, 'FontSize', 11, ...
                  'FontWeight','bold','Color',shape_colors{si});
        end

        % Row labels — antenna count on leftmost column only
        if si == 1
            ylabel(sprintf('N=%d\n(%.1f°)', N_ant, 114.6/N_ant), ...
                   'FontSize', 9, 'FontWeight','bold');
        else
            ylabel('');
        end

        % X axis label on bottom row only
        if ant_idx == N_ant_list
            xlabel('Angle (°)','FontSize',8);
        else
            set(gca,'XTickLabel',[]);
        end

        % Colourbar on rightmost column only
        if si == N_shapes
            colorbar('FontSize',7);
        end

        % Truth marker
        hold on;
        plot(0, obj_range, 'w+', 'MarkerSize',8, 'LineWidth',1.8);
    end

    fprintf('Row %d/%d done (N_ant=%d)\n', ant_idx, N_ant_list, N_ant);
end

sgtitle('Range-Azimuth Heatmaps  |  All Shapes × All Antenna Counts  |  4m  |  0°  |  Medium Size', ...
        'FontSize', 13, 'FontWeight', 'bold');

%% SAVE
exportgraphics(gcf, 'heatmaps_all_shapes_all_antennas.png', ...
               'Resolution', 200);
fprintf('\nSaved: heatmaps_all_shapes_all_antennas.png\n');