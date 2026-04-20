function RA = generate_RA(world_sc, rp)
%GENERATE_RA  Generate Range-Angle heatmap from world scatterers
%
%  Inputs:
%    world_sc : [N x 3] matrix — [x_world, y_world, amplitude]
%    rp       : struct with radar parameters:
%                 rp.c, rp.fc, rp.lambda, rp.S, rp.Fs
%                 rp.N_s, rp.N_ant, rp.D, rp.d_max
%                 rp.NFFT_r, rp.NFFT_a, rp.snr_db
%
%  Output:
%    RA       : [NFFT_r x NFFT_a] Range-Angle magnitude map

    t  = (0:rp.N_s-1).' / rp.Fs;
    IF = zeros(rp.N_s, rp.N_ant);

    for s = 1:size(world_sc,1)
        sx  = world_sc(s,1);
        sy  = world_sc(s,2);
        amp = world_sc(s,3);

        d  = sqrt(sx^2 + sy^2);
        th = atan2d(sx, sy);

        if d < 0.1 || d > rp.d_max; continue; end

        f_beat     = rp.S * (2*d) / rp.c;
        phi_r      = (4*pi*rp.fc*d) / rp.c;
        phi_a_base = (2*pi/rp.lambda) * rp.D * sind(th);

        for n = 1:rp.N_ant
            phi_a  = phi_a_base * (n-1);
            signal = amp * exp(1j*(2*pi*f_beat*t + phi_r + phi_a));
            IF(:,n) = IF(:,n) + signal;
        end
    end

    % Noise
    noise_amp = 10^(-rp.snr_db / 20);
    IF = IF + noise_amp * (randn(size(IF)) + 1j*randn(size(IF)));

    % Hanning window + Range FFT (full spectrum — complex signal)
    win     = hanning(rp.N_s);
    rng_fft = fft(IF .* win, rp.NFFT_r, 1);

    % Angle FFT across antennas
    ang_fft = fftshift(fft(rng_fft, rp.NFFT_a, 2), 2);

    % Magnitude map
    RA = abs(ang_fft);   % [NFFT_r x NFFT_a]

end
