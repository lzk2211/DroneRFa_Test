% 清除环境变量，关闭所有窗口
close all;

% 加载信号
signal = RF0_I + 1i * RF0_Q;
% signal = RF1_I + 1i * RF1_Q;

% 基础参数
T = 256 * 256;
N = 256;
fs = 1e8;
h = hanning(N);
noverlap = N / 2;

% 截取一部分信号进行处理
signal_segment = signal(6*T+1:7*T);

figure;
subplot(221);
plot(real(signal_segment));
hold on
plot(imag(signal_segment));


%% ---- STFT 部分 ----
[S, f, t] = spectrogram(signal_segment, h, noverlap, N, fs);
S = fftshift(S);

subplot(222);
A = 20 * log10(abs(S)); % 转换为分贝
imagesc(t, f, A); % 绘制STFT的时频图
colorbar;
colormap("jet");
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('STFT Magnitude');


%% ---- CWT 部分 ----
% CWT 参数
% scales = 1:256; % 尺度范围
cwt_wavelet = 'amor'; % 修正为有效的复Morlet小波

% 计算小波变换
[coeffs, freq] = cwt(signal_segment, cwt_wavelet, fs);

% 绘制 CWT Scalogram
subplot(223);
energy_total = abs(coeffs(:, :, 1)).^2 + abs(coeffs(:, :, 2)).^2;
% A_cwt = abs(coeffs).^2; % 能量密度
imagesc((1:length(signal_segment))/fs, freq, 10*log10(energy_total));
colorbar;
colormap("parula");
xlabel('Time (s)');
ylabel('Frequency (Hz)');
title('CWT Scalogram');

% CWT结果归一化并保存
% A_min_cwt = min(A_cwt(:));
% A_max_cwt = max(A_cwt(:));
% A_range_cwt = A_max_cwt - A_min_cwt;
% A_normalized_cwt = (A_cwt - A_min_cwt) / A_range_cwt;
% CWT_GRAY_image = cat(3, A_normalized_cwt, A_normalized_cwt, A_normalized_cwt);
% 
% % 保存为图像
% imwrite(CWT_GRAY_image, 'CWT_long_pic.png');

%% ---- WST 部分 ----
% WST 参数
% filter_bank = waveletScattering('SignalLength', length(signal_segment), ...
%                                 'SamplingFrequency', fs, ...
%                                 'QualityFactors', [8, 1]);
% 
% % 计算 WST 系数
% S_wst = filter_bank(signal_segment);
% 
% % 可视化 WST 系数
% subplot(224);
% imagesc(1:size(S_wst, 2), 1:size(S_wst, 1), 10*log10(S_wst));
% colorbar;
% colormap("parula");
% xlabel('Scattering Coefficient Index');
% ylabel('Frequency Band');
% title('WST Scattergram');

% WST结果归一化并保存
% A_min_wst = min(S_wst(:));
% A_max_wst = max(S_wst(:));
% A_range_wst = A_max_wst - A_min_wst;
% A_normalized_wst = (S_wst - A_min_wst) / A_range_wst;
% WST_GRAY_image = cat(3, A_normalized_wst, A_normalized_wst, A_normalized_wst);
% 
% % 保存为图像
% imwrite(WST_GRAY_image, 'WST_long_pic.png');

% 创建 Wavelet Scattering 对象
sf = waveletScattering('SignalLength', numel(abs(signal_segment)), ...
                       'SamplingFrequency', fs);

% 进行小波散射变换
[S, U] = scatteringTransform(sf, abs(signal_segment));

% 时间轴
t = (0:length(signal_segment)-1) / fs;

% 绘制原始信号
figure;
subplot(2, 1, 1);
plot(t, signal_segment);
grid on;
axis tight;
xlabel('Time (s)');
title('Signal Segment');

% 绘制零阶散射系数
subplot(2, 1, 2);
plot(S{1}.signals{1}, 'x-');
grid on;
axis tight;
xlabel('Scattering Coefficient Index');
ylabel('Amplitude');
title('Zeroth-Order Scattering Coefficients');

% 绘制散射图
figure;
scattergram(sf, U, 'FilterBank', 1);
