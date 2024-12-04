close all;

strs = ["0001", "0010", "0011", "0100", "0101", "0111", "1001"];
distance = "D10";

% str = "0001";
% 使用for循环遍历字符串数组
for k = 1:length(strs)
    % 输出当前字符串
    str = strs(k);
    filename = sprintf('F:\\%s\\T%s\\*.mat', distance, str);
    if exist(sprintf('F:\\%s\\data_set_1e7\\T%s', distance, str))==0 %%判断文件夹是否存在
        mkdir(sprintf('F:\\%s\\data_set_1e7\\T%s', distance, str));  %%不存在时候，创建文件夹
    end
    
    Files = dir(fullfile(filename));
    LengthFiles = length(Files);
    
    index = 0;
    N = 1024;
    fs = 1e8;
    h = hanning(N);
    noverlap = 48;
    M = 10e6;
    Q = 1e6;
    
    for j=1:LengthFiles
        name=Files(j).name;           %读取struct变量的格式
        folder=Files(j).folder;
        disp(['loading... ' folder,'\',name]);
        load([folder,'\',name]);    %导入文件
     
        signal = RF0_I + 1i*RF0_Q;
        
        disp('packing pic...');
        for i = 0:(size(signal)/Q)-10
            signal_data = signal((i*Q+1):(i*Q+M));
            [S, f, t] = spectrogram(signal_data, h, noverlap, N, fs);
            S = fftshift(S);
            A = 20*log10(abs(S));
    
            pic_name = sprintf('F:\\%s\\data_set_1e7\\T%s\\%d.png', distance, str, index + i);
            
            % 归一化矩阵
            A_min = min(A(:)); % 矩阵的最小值
            A_max = max(A(:)); % 矩阵的最大值
            
            % 计算最大值和最小值之间的差
            A_range = A_max - A_min;
            
            % 归一化
            A_normalized = (A - A_min) / A_range;
    
            GRAY_image = cat(3, A_normalized, A_normalized, A_normalized);
    
            % 保存图像
    %         save(mat_name, "A");
            GRAY_image = imresize(GRAY_image,[1024,1024]);
            imwrite(GRAY_image, pic_name);
            fprintf('%d ', index + i);
            
        end
        
        fprintf('\n');
        index = index + i + 1;
        
        % print(img, 'myImage.png');
        % caxis([min_val max_val])
        % xlabel('Time (s)');
        % ylabel('Frequency (Hz)');
        % zlabel('Magnitude');
        % title('STFT Magnitude');
    
    end

end
