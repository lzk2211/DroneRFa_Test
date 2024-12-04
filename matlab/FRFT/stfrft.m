function result = stfrft(signal_data, h, noverlap, N, a)
    % 短时分数傅里叶变换 (STFRFT)
    % 输入参数：
    % - signal_data: 原始信号
    % - h: 窗函数
    % - noverlap: 窗口重叠点数
    % - N: 每段的采样点数
    % - a: 分数傅里叶变换的阶数
    % 输出：
    % - result: STFRFT 结果矩阵

    signal_length = length(signal_data);   % 原始信号长度
    hop = N - noverlap;                    % 每次移动的样本数
    num_segments = floor((signal_length - noverlap) / hop); % 分段数
    
    % 初始化结果矩阵
    result = zeros(N, num_segments);
    
    % 对每一段信号进行分数傅里叶变换
    for k = 1:num_segments
        % 提取当前段的信号
        start_idx = (k - 1) * hop + 1;
        end_idx = start_idx + N - 1;
        segment = signal_data(start_idx:end_idx);
    
        % 应用窗函数
        segment = segment .* h;
    
        % 对当前段进行分数傅里叶变换
        result(:, k) = frft(segment, a);
    end
end
