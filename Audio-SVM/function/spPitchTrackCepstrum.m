% 名称
%   spPitchTrackCepstrum: Pitch Tracking via the Cepstral Method 
% 使用
%   [F0, T, C] = 
%     spPitchTrackCepstrum(x, fs, frame_length, frame_overlap, window, show)
% 描述
%   在时域跟踪 F0变化
% 输入
%   x               大小Nx1.
%   fs              采样率，单位Hz. 
%   [frame_length]  声音片段长度，默认30（ms） 
%   [frame_overlap] 声音片段重叠部分，默认重叠长度的一半。
%   [window] (字符型)窗型'rectwin'（默认）或者'hamming'  
%   [show]   是否画出来，默认 false
% 输出
%   F0              1*k 包含基本频率，K 是声音片段数量。
%   T               1*k,每个声音片段中间的值
%   [C]             M*K 包含cepstrogram 

function [F0, T, C] = spPitchTrackCepstrum(x, fs, frame_length, frame_overlap, window, show)
 %% 初始化
 N = length(x);
 if ~exist('frame_length', 'var') || isempty(frame_length)
     frame_length = 30;
 end
 if ~exist('frame_overlap', 'var') || isempty(frame_overlap)
     frame_overlap = 20;
 end
 if ~exist('window', 'var') || isempty(window)
     window = 'hamming';
 end
 if ~exist('show', 'var') || isempty(show)
     show = 0;
 end
 nsample = round(frame_length  * fs / 1000);
 noverlap = round(frame_overlap * fs / 1000); 
 if ischar(window)
     window   = eval(sprintf('%s(nsample)', window)); % e.g., hamming(nfft)
 end

  %% 基音监测
 pos = 1; i = 1;
 while (pos+nsample < N)
     frame = x(pos:pos+nsample-1);
     C(:,i) = spCepstrum(frame, fs, window);
     F0(i) = spPitchCepstrum(C(:,i), fs);
     pos = pos + (nsample - noverlap);
     i = i + 1;
 end
 T = (round(nsample/2):(nsample-noverlap):N-1-round(nsample/2))/fs;

if show 
     % 画出波形
    subplot(2,1,1);
    t = (0:N-1)/fs;
    plot(t, x);
    legend('Waveform');
    xlabel('Time (s)');
    ylabel('Amplitude');
    xlim([t(1) t(end)]);

    % 画出 F0跟踪
    subplot(2,1,2);
    plot(T,F0);
    legend('pitch track');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    xlim([t(1) t(end)]);
end
end


% 名称
%   spCepstrum 
% 用法
%   [c, y] = spCepstrum(x, fs, window, show)
% 面述
%   一个信号的倒谱
% 输入
%   x        N*1的向量用于容纳声音信号
%   fs       采样频率
%   [window] (字符型)窗型'rectwin'（默认）或者'hamming'  
%   [show]   是否画出来，默认 false
% OUTPUTS
%   c        N*1的倒谱信息。
%   [y]      N*1的傅里叶响应。
function [c, y] = spCepstrum(x, fs, window, show)
 %% 初始化
 N = length(x);
 x = x(:); % 确保一下是纵向量
 if ~exist('show', 'var') || isempty(show)
     show = 0;
 end
 if ~exist('window', 'var') || isempty(window)
     window = 'rectwin';
 end
 if ischar(window);
     window = eval(sprintf('%s(N)', window)); % hamming(N)
 end

 %% 窗形信号的傅里叶变换
 x = x(:) .* window(:);
 y = fft(x, N);

 %% 倒谱是log谱的 IDFT（或者 DFT）
 c = ifft(log(abs(y)+eps));

 if show
     ms1=fs/1000; % 1ms. 声音的最大 FX（1000Hz）
     ms20=fs/50;  % 20ms.  50Hzs声音最小 FX(50Hz)

     %% 画出波形
     t=(0:N-1)/fs;        % 采样次数
     subplot(2,1,1);
     plot(t,x);
     legend('Waveform');
     xlabel('Time (s)');
     ylabel('Amplitude');

     %% 画出1ms (=1000Hz)到20ms (=50Hz)的倒谱
     %% DC 部分 c(0) 太大
     q=(ms1:ms20)/fs;
     subplot(2,1,2);
     plot(q,abs(c(ms1:ms20)));
     legend('Cepstrum');
     xlabel('Quefrency (s) 1ms (1000Hz) to 20ms (50Hz)');
     ylabel('Amplitude');
 end
end