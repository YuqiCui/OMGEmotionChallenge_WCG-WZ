function d=stftms(x,win,nfft,inc)
%短时傅里叶变换函数,给一个语音信号自动分帧进行短时傅里叶变换
%输出nfft/2+1个列，每一列是一帧快速傅里叶变换的数值
%x： 信号
%win: 窗口函数或分帧长度，如果设置为分帧长度，就自动设置海宁窗；
%nfft：傅里叶变换的窗口的长度，大于或者等于帧长
%Inc：帧移
if length(win)==1               %如果win是帧长
    wlen=win;
    win=hanning(wlen);
else
    wlen=length(win);           %如果win是傅里叶变换的窗口
end
x=x( : ); win=win( : );
s=length(x);
c=1;
d=zeros((1+nfft/2),1+fix((s-wlen)/inc));

for b=0:inc:(s-wlen)
    u=win.*x((b+1):(b+wlen));
    t=fft(u,nfft);
    d( : ,c)=t(1:(1+nfft/2));
    c=c+1;
end
% d=real(d);%存储为实数
end