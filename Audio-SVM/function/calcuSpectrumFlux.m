function SpecFlux=calcuSpectrumFlux(signal,wlen,inc)
%输入语音信号矩阵，返回一维特征Spectrum Flux
num=size(signal, 2);%初始化矩阵的列数
win=hanning(wlen);%给出海宁窗
SpecFlux=zeros(num,1);


for i=1:num;           %遍历每一列音频信号
   x=signal( : ,i);       %获得一个音频信号
    X=enframe(x,win,inc)';    %将信号分帧，存储为矩阵
    [m,n]=size(X);                % 行为每帧的长度，列为帧数
   
    Y=zeros(m/2+1,n) ;              %新建一个矩阵，用来存储傅里叶变换后的结果
    
   for ii=1:n    %遍历每一帧数据
       %首先对每一帧数据进行一个离散变换
       u=X( : , ii);
       nfft=wlen;
       Y( : ,ii)=computeDFTviaFFT(u,wlen,nfft);   %对每一帧数据进行一个离散傅里叶变换，存入Y中，最后的向量元素个数应该是nfft/2+1
       [yraw,ycol]=size(Y);
   end
    
        M=zeros(yraw,ycol-1);
   %计算特征值
   for k=2:n  %遍历每一帧
       M( : ,ycol-1)=power((log(Y( : ,k))-log(Y( : ,k-1))),2);  
   end
   A=ones(ycol-1,1);
   B=ones(1,yraw);
   SpecFlux(i)=(B*(M*A))/((ycol-1)*yraw);
 
end
end

function [Xx] = computeDFTviaFFT(xin,nx,nfft)
% Use FFT to compute raw STFT and return the F vector.
%nx: 信号长度，帧长
%nfft： 傅里叶变换的窗口长度
% Handle the case where NFFT is less than the segment length, i.e., "wrap"
% the data as appropriate.
xin_ncol = size(xin,2);          
xw = zeros(nfft,xin_ncol);   
if nx > nfft                         %判断了一下帧长和傅里叶变换窗口长度的关系  %如果帧长大于窗口长
    for j = 1:xin_ncol
        xw(:,j) = datawrap(xin(:,j),nfft);
    end
else
    xw = xin;
end
Xx1 = abs(fft(xw,nfft));
m=size(Xx1,2);
mm=m/2+1;
Xx=Xx1(1:mm);
%f = psdfreqvec('npts',nfft,'Fs',Fs);

end

