function mfcc=calcumfcc(signal,fs,wlen,inc)
%输入语音信号矩阵，计算其mfcc特征，返回矩阵，行=语音信号的列数，列=MFCC参数个数
%初始化矩阵的列数，遍历每一列信号
mfcc=zeros(1,36);
    temp=mfcc_m(signal,fs,16,wlen,inc);%返回的是一个帧数*24的矩阵
    %这两行代码是后面加的，为了求12维MFCC参数的方差
    temp1=temp( : ,1:12);
    varTemp=std(temp1,0,1).^2; %按列先求标准差，再平方得方差
    mfcc(1: 24)=mean(temp,1); %按列求平均
    mfcc(25: 36)=varTemp;
end

function ccc=mfcc_m(x,fs,p,framesize,inc)
%对输入的语音序列x进行MFCC参数提取，返回MFCC参数和一阶差分MFCC参数，滤波器的个数为p,采样频率为fs
%对x每framesize点分为一帧，相邻两帧之间的帧移为inc
%FFT的长度为帧长
%按照帧长为frameSize，Mel滤波器的个数为p,采样频率为fs
%提取mel滤波器的参数，用汉明窗函数
bank=melbankm(p,framesize,fs,0,0.5,'m');
%归一化Mel滤波器组系数
bank=full(bank);
bank=bank/max(bank(:));

%DCT系数，12*p
for k=1:12
    n=0:p-1;
    dctcoef(k, : )=cos((2*n+1)*k*pi/(2*p));
end

%归一化倒谱系数窗口
w=1+6*sin(pi*[1:12]./12);
w=w/max(w);

%预加重滤波器
xx=double(x);
xx=filter([1-0.9375],1,xx);

%语音信号分帧
xx=enframe(xx,framesize,inc);
n2=fix(framesize/2)+1;

%计算每帧的MFCC参数
for i=1:size(xx,1)
    y=xx(i,:);
    s=y'.*hamming(framesize);
    t=abs(fft(s));
    t=t.^2;
    c1=dctcoef*log(bank*t(1:n2));
    c2=c1.*w';
    m(i,:)=c2';
end

%差分系数
dtm=zeros(size(m));
for i=3:size(m,1)-2
    dtm(i,:)=-2*m(i-2,:)-m(i-1,:)+m(i+1,:)+2*m(i+2,:);
end
dtm=dtm/3;

%合并MFCC参数和一阶差分MFCC参数
ccc=[m dtm];

%去除首尾两帧，因为这两帧的一阶差分参数为0
ccc=ccc(3:size(m,1)-2,:);
ccc(find(isnan(ccc)==1)) = 0;
end