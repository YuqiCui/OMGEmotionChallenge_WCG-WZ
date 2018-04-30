function lpcc=calculpcc(signal,wlen,inc)
%输入语音信号矩阵，计算其mfcc特征，返回矩阵，行=语音信号的列数，列=MFCC参数个数
%初始化矩阵的列数，遍历每一列信号
lpcc=zeros(1,24); %每一个信号返回12维参数
temp=lpcc_m(signal,wlen,inc);%返回的是一个帧数*12的矩阵
lpcc(1: 24)=temp;

end

function ppp=lpcc_m(x,framesize,inc)
%输入一段音频信号，对信号进行分帧，然后求每帧的lpcc系数，并且求得均值

%预加重滤波器
xx=double(x);
xx=filter([1-0.9375],1,xx);

%语音信号分帧
xx=enframe(xx,framesize,inc);
m=zeros(size(xx,1),12);

for i=1:size(xx,1) %遍历每一帧
    X=xx(i,:);
    [ar,~]=lpc(X,12);%x是一帧的数据，p是线性预测的阶数。可以取12
    temp=lpc2lpccm(ar,12,12);%获得lpcc系数，行向量
    m(i,:)=temp;
end
m(find(isnan(m)==1)) = 0;
ppp(1:12)=mean(m,1);%求得每一列的均值
ppp(13:24)=var(m);
ppp(find(isnan(ppp)==1)) = 0;
end

function lpcc=lpc2lpccm(ar,n_lpc,n_lpcc)
lpcc=zeros(n_lpcc,1);
lpcc(1)=ar(1);
for n=2:n_lpc
    lpcc(n)=ar(n);
    for l=1:n-1
        lpcc(n)=lpcc(n)+ar(l)*lpcc(n-l)*(n-l)/n;
    end
end
for n=(n_lpc+1):n_lpcc
    lpcc(n)=0;
    for l=1:n_lpc
        lpcc(n)=lpcc(n)+ar(l)*lpcc(n-l)*(n-l)/n;
    end
end
lpcc=-lpcc;
end

