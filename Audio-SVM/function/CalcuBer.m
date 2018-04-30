function Ber=CalcuBer(signal,wlen,inc)
%给一个音频信号矩阵,帧长和帧移,采样率，返回一个一列的向量，算出来的band energy redio
%calculate the short-time average rate
%|signal|:  the vector of discrete time audio signal
%|len|:  the size of processing window
Ber=zeros(1,2);
nfft=wlen;%设置快速傅里叶变换的窗口长度和帧长相同
M=stftms(signal,wlen,nfft,inc);  %将每一列信号输入到短时傅里叶变换函数中进行短时傅里叶变换
[m,n]=size(M);%初始化M矩阵的行和列
%计算每一帧的spectrum centroid，并且求和
add=zeros(1,n);
for j=1:n
    u=M( : ,j);    %取出一帧的数据
    h=floor(m/4);
    if sum(u)==0
        add(j)=0;
    else
        add(j)=sum(u(1:h))/sum(u);
    end
end
Ber(1)=mean(add);
Ber(2)=var(add);
Ber=real(Ber);
end
