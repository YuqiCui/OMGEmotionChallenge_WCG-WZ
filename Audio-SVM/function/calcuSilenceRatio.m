function SR=calcuSilenceRatio(signal,fs)
%输入信号矩阵，求得每一列信号的silence ratio，我这里认为silence ratio就是每列信号的silence frame的个数
%fs 用来确定当每一秒分成50帧的时候的帧长，从而方便将信号分割成帧

wlen=floor(fs/50);    %确定此时的帧长
wlen1=50*wlen;
X1=enframe(signal,wlen)';     %以帧长为wlen分割信号为帧,可以没有帧移
rms1=rms(X1);%计算每一帧的均方根（RMS）,返回一个行向量
%     rms1len=size(rms1);
X2=enframe(signal,wlen1)';%以帧长为wlen1,分割信号；
rms2=rms(X2);           %计算每50帧那么长的信号的均方根,也返回一个行向量
rms2len=size(rms2,2);
count=0;
%     sr=zeros(1,rms1len);
for j=1:rms2len
    for jj=1:50
        if (rms1((j-1)*50+jj)/rms2(j))<0.5
            count=count+1;
        end
    end
end
SR=count;
end


