function formants=CalcuFormants(signal,fs,wlen,inc)
%给一个音频信号矩阵,帧长和帧移，返回返回四列共振峰数据
%calculate the short-time average rate
%|signal|:  the vector of discrete time audio signal
%|len|:  the size of processing window 
    X=enframe(signal,wlen,inc)';     %信号分帧
    xcol=size(X,2);%获得信号的帧数
   formant1=zeros(xcol,5);
    %对每一帧的信号进行处理，获取共振峰
    for j=1:xcol
        x=X( : ,j);
       formant1(j, : )=calformant(x,fs);%每帧的共振峰按照列保存
    end
   formants=mean(formant1);
end   

function formant=calformant(x,fs)
%这个函数用于提取一帧信号的共振峰，返回一个一行五列的向量
        u=filter([1-.99],1,x);%预加重
        wlen=length(u); %一帧信号的长度
        cepstL=6;%倒频率上窗函数的宽度
        wlen2=wlen/2;
        freq=(0:wlen2-1)*fs/wlen; %freq是一个大小为1*wlen2的一维向量
        u2=u.*hamming(wlen);  %乘以窗函数之后，u2de 大小仍然为u1的大小,wlen*1
        U=fft(u2);%变换后大小让位wlen*1
        U_abs=log(abs(U(1:wlen2)));  %取幅值后取对数  wlen2*1，0取对数变化了负无穷
        Cepst=ifft(U_abs);   %进行逆傅里叶变换，得倒谱序列
        Cepst(find(isnan(Cepst)==1)) = 0;           %如果因为上面的-lnf产生了空值，就置为0
        cepst=zeros(1,wlen2);   %1*wlen2
        cepst(1:cepstL)=Cepst(1:cepstL);
        cepst(end-cepstL+2:end)=Cepst(end-cepstL+2:end);
        spect=real(fft(cepst));    %傅里叶变换得包络线   1*wlen2
        Val1=zeros(1,5);
        if sum(spect)==0
            formant=zeros(1,5); %如果出现全部数据为0的情况，直接返回五个为0的共振峰
        else
            [~,Val]=findpeaks(spect);   %寻找峰值,
            if length(Val)>5
                Val1=Val( : ,1:5);
                formant=freq(Val1);   %获得峰值的频率
            else
                 formant=freq(Val);   %获得峰值的频率
                 formant=[formant , zeros(1,5-length(formant))]; %如果只有四个共振峰，就补零
            end
        end

end