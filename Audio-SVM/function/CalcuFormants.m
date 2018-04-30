function formants=CalcuFormants(signal,fs,wlen,inc)
%��һ����Ƶ�źž���,֡����֡�ƣ����ط������й��������
%calculate the short-time average rate
%|signal|:  the vector of discrete time audio signal
%|len|:  the size of processing window 
    X=enframe(signal,wlen,inc)';     %�źŷ�֡
    xcol=size(X,2);%����źŵ�֡��
   formant1=zeros(xcol,5);
    %��ÿһ֡���źŽ��д�����ȡ�����
    for j=1:xcol
        x=X( : ,j);
       formant1(j, : )=calformant(x,fs);%ÿ֡�Ĺ���尴���б���
    end
   formants=mean(formant1);
end   

function formant=calformant(x,fs)
%�������������ȡһ֡�źŵĹ���壬����һ��һ�����е�����
        u=filter([1-.99],1,x);%Ԥ����
        wlen=length(u); %һ֡�źŵĳ���
        cepstL=6;%��Ƶ���ϴ������Ŀ��
        wlen2=wlen/2;
        freq=(0:wlen2-1)*fs/wlen; %freq��һ����СΪ1*wlen2��һά����
        u2=u.*hamming(wlen);  %���Դ�����֮��u2de ��С��ȻΪu1�Ĵ�С,wlen*1
        U=fft(u2);%�任���С��λwlen*1
        U_abs=log(abs(U(1:wlen2)));  %ȡ��ֵ��ȡ����  wlen2*1��0ȡ�����仯�˸�����
        Cepst=ifft(U_abs);   %�����渵��Ҷ�任���õ�������
        Cepst(find(isnan(Cepst)==1)) = 0;           %�����Ϊ�����-lnf�����˿�ֵ������Ϊ0
        cepst=zeros(1,wlen2);   %1*wlen2
        cepst(1:cepstL)=Cepst(1:cepstL);
        cepst(end-cepstL+2:end)=Cepst(end-cepstL+2:end);
        spect=real(fft(cepst));    %����Ҷ�任�ð�����   1*wlen2
        Val1=zeros(1,5);
        if sum(spect)==0
            formant=zeros(1,5); %�������ȫ������Ϊ0�������ֱ�ӷ������Ϊ0�Ĺ����
        else
            [~,Val]=findpeaks(spect);   %Ѱ�ҷ�ֵ,
            if length(Val)>5
                Val1=Val( : ,1:5);
                formant=freq(Val1);   %��÷�ֵ��Ƶ��
            else
                 formant=freq(Val);   %��÷�ֵ��Ƶ��
                 formant=[formant , zeros(1,5-length(formant))]; %���ֻ���ĸ�����壬�Ͳ���
            end
        end

end