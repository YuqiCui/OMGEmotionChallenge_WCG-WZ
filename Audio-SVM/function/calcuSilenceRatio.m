function SR=calcuSilenceRatio(signal,fs)
%�����źž������ÿһ���źŵ�silence ratio����������Ϊsilence ratio����ÿ���źŵ�silence frame�ĸ���
%fs ����ȷ����ÿһ��ֳ�50֡��ʱ���֡�����Ӷ����㽫�źŷָ��֡

wlen=floor(fs/50);    %ȷ����ʱ��֡��
wlen1=50*wlen;
X1=enframe(signal,wlen)';     %��֡��Ϊwlen�ָ��ź�Ϊ֡,����û��֡��
rms1=rms(X1);%����ÿһ֡�ľ�������RMS��,����һ��������
%     rms1len=size(rms1);
X2=enframe(signal,wlen1)';%��֡��Ϊwlen1,�ָ��źţ�
rms2=rms(X2);           %����ÿ50֡��ô�����źŵľ�����,Ҳ����һ��������
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


