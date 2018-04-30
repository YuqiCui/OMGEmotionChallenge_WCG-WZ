function Ber=CalcuBer(signal,wlen,inc)
%��һ����Ƶ�źž���,֡����֡��,�����ʣ�����һ��һ�е��������������band energy redio
%calculate the short-time average rate
%|signal|:  the vector of discrete time audio signal
%|len|:  the size of processing window
Ber=zeros(1,2);
nfft=wlen;%���ÿ��ٸ���Ҷ�任�Ĵ��ڳ��Ⱥ�֡����ͬ
M=stftms(signal,wlen,nfft,inc);  %��ÿһ���ź����뵽��ʱ����Ҷ�任�����н��ж�ʱ����Ҷ�任
[m,n]=size(M);%��ʼ��M������к���
%����ÿһ֡��spectrum centroid���������
add=zeros(1,n);
for j=1:n
    u=M( : ,j);    %ȡ��һ֡������
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
