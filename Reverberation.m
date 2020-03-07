
function revb_x = Reverberation(x,fs,beta)

%%%
%
% This function is mainly retrieved from https://github.com/JackFrost168/Reverberation
%
%%%

%  Schroeder reverb modle:8 lowpass comb filter in parallel and 2 allpass filter with series connection     

f=beta*0.8; %beta is hyperparameter control the reverb room size which is between 0 and 1

n = 8;
%  8th lowpass filter
m1=floor(fs*0.0353);         %echo interval 0.0353
den1 = zeros(m1,1);
den1(1)=1;
den1(2)=-0.2;
den1(m1) = -f;    
lowpass1=filter([1,den1(2)],den1,x/n);

m2=floor(fs*0.0367);     %echo interval 0.0367
den2 = zeros(m2,1);
den2(1)=1;
den2(2)=-0.2;
den2(m2) = -f;    
lowpass2=filter([1,den2(2)],den2,x/n);

m3=floor(fs*0.0338);         %echo interval 0.0338
den3 = zeros(m3,1);
den3(1)=1;
den3(2)=-0.2;
den3(m3) = -f;    
lowpass3=filter([1,den3(2)],den3,x/n);

m4=floor(fs*0.0322);          %echo interval 0.0322
den4 = zeros(m4,1);
den4(1)=1;
den4(2)=-0.2;
den4(m4) = -f;     
lowpass4=filter([1,den4(2)],den4,x/n);

low1=lowpass1+lowpass2+lowpass3+lowpass4;

m5=floor(fs*0.02895);    %echo interval 0.02895
den5 = zeros(m5,1);
den5(1)=1;
den5(2)=-0.2;
den5(m5) = -f;     
lowpass5=filter([1,den5(2)],den5,x/n);

m6=floor(fs*0.0307);    %echo interval 0.0307
den6 = zeros(m6,1);
den6(1)=1;
den6(2)=-0.2;
den6(m6) = -f;     
lowpass6=filter([1,den6(2)],den6,x/n);

m7=floor(fs*0.0269);         %echo interval 0.0269
den7 = zeros(m7,1);
den7(1)=1;
den7(2)=-0.2;
den7(m7) = -f;     %the lager the more time
lowpass7=filter([1,den7(2)],den7,x/n);

m8=floor(fs*0.0253);         %echo interval 0.0253
den8 = zeros(m8,1);
den8(1)=1;
den8(2)=-0.2;
den8(m8) = -f;     
lowpass8=filter([1,den8(2)],den8,x/n);

low2=lowpass5+lowpass6+lowpass7+lowpass8;
low=low1+low2;

%  the first allpass filter
n1=floor(fs*0.0051);          %0.0051
g1=0.5;
numallpass1=zeros(n1,1);
numallpass1(1)=-g1;
numallpass1(n1)=1;

denallpass1=zeros(n1,1);
denallpass1(1)=1;
denallpass1(n1)=-g1;
allpass1=filter(numallpass1,denallpass1,low);

%  the second allpass filter
n2=floor(fs*0.0126);          %0.0126
g2=0.5;
numallpass2=zeros(n2,1);
numallpass2(1)=-g2;
numallpass2(n2)=1;

denallpass2=zeros(n2,1);
denallpass2(1)=1;
denallpass2(n2)=-g2;
revb_x = filter(numallpass2,denallpass2,allpass1);
end
