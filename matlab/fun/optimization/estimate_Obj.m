function Obj = estimate_Obj(p)
    %目标函数
    [y1,y2,t]=get_resp(p(1),p(2),p(3));
    err=std(y2-sin(t'));
    Obj=[max(y1)-1,err];
end


function  [y1,y2,t]=get_resp(P,I,D)

    N=100;
    m=1000*0.05*0.05*0.5;   %杆的质量
    J=m*(0.5^2+0.05^2)/12+m*0.25^2;  %杆的转动惯量
    %PID传递函数
    num1=[P+N*D,P*N+I,I*N];
    den1=[1,N,0];
    %物理系统的传递函数
    num2=[1];
    den2=[J,0,0];
    %PID串联在物理系统
    [numg,deng]=series(num1,den1,num2,den2);
    %增加反馈
    numf=[1];
    denf=[1];
    [num,den]=feedback(numg,deng,numf,denf);
    
    t=0:0.001:10;
    y1=step(num,den,t);  %阶跃响应
    r=sin(t);
    y2=lsim(num,den,r,t); %正弦响应
end