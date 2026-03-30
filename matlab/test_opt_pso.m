clear
clc

addpath(genpath(pwd));
run_opt = false

if ~run_opt
    %% 运行测试
    P=1;
    I=0.01;
    D=0.2;
    test_demo(P, I, D);
else
    %% 运行优化求解
    p_range=[0,0,0;10,10,10];  %设计变量PID的取值范围
    p_discrete=[0,0,0]; %设计变量的离散程度，0表示是连续值
    Ini_gene_method=0; %采用第一种初始化方法，随机产生初值
    p0_ini_para=0; %第一种方法无需这个参数
    NO=2; %目标数量
    pop = PSO_MO_main(p_range,p_discrete,Ini_gene_method,p0_ini_para,NO);
end

function test_demo(P, I, D)
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
    subplot(1,2,1)
    step(num,den,t);  %画出阶跃响应图像
    subplot(1,2,2)
    lsim(num,den,sin(t),t);  %画出正弦响应图像
end