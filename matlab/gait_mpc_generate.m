clear
clc

addpath(genpath(pwd));

dt=0.001;
q=zeros(9501,13);
q(:,1)=0:dt:9.5;

%下蹲
Tsqu=0.5; %下蹲时间
Hmax=1.225; %0位质心高度
Hwalk=1.15; %下蹲后质心高度

for i=1:round(Tsqu/dt)
    t=(i-1)*dt;
    A=Hwalk-Hmax; %高度变化量
    [a,~,~] = fun_quintic_polynomial_interpolation(0,A,0,0,0,0,Tsqu,t);
    pc=[0,0,Hmax+a];
    pfl=[0,0.1,0];
    [L_Hip_Yaw,L_Hip_Roll,L_Hip_Pitch,L_knee,L_Ankle_Pitch,L_Ankle_Roll]=fun_ik_gait_demo(pc,pfl,1);
    pfr=[0,-0.1,0];
    [R_Hip_Yaw,R_Hip_Roll,R_Hip_Pitch,R_knee,R_Ankle_Pitch,R_Ankle_Roll]=fun_ik_gait_demo(pc,pfr,2);
    q(500+i,2:13)=[L_Hip_Yaw,L_Hip_Roll,L_Hip_Pitch,L_knee,L_Ankle_Pitch,L_Ankle_Roll,R_Hip_Yaw,R_Hip_Roll,R_Hip_Pitch,R_knee,R_Ankle_Pitch,R_Ankle_Roll];
end

%行走
gait=fun_generate_gait();
for i=1:size(gait,1)
    pc=[gait(i,2),gait(i,3),Hwalk];
    pfl=gait(i,4:6);
    pfr=gait(i,7:9);
    [L_Hip_Yaw,L_Hip_Roll,L_Hip_Pitch,L_knee,L_Ankle_Pitch,L_Ankle_Roll]=fun_ik_gait_demo(pc,pfl,1);
    [R_Hip_Yaw,R_Hip_Roll,R_Hip_Pitch,R_knee,R_Ankle_Pitch,R_Ankle_Roll]=fun_ik_gait_demo(pc,pfr,2);
    q(1000+i,2:13)=[L_Hip_Yaw,L_Hip_Roll,L_Hip_Pitch,L_knee,L_Ankle_Pitch,L_Ankle_Roll,R_Hip_Yaw,R_Hip_Roll,R_Hip_Pitch,R_knee,R_Ankle_Pitch,R_Ankle_Roll];
end


function [L_Hip_Yaw,L_Hip_Roll,L_Hip_Pitch,L_knee,L_Ankle_Pitch,L_Ankle_Roll]=fun_ik_gait_demo(pc,pf,lr)
    %腿部逆运动学
    %已知质心位置pc，脚底中心位置pf，计算左腿关节角度
    %输出结果为从上到下依次关节角度，默认膝关节前弯
    if lr==1 %左腿
        phip=pc+[0 0.1 -0.175]; %髋关节位置
    else %右腿
        phip=pc+[0 -0.1 -0.175]; %髋关节位置
    end
    pankle=pf+[0,0,0.05]; %踝关节位置
    Lleg=norm(phip-pankle); %腿长
    %膝关节角度
    L_knee=pi-2*asin(Lleg);
    %侧摆
    L_Hip_Roll=asin((pankle(2)-phip(2))/norm(phip(2:3)-pankle(2:3)));
    L_Ankle_Roll=-L_Hip_Roll;
    %前摆
    L_Hip_Pitch=-asin((pankle(1)-phip(1))/Lleg)-L_knee/2;
    L_Ankle_Pitch=asin((pankle(1)-phip(1))/Lleg)-L_knee/2;
    L_Hip_Yaw=0;
end