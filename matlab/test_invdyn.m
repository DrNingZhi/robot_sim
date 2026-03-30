clear
clc

addpath(genpath(pwd));

%构造一个3连杆系统
T1=[eye(3),[0,0,0]';0,0,0,1];
T2=[1,0,0,0;
    0,0,1,0.5;
    0,-1,0,0;
    0,0,0,1];
T3=[0,0,1,0;
    0,1,0,-0.5;
    -1,0,0,0;
    0,0,0,1];
% 随机赋予质量属性
m1=rand*5;
m2=rand*3;
m3=rand;
c1=rand(1,3).*[0.1,0.5,0.1];
c2=rand(1,3).*[0.1,-0.5,0.1];
c3=rand(1,3).*[0.1,-0.5,0.1];
r1=rand(1,6).*[1,1,1,0.1,0.1,0.1]*5;
I1=[r1(1),r1(6),r1(5);
    r1(6),r1(2),r1(4);
    r1(5),r1(4),r1(3)];
r2=rand(1,6).*[1,1,1,0.1,0.1,0.1]*3;
I2=[r2(1),r2(6),r2(5);
    r2(6),r2(2),r2(4);
    r2(5),r2(4),r2(3)];
r3=rand(1,6).*[1,1,1,0.1,0.1,0.1]*1;
I3=[r3(1),r3(6),r3(5);
    r3(6),r3(2),r3(4);
    r3(5),r3(4),r3(3)];

%运动规划
q0=-rand(1,3)*pi/2;
q1=rand(1,3)*pi/2;
T=1;
dt=0.001;
t=0:dt:T;
q=zeros(length(t),3);
qd=zeros(length(t),3);
qdd=zeros(length(t),3);
for i=1:3
    for j=1:length(t)
        q(j,i)=fun_step5(t(j),0,q0(i),T,q1(i),0);
        qd(j,i)=fun_step5(t(j),0,q0(i),T,q1(i),1);
        qdd(j,i)=fun_step5(t(j),0,q0(i),T,q1(i),2);
    end
end

%用自己写的逆动力学函数求解关节扭矩
TT=[T1;T2;T3];
m=[m1,m2,m3];
c=[c1',c2',c3'];
I=[I1;I2;I3];
tau=zeros(length(t),3);
for i=1:length(t)
    tau(i,:)=myID(3,TT,m,c,I,q(i,:),qd(i,:),qdd(i,:));
end

%构造matlab robotics system toolbox机器人模型
robot=rigidBodyTree;

J1=rigidBodyJoint('J1','revolute');
setFixedTransform(J1,T1);
Link1=rigidBody('Link1');
Link1.Joint=J1;
Link1.Mass=m1;
Link1.CenterOfMass=c1;
I1p=fun_parallel_axis_move(I1,m1,c1');   %toolbox的转动惯量要求是对齐link坐标系的，因此需要进行平行移轴
Link1.Inertia=[I1p(1,1),I1p(2,2),I1p(3,3),I1p(2,3),I1p(3,1),I1p(1,2)];
addBody(robot,Link1,'base');

J2=rigidBodyJoint('J2','revolute');
setFixedTransform(J2,T2);
Link2=rigidBody('Link2');
Link2.Joint=J2;
Link2.Mass=m2;
Link2.CenterOfMass=c2;
I2p=fun_parallel_axis_move(I2,m2,c2');   %toolbox的转动惯量要求是对齐link坐标系的，因此需要进行平行移轴
Link2.Inertia=[I2p(1,1),I2p(2,2),I2p(3,3),I2p(2,3),I2p(3,1),I2p(1,2)];
addBody(robot,Link2,'Link1');

J3=rigidBodyJoint('J3','revolute');
setFixedTransform(J3,T3);
Link3=rigidBody('Link3');
Link3.Joint=J3;
Link3.Mass=m3;
Link3.CenterOfMass=c3;
I3p=fun_parallel_axis_move(I3,m3,c3');   %toolbox的转动惯量要求是对齐link坐标系的，因此需要进行平行移轴
Link3.Inertia=[I3p(1,1),I3p(2,2),I3p(3,3),I3p(2,3),I3p(3,1),I3p(1,2)];
addBody(robot,Link3,'Link2');
robot.DataFormat='row';
robot.Gravity=[0,0,-9.8];
% show(robot)

% 用toolbox逆动力学函数求关节扭矩
tau1=zeros(length(t),3);
for i=1:length(t)
    tau1(i,:)=inverseDynamics(robot,q(i,:),qd(i,:),qdd(i,:));
end

%画出扭矩结果
for i=1:3
    subplot(1,3,i)
    plot(t,tau(:,i),'r','LineWidth',3)
    hold on
    plot(t,tau1(:,i),'b-.','LineWidth',2)
    hold on
    legend('myID','toolbox')
    xlabel('Time(s)');
    ylabel('Torque(Nm)')
    set(gca,'FontSize',15)
    set(gcf,'color',[1 1 1])
end