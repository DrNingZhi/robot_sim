clear
clc

Tsup=0.8;
zc=0.8;
g=9.8;
Tc=sqrt(zc/g);
S=sinh(Tsup/Tc);
C=cosh(Tsup/Tc);

dt=0.01;

%权重系数
a=10;b=1;

%步长
sx=[0,0.25,0.25,0.25,0];
sy=[0.2,0.2,0.2,0.2,0.2];
st=[0,20,40,60,60]*pi/180;

%计划落脚点
pxplan=zeros(1,length(sx));
pyplan=zeros(1,length(sy));
pxp=0;
pyp=0;
for i=1:length(sx)
    pxp=pxp+cos(st(i))*sx(i)+sin(st(i))*(-1)^i*sy(i);
    pyp=pyp+sin(st(i))*sx(i)-cos(st(i))*(-1)^i*sy(i);
    pxplan(i)=pxp;
    pyplan(i)=pyp;
end
%修正落脚点
px=zeros(1,length(sx));
py=zeros(1,length(sy));


%第一步末状态
xe=0;
ye=sy(1)/2;
vxe=(C+1)/(Tc*S) *xe;
vye=(C-1)/(Tc*S) *ye;
%求第一步运动
t=0:dt:Tsup;
t=-fliplr(t);
x=xe*cosh(t/Tc)+Tc*vxe*sinh(t/Tc);
y=ye*cosh(t/Tc)+Tc*vye*sinh(t/Tc);
vx=xe/Tc*sinh(t/Tc)+vxe*cosh(t/Tc);
vy=ye/Tc*sinh(t/Tc)+vye*cosh(t/Tc);
t=0:dt:Tsup;
ppx=zeros(1,length(t));
ppy=zeros(1,length(t));

%求中间步运动
for i=1:length(sx)-1
    %期望步行单元末状态（相对于本落脚点，对齐世界坐标系）
    xe=cos(st(i+1))*sx(i+1)/2-sin(st(i+1))*(-1)^i*sy(i+1)/2;
    ye=sin(st(i+1))*sx(i+1)/2+cos(st(i+1))*(-1)^i*sy(i+1)/2;
    vxe=cos(st(i+1))*(C+1)/(Tc*S)*sx(i+1)/2-sin(st(i+1))*(C-1)/(Tc*S)*(-1)^i*sy(i+1)/2;
    vye=sin(st(i+1))*(C+1)/(Tc*S)*sx(i+1)/2+cos(st(i+1))*(C-1)/(Tc*S)*(-1)^i*sy(i+1)/2;
    xe=xe+pxplan(i);
    ye=ye+pyplan(i);
    %步行单元初状态（相对于本落脚点，对齐世界坐标系）
    xi=x(end);
    yi=y(end);
    vxi=vx(end);
    vyi=vy(end);
    %修正落脚点位置
    D=a*(C-1)^2+b*(S/Tc)^2;
    px(i)=a*(C-1)*(-xe+C*xi+Tc*S*vxi)/D+b*S*(-vxe+S*xi/Tc+C*vxi)/(Tc*D);
    py(i)=a*(C-1)*(-ye+C*yi+Tc*S*vyi)/D+b*S*(-vye+S*yi/Tc+C*vyi)/(Tc*D);
    %求解运动
    tt=dt:dt:Tsup;
    xx=(xi-px(i))*cosh(tt/Tc)+Tc*vxi*sinh(tt/Tc)+px(i);
    yy=(yi-py(i))*cosh(tt/Tc)+Tc*vyi*sinh(tt/Tc)+py(i);
    vvx=(xi-px(i))*sinh(tt/Tc)/Tc+vxi*cosh(tt/Tc);
    vvy=(yi-py(i))*sinh(tt/Tc)/Tc+vyi*cosh(tt/Tc);
    pppx=ones(1,length(tt))*px(i);
    pppy=ones(1,length(tt))*py(i);
    t=[t,t(end)+tt];
    x=[x,xx];
    y=[y,yy];
    vx=[vx,vvx];
    vy=[vy,vvy];
    ppx=[ppx,pppx];
    ppy=[ppy,pppy];
end

%最后一次运动
%最后一步初状态
xi=x(end);
yi=y(end);
vxi=vx(end);
vyi=vy(end);
tt=dt:dt:Tsup;
i=length(sx);
xx=(xi-pxplan(i))*cosh(tt/Tc)+Tc*vxi*sinh(tt/Tc)+pxplan(i);
yy=(yi-pyplan(i))*cosh(tt/Tc)+Tc*vyi*sinh(tt/Tc)+pyplan(i);
vvx=(xi-pxplan(i))*sinh(tt/Tc)/Tc+vxi*cosh(tt/Tc);
vvy=(yi-pyplan(i))*sinh(tt/Tc)/Tc+vyi*cosh(tt/Tc);
pppx=ones(1,length(tt))*pxplan(i);
pppy=ones(1,length(tt))*pyplan(i);
t=[t,t(end)+tt];
x=[x,xx];
y=[y,yy];
vx=[vx,vvx];
vy=[vy,vvy];
ppx=[ppx,pppx];
ppy=[ppy,pppy];

subplot(1,2,1)
plot(x,y)  %质心位置
hold on
scatter(px,py,100, 'x')   %实际落脚点
hold on
scatter(pxplan,pyplan,500,'.') %计划落脚点
axis equal

subplot(1,2,2)
for i=1:length(t)
    xx=[ppx(i),x(i)];
    yy=[ppy(i),y(i)];
    zz=[0,0.8];
    plot3(xx,yy,zz,'b')
    hold on
    scatter3(x(i),y(i),0.8,'r');
    hold on
end
axis equal