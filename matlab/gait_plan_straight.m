clear
clc

Tsup=0.8; %每步时间
zc=0.8; %质心高度
g=9.8;
Tc=sqrt(zc/g);
S=sinh(Tsup/Tc);
C=cosh(Tsup/Tc);
dt=0.01; 
a=10;b=1; %权重系数

%步长
sx=[0,0.2,0.2,0.2,0];
sy=[0.2,0.3,0.1,0.2,0.2];
%计划落脚点
pxplan=zeros(1,length(sx));
pyplan=zeros(1,length(sy));
pxp=0;
pyp=0;
for i=1:length(sx)
    pxp=pxp+sx(i);
    pyp=pyp-(-1)^i*sy(i);
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
    %期望步行单元末状态（相对世界坐标系）
    xe=sx(i+1)/2;
    ye=(-1)^i*sy(i+1)/2;
    vxe=(C+1)/(Tc*S) *xe;
    vye=(C-1)/(Tc*S) *ye;
    xe=xe+pxplan(i);
    ye=ye+pyplan(i);
    %步行单元初状态（相对世界坐标系）
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