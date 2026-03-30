clear
clc

m=1;
dt = 0.001;
A = [1,dt;0,1];
B = [0,dt/m]';

Q = [1,0;0,1];
R = 0.1;

K = dlqr(A,B,Q,R)

% simulate the dynamic process
x0 = [0,1]';
t = 0:dt:20;
N = length(t);
x = zeros(2,N);
x(:,1) = x0;
% x_ref = zeros(2,N);
% x_ref = [ones(1,N);zeros(1,N)];
x_ref = [sin(t);cos(t)];
u = zeros(1,N-1);
for i=1:(N-1)
    u(i) = -K * (x(:,i)-x_ref(:,i));
    x(:,i+1) = A * x(:,i) + B * u(i);
end

subplot(1,3,1)
plot(t,x(1,:),'r','lineWidth',2)
hold on
plot(t,x_ref(1,:),'k--','lineWidth',2)
xlabel('time (s)');
ylabel('x');
legend('act','ref');
set(gca,'FontSize',15)

subplot(1,3,2)
plot(t,x(2,:),'r','lineWidth',2);
hold on
plot(t,x_ref(2,:),'k--','lineWidth',2);
xlabel('time (s)');
ylabel('$\dot{x}$','Interpreter','latex');
set(gca,'FontSize',15)

subplot(1,3,3)
plot(t(2:end),u,'r','lineWidth',2);
xlabel('time (s)');
ylabel('u');
set(gca,'FontSize',15)

set(gcf,'color',[1 1 1])