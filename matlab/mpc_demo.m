clear
clc

m = 1;
dt = 0.01;
A = [1,dt;0,1];
B = [0,dt/m]';

n = size(A,1);
m = size(B,2);
p = 1000; %预测步数

x0 = [1,-1]'; %初始状态
R = zeros(n*p,1); %参考状态
R(1:2:n*p)= sin(dt:dt:dt*p);
R(2:2:n*p)= cos(dt:dt:dt*p);

Phi = zeros(n*p,n);
for i=1:p
    Phi((i-1)*n+(1:n),:) = A^i;
end

Theta = zeros(n*p,m*p);
for i=1:p
    for j=1:i
        Theta((i-1)*n+(1:n),(j-1)*m+(1:m)) = A^(i-j) * B;
    end
end

E = Phi*x0 - R;

Q = eye(n*p);
Q(n*p-1, n*p-1)=1;
Q(n*p, n*p)=1;
W = eye(m*p) * 0.1;

H = 2 * (Theta' * Q * Theta + W);
g = 2 * E' * Q * Theta;

U_lb = -1 * ones(m*p,1);
U_ub = 1 * ones(m*p,1);

U = quadprog(H,g',[],[],[],[],U_lb,U_ub,x0);
X = Phi*x0 + Theta*U;

t = dt:dt:dt*p;
x = X(1:2:n*p);
x_dot = X(2:2:n*p);
x_ref = R(1:2:n*p);
x_dot_ref = R(2:2:n*p);

subplot(1,3,1)
plot(t,x,'r','lineWidth',2)
hold on
plot(t,x_ref,'k--','lineWidth',2)
xlabel('time (s)');
ylabel('x');
legend('act','ref');
set(gca,'FontSize',15)

subplot(1,3,2)
plot(t,x_dot,'r','lineWidth',2);
hold on
plot(t,x_dot_ref,'k--','lineWidth',2);
xlabel('time (s)');
ylabel('$\dot{x}$','Interpreter','latex');
set(gca,'FontSize',15)

subplot(1,3,3)
plot(t,U,'r','lineWidth',2);
xlabel('time (s)');
ylabel('u');
set(gca,'FontSize',15)

set(gcf,'color',[1 1 1])