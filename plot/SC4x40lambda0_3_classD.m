Lx=40; Ly=4;
omega = 5; 
g = 2.4495;
Np=3;

U = 8; Numhole = Lx*Ly/8;

% begin = 8;
% endx= 32;

 begin = 11;
 endx= 34;

Dset=[8000,10000,12000, 14000,15000];%bond dimension set

trunc_err=1e7*[3.46e-6,2.86e-6,2.47e-06,2.19e-06,2.03e-6];

Db=Dset(1);
FileNamePostfix=['begin',num2str(begin),'end',num2str(endx),...
    'ssh',num2str(Ly),'x',num2str(Lx),'U',num2str(U),'g',num2str(g),...
    'omega',num2str(omega),'Np',num2str(Np),'hole',num2str(Numhole),'D',num2str(Db),'.json'];
A = jsondecode(fileread(['../data/scsyya',FileNamePostfix]));
distance=zeros(1,numel(A));
for i=1:numel(A)
    distance(i) = (A{i}{1}(3)-A{i}{1}(1))/(2*Np+1)/Ly;
end

scsyy=zeros(numel(Dset),numel(A));
for j = 1:numel(Dset)
    Db = Dset(j);
    FileNamePostfix=['begin',num2str(begin),'end',num2str(endx),...
    'ssh',num2str(Ly),'x',num2str(Lx),'U',num2str(U),'g',num2str(g),...
    'omega',num2str(omega),'Np',num2str(Np),'hole',num2str(Numhole),'D',num2str(Db),'.json'];
    A = jsondecode(fileread(['../data/scsyya',FileNamePostfix]));
    B = jsondecode(fileread(['../data/scsyyb',FileNamePostfix]));
    C = jsondecode(fileread(['../data/scsyyc',FileNamePostfix]));
    D = jsondecode(fileread(['../data/scsyyd',FileNamePostfix]));
    for i=1:numel(A)
        scsyy(j,i) = A{i}{2}+B{i}{2}+C{i}{2}+D{i}{2};
    end
end

h=loglog(distance,scsyy,'x');hold on;



scsyy_ex=zeros(size(distance));
%fit_x=[1/8,1/10,1/12,1/14];%1/D
% fit_x=1e7*[ 5.59e-06, 4.56e-06];%Site  560
fit_x = trunc_err;
for i=1:numel(distance)
    p = fit(fit_x(1:end)',scsyy(1:end,i),'poly2');
    scsyy_ex(i)=p.p3;
end

loglog(distance, scsyy_ex,'o');hold on;
I=find(distance==Lx/2);
fprintf("<Delta_yy^dag Delta_yy>(Lx/2) = %.6f\n",mean(scsyy_ex(I)));


fit_x=[6,7,10,11];
fit_y=zeros(size(fit_x));
for i=1:numel(fit_x)
    I = find(distance==fit_x(i));
    fit_y(i)=mean(scsyy_ex(I));
end


p = fit((fit_x'),log(abs(fit_y')),'poly1');
fprintf('correlation length=%.5f\n',-1/p.p1);
x = fit_x;
loglog(x,exp(p.p2+p.p1*x),'-.');%fitted line
T=text(10.2,6e-3,['$\xi=',num2str(-1/p.p1),'$']);
set(T,'Interpreter','latex');set(T,'Fontsize',24);


p = fit(log(fit_x'),log(abs(fit_y')),'poly1');
fprintf('Ksc=%.5f\n',-p.p1);
x = fit_x(1):0.5:fit_x(end);
fl=loglog(x,exp(p.p2)*x.^p.p1,'-.');
T=text(10,2.5e-3,['$K_{sc}=',num2str(-p.p1),'$']);
set(T,'Interpreter','latex');set(T,'Fontsize',24);




l=legend(h,'$D=8000$', '$10000$','$12000$','$14000$','$15000$');
set(l,'Box','off');set(l,'Interpreter','latex');
set(l,'Fontsize',24);
set(l,'Location','SouthWest');


set(gca,'fontsize',24);
set(gca,'linewidth',1.5);
set(get(gca,'Children'),'linewidth',2); % Set line width 1.5 pounds
xlabel('$x$','Interpreter','latex');
ylabel('$|\langle\Delta_s^\dagger(x)\Delta_s(0)\rangle|$','Interpreter','latex');
set(get(gca,'XLabel'),'FontSize',24); 
set(get(gca,'YLabel'),'FontSize',24); 





