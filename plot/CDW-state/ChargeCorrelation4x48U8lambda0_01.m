Lx=48; Ly=4;
omega = 5; 
g = 0.4472;
Np=1;

U = 8; Numhole = Lx*Ly/8;

Dset=[10000,12000,14000,16000];%bond dimension set

trunc_err=1e7*[1.15e-6,9.74e-07, 8.89e-07, 8.26e-07];%Site278

D = Dset(1);

FileNamePostfix=['ssh',num2str(Ly),'x',num2str(Lx),'U',num2str(U),'g',num2str(g),'omega',num2str(omega),'Np',num2str(Np),'hole',num2str(Numhole),'D',num2str(D),'.json'];
ChargeCorrelation = jsondecode(fileread(['../../data/nfnf',FileNamePostfix]));
distance = zeros(1,numel(ChargeCorrelation));
for i=1:numel(ChargeCorrelation)
    FermionSite1 = Site2FermionSite(ChargeCorrelation{i}{1}(1),Ly,Np);
    FermionSite2 = Site2FermionSite(ChargeCorrelation{i}{1}(2),Ly,Np);
    distance(i)=(FermionSite2-FermionSite1)/Ly;
end



ReducedChargeCorrelation = zeros(numel(Dset), numel(ChargeCorrelation));
for j = 1:numel(Dset)
    D = Dset(j);
    FileNamePostfix=['ssh',num2str(Ly),'x',num2str(Lx),'U',num2str(U),'g',num2str(g),'omega',num2str(omega),'Np',num2str(Np),'hole',num2str(Numhole),'D',num2str(D),'.json'];
    ChargeDensity = jsondecode(fileread(['../../data/nf',FileNamePostfix]));
    ChargeCorrelation = jsondecode(fileread(['../../data/nfnf',FileNamePostfix]));
    for i=1:numel(ChargeCorrelation)
        I1=find(ChargeDensity(:,1)==ChargeCorrelation{i}{1}(1));
        I2=find(ChargeDensity(:,1)==ChargeCorrelation{i}{1}(2));
        ReducedChargeCorrelation(j, i) = ChargeCorrelation{i}{2}-ChargeDensity(I1,2)*ChargeDensity(I2,2);
    end
end

[distance, I]=sort(distance); ReducedChargeCorrelation=ReducedChargeCorrelation(:, I);
h = semilogy(distance,abs(ReducedChargeCorrelation),'x');hold on;



ReducedChargeCorrelation_ex=zeros(size(distance));
fit_x=trunc_err;
for i=1:numel(distance)
    p = fit(fit_x',ReducedChargeCorrelation(:,i),'poly2');
    ReducedChargeCorrelation_ex(i)=p.p3;
end

hex = semilogy(distance, abs(ReducedChargeCorrelation_ex),'o');hold on;





fit_x=[10,11,18,19];
%  fit_x=[6,7,10,11,14];
fit_y=zeros(size(fit_x));
for i=1:numel(fit_x)
    I = find(distance==fit_x(i));
    fit_y(i)=mean(ReducedChargeCorrelation_ex(I));
end


p = fit((fit_x'),log(abs(fit_y')),'poly1');
fprintf('correlation length=%.5f\n',-1/p.p1);
x = fit_x;
semilogy(x,exp(p.p2+p.p1*x),'-.');%fitted line
T=text(10,2.5e-3,['$\xi=',num2str(-1/p.p1),'$']);
set(T,'Interpreter','latex');set(T,'Fontsize',24);


p = fit(log(fit_x'),log(abs(fit_y')),'poly1');
fprintf('Kc=%.5f\n',-p.p1);
x = fit_x(1):0.5:fit_x(end);
% fl=semilogy(x,exp(p.p2)*x.^p.p1,'-.');
% T=text(8,4.5e-3,['$K_{c}=',num2str(-p.p1),'$']);
% set(T,'Interpreter','latex');set(T,'Fontsize',24);




l=legend(h,'$D=10000$', '$12000$','$14000$','$16000$');
set(l,'Box','off');set(l,'Interpreter','latex');
set(l,'Fontsize',24);
set(l,'Location','SouthWest');



set(gca,'fontsize',24);
set(gca,'linewidth',1.5);
set(get(gca,'Children'),'linewidth',2); % Set line width 1.5 pounds
xlabel('$x$','Interpreter','latex');
ylabel('$|\langle n(x)n(0)\rangle -\langle n(x)\rangle \langle n(0)\rangle|$','Interpreter','latex');
set(get(gca,'XLabel'),'FontSize',24); 
set(get(gca,'YLabel'),'FontSize',24); 

