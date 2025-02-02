figure;
Lx=48; Ly=4;
omega = 5; g =   1.414; Np = 2; U = 8; Numhole = Lx*Ly/8;
addpath('../');

Dset=[6000, 8000, 10000,12000,16000];
trunc_err=[6.4665e-07, 4.9815e-07, 4.7861e-07, 4.7547e-07,3.7811e-07];
D=Dset(1);
FileNamePostfix=['ssh',num2str(Ly),'x',num2str(Lx),'U',num2str(U),'g',num2str(g),'omega',num2str(omega),'Np',num2str(Np),'hole',num2str(Numhole),'D',num2str(D),'.json'];
ChargeDensityData = jsondecode(fileread(['../../data/nf',FileNamePostfix]));
distance = zeros(1, size(ChargeDensityData,1));
for i=1:numel(distance)
    FermionSite = Site2FermionSite(ChargeDensityData(i, 1),Ly,Np);
    distance(i) = fix((FermionSite)/Ly);
end

ChargeDensity = zeros( numel(Dset), numel(distance));

for j = 1:numel(Dset)
    D=Dset(j);
    FileNamePostfix=['ssh',num2str(Ly),'x',num2str(Lx),'U',num2str(U),'g',num2str(g),'omega',num2str(omega),'Np',num2str(Np),'hole',num2str(Numhole),'D',num2str(D),'.json'];
    ChargeDensityData = jsondecode(fileread(['../../data/nf',FileNamePostfix]));
    
    ChargeDensity(j, :) = transpose(ChargeDensityData(:,2));
    
    charge_density_avg = mean(ChargeDensityData(:,2));
    if (charge_density_avg-0.875) > 1e-8
        error("charge density is not right");
    end
end
ChargeDensity = (ChargeDensity+ChargeDensity(:, end:-1:1))/2;
% h = plot(distance, ChargeDensity,'-x'); hold on;

ChargeDensity_ex = zeros(1, numel(distance) );

fit_x=1e7*trunc_err;
for i=1:numel(distance)
    p = fit(fit_x',ChargeDensity(:,i),'poly2');
    ChargeDensity_ex(i)=p.p3;
end

hex = plot(distance, ChargeDensity_ex,'-o'); hold on;


% l=legend([h; hex], '$10000$', '$12000$','$14000$','$16000$', '$180000$', '$\infty$');
% set(l,'Box','off');set(l,'Interpreter','latex');
% set(l,'Fontsize',24);
% set(l,'Location','SouthWest');


set(hex, 'Markersize',9);


set(gca,'fontsize',24);
set(gca,'linewidth',1.5);
set(get(gca,'Children'),'linewidth',2); % Set line width 1.5 pounds
xlabel('$x$','Interpreter','latex');
ylabel('Charge Density','Interpreter','latex');
set(get(gca,'XLabel'),'FontSize',24); 
set(get(gca,'YLabel'),'FontSize',24); 
