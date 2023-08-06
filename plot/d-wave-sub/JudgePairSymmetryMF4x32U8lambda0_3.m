figure;
clear;

Lx=32; Ly=4;
omega = 5; g = 2.4495; Np = 0; U = 8; Numhole = Lx*Ly/8;


Dset=[10000];%bond dimension set


% ******* On site pair
D=Dset(1);
FileNamePostfix=['mf_ssh',num2str(Ly),'x',num2str(Lx),'U',num2str(U),'g',num2str(g),'omega',num2str(omega),'Np',num2str(Np),'hole',num2str(Numhole),'D',num2str(D),'.json'];
OnSitePairData = jsondecode(fileread(['../../data/onsitesc',FileNamePostfix]));
distance=zeros(1,numel(OnSitePairData));
for i=1:numel(OnSitePairData)
    distance(i) = (OnSitePairData{i}{1}(2)-OnSitePairData{i}{1}(1))/(2*Np+1)/Ly;
end

scs_onsite=zeros(numel(Dset),numel(OnSitePairData));
for j = 1:numel(Dset)
    D = Dset(j);
    FileNamePostfix=['mf_ssh',num2str(Ly),'x',num2str(Lx),'U',num2str(U),'g',num2str(g),'omega',num2str(omega),'Np',num2str(Np),'hole',num2str(Numhole),'D',num2str(D),'.json'];
    OnSitePairData = jsondecode(fileread(['../../data/onsitesc',FileNamePostfix]));
    for i=1:numel(OnSitePairData)
        scs_onsite(j,i) = OnSitePairData{i}{2};
    end
end

hs=loglog(distance,scs_onsite,'s');hold on;


% ****** yy bond pair ********** %
D=Dset(1);
FileNamePostfix=['mf_ssh',num2str(Ly),'x',num2str(Lx),'U',num2str(U),'g',num2str(g),'omega',num2str(omega),'Np',num2str(Np),'hole',num2str(Numhole),'D',num2str(D),'.json'];
A = jsondecode(fileread(['../../data/scsyya',FileNamePostfix]));
distance=zeros(1,numel(A));
for i=1:numel(A)
    distance(i) = (A{i}{1}(3)-A{i}{1}(1))/(2*Np+1)/Ly;
end

scsyy=zeros(numel(Dset),numel(A));
for j = 1:numel(Dset)
    D = Dset(j);
    FileNamePostfix=['mf_ssh',num2str(Ly),'x',num2str(Lx),'U',num2str(U),'g',num2str(g),'omega',num2str(omega),'Np',num2str(Np),'hole',num2str(Numhole),'D',num2str(D),'.json'];
    A = jsondecode(fileread(['../../data/scsyya',FileNamePostfix]));
    B = jsondecode(fileread(['../../data/scsyyb',FileNamePostfix]));
    C = jsondecode(fileread(['../../data/scsyyc',FileNamePostfix]));
    D = jsondecode(fileread(['../../data/scsyyd',FileNamePostfix]));
    for i=1:numel(A)
        scsyy(j,i) = A{i}{2}+B{i}{2}+C{i}{2}+D{i}{2};
    end
end
[order_distance, I] = sort(distance);
order_scsyy = scsyy(I);

h0=loglog(order_distance,(order_scsyy),'-x');hold on;




%%======Pyx======%%
% distance = distance(1:numel(distance)/4);
scsPS=zeros(numel(Dset),numel(A));
for j = 1:numel(Dset)
    D = Dset(j);
    FileNamePostfix=['mf_ssh',num2str(Ly),'x',num2str(Lx),'U',num2str(U),'g',num2str(g),'omega',num2str(omega),'Np',num2str(Np),'hole',num2str(Numhole),'D',num2str(D),'.json'];
    A = jsondecode(fileread(['../../data/scsPSa',FileNamePostfix]));
    B = jsondecode(fileread(['../../data/scsPSb',FileNamePostfix]));
    C = jsondecode(fileread(['../../data/scsPSc',FileNamePostfix]));
    D = jsondecode(fileread(['../../data/scsPSd',FileNamePostfix]));
    for i=1:numel(A)
        scsPS(j,i) = A{i}{2}+B{i}{2}+C{i}{2}+D{i}{2};
    end
end

PS_data_size = numel(A);
Pyx_data_size = numel(A)/4;
Pyyp_data_size = numel(A)/4;

scsyx=scsPS(:, 1:Pyx_data_size);
scsyx_ex = zeros(1, Pyx_data_size);

distance = distance(1:numel(distance)/4);
if numel(distance) ~= Pyx_data_size
    error("numel(distance) ~= Pyx_data_size \n");
end
h1=plot(distance, abs(scsyx),'-o');hold on;
h1prime =plot(distance(scsyx<0), abs(scsyx(scsyx<0)),'o');hold on;
set(h1prime,'MarkerEdgeColor',get(h1,'color'));
set(h1prime,'MarkerFaceColor',get(h1,'color'));

%%======Pyy'======%%
scsyyp = scsPS(:, Pyx_data_size+1:2*Pyx_data_size);
h2=plot(distance, abs(scsyyp),'-^');hold on;
h2prime =plot(distance(scsyyp<0), abs(scsyyp(scsyyp<0)),'^');hold on;
set(h2prime,'MarkerEdgeColor',get(h2,'color'));
set(h2prime,'MarkerFaceColor',get(h2,'color'));


% %%======Pyy''======%%
% scsyypp = scsPS(:, 2*Pyx_data_size+1:3*Pyx_data_size);
% h3=plot(distance, scsyypp,'-s');hold on;


l=legend([hs, h0,h2,h1],'$\Phi_{s}(x) > 0$', '$\Phi_{yy}(x) > 0$',  '$|\Phi_{yy}^{\prime}(x)|$, solid circle negative',  '$|\Phi_{yx}(x)|$, solid circle negative');
set(l,'Box','off');set(l,'Interpreter','latex');
set(l,'Fontsize',36);
set(l,'Location','SouthWest');



set(gca,'fontsize',24);
set(gca,'linewidth',1.5);
set(get(gca,'Children'),'linewidth',2); % Set line width 1.5 pounds
set(get(gca,'Children'),'markersize',15); % Set line width 1.5 pounds
xlabel('$x$','Interpreter','latex');
ylabel('$|\langle\Delta_s^\dagger(x)\Delta_s(0)\rangle|$','Interpreter','latex');
set(get(gca,'XLabel'),'FontSize',24); 
set(get(gca,'YLabel'),'FontSize',24); 



