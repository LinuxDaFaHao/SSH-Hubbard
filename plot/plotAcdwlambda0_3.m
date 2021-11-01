Lx = [24, 32, 40, 48];
A_cdw = [0.020132,0.023635, 0.021030, 0.017995];
loglog(Lx, A_cdw,'o');hold on;

p = fit(log(Lx(2:4)'),log(A_cdw(2:4)'),'poly1');
fprintf('Kc=%.5f\n',-p.p1*2);
x = fit_x(1):0.5:fit_x(end);
fl=loglog(Lx,exp(p.p2)*Lx.^p.p1,'-.');
T=text(8,4.5e-3,['$K_{c}=',num2str(-p.p1*2),'$']);
set(T,'Interpreter','latex');set(T,'Fontsize',24);



set(gca,'fontsize',24);
set(gca,'linewidth',1.5);
set(get(gca,'Children'),'linewidth',2); % Set line width 1.5 pounds
xlabel('$L_x$','Interpreter','latex');
ylabel('$A_{cdw}$','Interpreter','latex');
set(get(gca,'XLabel'),'FontSize',24); 
set(get(gca,'YLabel'),'FontSize',24); 
