clear;

White_s_yy_data=[1.0015174506828526, 0.025132130355934434
1.97268588770865, 0.0036718891620243234
3.0045523520485595, 0.0009323163734412452
4.9772382397572095, 0.0005481388381846198
5.978755690440062, 0.0003221083797458977
7.010622154779972, 0.00021844516571914248
8.012139605462824, 0.00022307653309485483
8.983308042488618, 0.0001933959964363762
9.984825493171472, 0.0001394594077585788
10.986342943854327, 0.00013122111066108847
11.987860394537178, 0.00013128837373175466
13.019726858877087, 0.00008903621377206683
13.990895295902886, 0.00008041518294249622
14.992412746585737, 0.00009285019837439972
16.995447647951444, 0.0000473036466043967
17.9969650986343, 0.00005461845400751623
18.998482549317153, 0.000055776447708908035
19.96965098634295, 0.000031461380686962276
21.001517450682854, 0.000032128908396534725
22.033383915022767, 0.000037097743679223
23.004552352048563, 0.00002844511905782515
24.00606980273141, 0.000019689301770375758
26.00910470409712, 0.000021833324954407786
28.983308042488623, 0.00001365640438116479
31.98786039453718, 0.000009857841347297
34.992412746585735, 0.00000642366428913768
35.9939301972686, 0.000005684235492619364
39.028831562974204, 0.000004453278109652012
42.974203338391504, 0.000002675065035651925];

distance = White_s_yy_data(:,1);
P_yy = White_s_yy_data(:,2);

semilogy(distance, P_yy,'-o');hold on;
fit_x = distance(distance<33 & distance>3);
fit_y = P_yy(distance<33 & distance>3);

p = fit(log(fit_x),log(abs(fit_y)),'poly1');
fprintf('Ksc=%.5f\n',-p.p1);
x = 1:0.5:fit_x(end)+10;
fl=semilogy(x,exp(p.p2)*x.^p.p1,'-.');
T=text(10,2.5e-3,['$K_{sc}=',num2str(-p.p1),'$']);
set(T,'Interpreter','latex');set(T,'Fontsize',24);



set(gca,'fontsize',24);
set(gca,'linewidth',1.5);
set(get(gca,'Children'),'linewidth',2); % Set line width 1.5 pounds
xlabel('$d$','Interpreter','latex');
ylabel('$|\langle\Delta_s^\dagger(x)\Delta_s(0)\rangle|$','Interpreter','latex');
set(get(gca,'XLabel'),'FontSize',24); 
set(get(gca,'YLabel'),'FontSize',24); 



