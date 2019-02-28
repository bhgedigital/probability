clear all
clc

% Uncertain parameters - E,h

Ncalib = 2;

Bounds = [1.7e11,2.3e11;0.01,0.1];

BeamAnsys = xlsread('BeamAnsys.xls');

% Generate Experimental Data
Etrue = 2e11;
htrue = 0.05;
Exp(1) = 163.18;
Exp(2) = 978.27;
Exp(3:6) = [BeamAnsys(80,2),BeamAnsys(60,2),BeamAnsys(40,2),BeamAnsys(1,2)];
Exp(7:10) = [BeamAnsys(80,3),BeamAnsys(60,3),BeamAnsys(40,3),BeamAnsys(1,3)];

f=fopen('obs.dat','w+');
fprintf(f,'omega1\tomega2\tmode1x1\tmode1x2\tmode1x3\tmode1x4\tmode2x1\tmode2x2\tmode2x3\tmode2x4\n');
fprintf(f,'%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n',[Exp(:,1) Exp(:,2) Exp(:,3) Exp(:,4) Exp(:,5) Exp(:,6) Exp(:,7) Exp(:,8) Exp(:,9) Exp(:,10)]');
fclose(f);


% Generate Simulated Data
Npointssim = 20;
Datanormsim = lhsdesign(Npointssim,Ncalib);
Datasim = zeros(Npointssim,Ncalib);

for i = 1:Npointssim
    for j = 1:Ncalib
        Datasim(i,j) = Datanormsim(i,j)*(Bounds(j,2) - Bounds(j,1)) + Bounds(j,1);
    end
end

L = 0.5;
b = 0.1;
Beta1 = 1.8751/L;
Beta2 = 4.6941/L;
Beta1LSq = 1.8751^2;
Beta2LSq = 4.6941^2;
rho = 7850;
POS = [80,60,40,1];

clear x
% Find NormConst
integralMode1=quad(@(x)exact_int(x,Beta1,L),0,L);

NormConstMode1 = sqrt(1/(rho*b*integralMode1));

integralMode2=quad(@(x)exact_int(x,Beta2,L),0,L);

NormConstMode2 = -1*sqrt(1/(rho*b*integralMode2));



for i = 1:Npointssim

    Term = sqrt((Datasim(i,1)*Datasim(i,2)^2)/(12*rho*(L^4)));
    Datasim(i,Ncalib+1) = (Beta1LSq*Term)/(2*pi);
    Datasim(i,Ncalib+2) = (Beta2LSq*Term)/(2*pi);
    
    C = (sin(Beta1*L) + sinh(Beta1*L))/(cos(Beta1*L) + cosh(Beta1*L));
    for pp = 1:4
    x = BeamAnsys(POS(pp),1);
    Datasim(i,Ncalib+2+pp) = NormConstMode1*sqrt(1/Datasim(i,2))*(sin(Beta1*x) - sinh(Beta1*x) - C*(cos(Beta1*x) - cosh(Beta1*x)));
    end
   
    C = (sin(Beta2*L) + sinh(Beta2*L))/(cos(Beta2*L) + cosh(Beta2*L));
    for pp = 1:4
    x = BeamAnsys(POS(pp),1);
    Datasim(i,Ncalib+2+4+pp) = NormConstMode1*sqrt(1/Datasim(i,2))*(sin(Beta2*x) - sinh(Beta2*x) - C*(cos(Beta2*x) - cosh(Beta2*x)));
    end  
    
end

f=fopen('sim.dat','w+');
fprintf(f,'E\th\tomega1\tomega2\tmode1x1\tmode1x2\tmode1x3\tmode1x4\tmode2x1\tmode2x2\tmode2x3\tmode2x4\n');
fprintf(f,'%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\n',[Datasim(:,1) Datasim(:,2) Datasim(:,3) Datasim(:,4) Datasim(:,5) Datasim(:,6) Datasim(:,7) Datasim(:,8) Datasim(:,9) Datasim(:,10) Datasim(:,11) Datasim(:,12)]');
fclose(f);


bayesian_call('input_data.txt')

% load pout
% 
% % --Added by Ankur
% 'Pout saved'
% 
% 
% % showPvals(pout.pvals)
% 
% 
% % Sensitivity Analysis
% 
% sens=gSens(pout,'pvec',pout.pvec,'varlist','all');
% pout.sens = sens;
% save pout pout
% 
% Smain = pout.sens.smePm;
% Stotal = pout.sens.stePm;
% 
% Sjoint = pout.sens.siePm;
% 
% Allsens1 = [Smain,Sjoint]';
% 
% [Allsens2,Ind] = sort(Allsens1,'descend');
% 
% 
% Labels = [pout.labels.x;pout.labels.theta];
% 
% 
% c=0;
% clear A
% for i = 1:(Ndesign+Ntheta)-1
%     for j = i+1:(Ndesign+Ntheta)
%     
%         c=c+1;
%         A(c,1) = strcat(Labels(i),',',Labels(j));
%     end
% end
% 
% LabelsAll = [Labels;A];
% 
% LabelsAllsort = LabelsAll(Ind);
% 
% Pick = 3;
% Allsens = Allsens2(1:Pick);
% LabelsFinal = LabelsAllsort(1:Pick-1);
% LabelsFinal = [LabelsFinal;'Others'];
% 
% explode = zeros(size(Allsens));
% 
% clear c;clear offset
% [c,offset] = max(Allsens);
% explode(offset) = 1;
% 
% figure
% h = pie(Allsens,explode);
% colormap cool
% 
% textObjs = findobj(h,'Type','text');
% oldStr = get(textObjs,{'String'});
% val = get(textObjs,{'Extent'});
% oldExt = cat(1,val{:});
% 
% % Names = {'Product X: ';'Product Y: ';'Product Z: '};
% Names = LabelsFinal;
% newStr = strcat(Names,oldStr);
% set(textObjs,{'String'},newStr)
% 
% val1 = get(textObjs, {'Extent'});
% newExt = cat(1, val1{:});
% offset = sign(oldExt(:,1)).*(newExt(:,3)-oldExt(:,3))/2;
% pos = get(textObjs, {'Position'});
% textPos =  cat(1, pos{:});
% textPos(:,1) = textPos(:,1)+offset;
% set(textObjs,{'Position'},num2cell(textPos,[3,2]))
% 
% 
% 
% 
% 
% % Output pdf
% clear a1;clear a2;clear a3;
% [a1,a2,a3] = size(pout.pyhat);
% Pdf = zeros(a1*a3,2);
% 
% c=0;
% for i = 1:a1
%     for j = 1:a3
%         c=c+1;
%     Pdf(c,1) = pout.pyhat(i,1,j);
% 
%     Pdf(c,2) = pout.peta(i,1,j);
%     end
% end
% 
% 
%  nbin = 10;
% [Ankur1,Ankur2] = hist(Pdf(:,1),nbin);
% [Ankur3,Ankur4] = hist(Pdf(:,2),nbin);
%                 
% figure
% plot(Ankur2,Ankur1,'r','linewidth',2)    
% hold on
% plot(Ankur4,Ankur3,'linewidth',2)    
% grid on
% legend('BHM output pdf','SIM output pdf');
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% % BHM = [pout.pyhat_mean,pout.pyhat_std];
% 
% % SIM = [pout.peta_mean,pout.peta_std];
% 
% 
% % for i = 1:Ndesign
% %     
% %     figure(i)
% %     plot(Test(:,i),Test(:,end),'.r');
% %     hold on
% %     errorbar(Test(:,i),BHM(:,1),BHM(:,2),'.');
% %     hold on
% %     errorbar(Test(:,i),SIM(:,1),SIM(:,2),'.k');
% % 
% % end
% 
% % % Sensitivity Analysis
% % 
% % sens=gSens(pout,'pvec',pout.pvec,'varlist','all');
% % pout.sens = sens;
% % save pout pout
% % 
% % p=pout.model.p; q=pout.model.q;pu = pout.model.pu;
% % me=pout.sens.tmef.m; npred=size(me,2);
% % sd=pout.sens.tmef.sd;
% % xlabs = pout.labels.x;
% % sens = pout.sens;
% % xlabs{end+1}=pout.labels.theta{:};
% % 
% % tdat=0:1.0/(npred-1):1.0;
% % npl=min(p+q,5);j=1;
% % for j = 1:1
% %     figure(j);clf; colormap('copper');
% %     for ii=1:p+q
% %         r=squeeze(me(ii,:)); s=squeeze(sd(ii,:));
% % %         h(ii) = subplot(1,1,1);
% %         h(ii)=gPackSubplot(ceil((p+q)/5),npl,ceil(ii/npl),...
% %             mod(ii-1,npl)+1,0.3);
% %         plot(tdat,r,'b'); hold on;
% %         plot(tdat,r-2*s,'b--'); hold on; plot(tdat,r+2*s,'b--');
% %               xlabel(xlabs(ii)); ylabel('Y');
% %         set(gca,'Xgrid','on','Ygrid','on');
% %     end
% %     axisNorm(h,'xymax');
% % end
% % %    figure(3); print -depsc2 towermesens; close;
% % 
% % cou = 0;
% % for a1 = 1:(p+q)-1
% %     for a2 = a1+1:(p+q)
% %         
% %         cou=cou+1;
% % figure
% % [X,Y]=meshgrid(tdat,tdat);
% % clear Zm
% % clear Zstd
% % for kk = 1:size(tdat,2)
% %     Zm(kk,:) = sens.tjef.m(cou,kk,:);
% %     Zstd(kk,:) = sens.tjef.sd(cou,kk,:);
% % end
% %     
% % 
% % surf(X,Y,Zm,Zstd)
% % title('Joint sensitivity');
% % xlabel(xlabs(a1));ylabel(xlabs(a2));zlabel('Y')
% %     end
% % end
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% 
% 
% 
