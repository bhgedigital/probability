clear all
clc

% Design parameters - theta [0,180] 

% Uncertain parameters - r [0.5 < r < 1.5]

Ndesign = 1;
Ncalib = 1;

Cp1nominal = 1;
Cp1bounds = [0.5,1.5];

Bounds1 = [0,180];

Bounds = Bounds1;

Test = xlsread('TrueCp.xls');

% % Generate Validation Data
% Npointstest = 100;
% Datanormtest = lhsdesign(Npointstest,Ndesign);
% Test = zeros(Npointstest,Ndesign);
% 
% for i = 1:Npointstest
%     for j = 1:Ndesign
%         Test(i,j) = Datanormtest(i,j)*(Bounds(j,2) - Bounds(j,1)) + Bounds(j,1);
%     end
% end
% 
% % Use mean values of calibration parameters Cp1mean, Cp2mean and Cp3mean
% 
% for i = 1:Npointstest
%     
%     J1 = 1;
%     J2 = J1/Test(i,1);
%     K1 = Cp1mean;
%     K2 = Cp2mean;
%     K3 = Cp3mean;    
%       
%     Test(i,Ndesign+1) = ExactFrequency(J1,J2,K1,K2,K3);
%     
%     
%     Ke = K3 +(K1*K2)/(K1+K2);    
%     Je = J1+J2;
%     
%     Simtest(i,1) = sqrt(Ke/Je);
% end
% 
% figure
% plot(Test(:,1),Test(:,2),'.')
% hold on
% plot(Test(:,1),Simtest(:,1),'.r')
% grid on
% 

% f=fopen('Validation.dat','w+');
% fprintf(f,'theta\tCp\n');
% fprintf(f,'%f\t%f\n',[Test(:,1) Test(:,2)]');
% fclose(f);


% Generate Experimental Data
Npoints = 10;
Apos = round(linspace(1,size(Test,1),Npoints));

Data = Test(Apos,:);

f=fopen('obs.dat','w+');
fprintf(f,'theta\tCp\n');
fprintf(f,'%f\t%f\n',[Data(:,1) Data(:,2)]');
fclose(f);


% Generate Simulated Data
Npointssim = 200;
Datanormsim = lhsdesign(Npointssim,Ndesign+Ncalib);
Datasim = zeros(Npointssim,Ndesign+Ncalib);

BoundsAll = [Bounds;Cp1bounds];

for i = 1:Npointssim
    for j = 1:Ndesign+Ncalib
        Datasim(i,j) = Datanormsim(i,j)*(BoundsAll(j,2) - BoundsAll(j,1)) + BoundsAll(j,1);
    end
end

for i = 1:Npointssim

    th = Datasim(i,1);
    r = Datasim(i,2);
    
    ydx = tan(th*pi/180);
    
    sin2eta = (ydx^2)/(r^2+ydx^2);    
    
    Datasim(i,Ndesign+Ncalib+1) = 1 - ((1+r)^2)*(sin2eta/(r^2 + (1-r^2)*sin2eta));
end

f=fopen('sim.dat','w+');
fprintf(f,'theta\tratio\tCp\n');
fprintf(f,'%f\t%f\t%f\n',[Datasim(:,1) Datasim(:,2) Datasim(:,3)]');
fclose(f);






bayesian_call('input_data.txt')

load pout





% --Added by Ankur
'Pout saved'


% showPvals(pout.pvals)
% 
% % Prediction plot
% 
% 
% BHM = [pout.pyhat_mean,pout.pyhat_std];
% 
% SIM = [pout.peta_mean,pout.peta_std];
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
% % Sensitivity Analysis

% sens=gSens(pout,'pvec',pout.pvec,'varlist','all');
% pout.sens = sens;
% save pout pout
% % 


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









