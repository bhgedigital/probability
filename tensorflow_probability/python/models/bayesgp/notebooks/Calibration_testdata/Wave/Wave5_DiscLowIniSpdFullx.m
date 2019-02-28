clear all
clc

% Design parameters - x = [0 < x < 3]

% Uncertain parameters - c [0.05 < c < 0.5]

Ndesign = 1;
Ncalib = 1;

Bounds = [0,3;0.05,0.5];

ctrue = 0.1;
% ktrue = 3;

% Generate Experimental Data
Npoints = 30;
X = linspace(Bounds(1,1),Bounds(1,2),Npoints);

% figure
% plot(X,sin(pi*X))

f=fopen('test_doe.txt','w+');
fprintf(f,'x\n');
fprintf(f,'%f\n',X');
fclose(f);

Time = 0:0.2:20;

cd ./test-data
for i=1:Npoints

    clear U
    for timec = 1:size(Time,2)
    
        U(timec,1) = Time(timec);
        
        t = Time(timec);
               
        U(timec,2) = 0.5*(sin(pi*(X(i) - ctrue*Time(timec))) + sin(pi*(X(i) + ctrue*Time(timec)))) + X(i)*Time(timec)/20;
    end
    
% % %     Data(:,:,i) = U;
    TrueDataSurf(:,i) = U(:,2);
    
    clear name
    name = sprintf('test_output_%d.dat',i);
    
f=fopen(name,'w+');
fprintf(f,'Time\tu\n');
fprintf(f,'%f\t%f\n',[U(:,1) U(:,2)]');
fclose(f);

end
cd ..

figure
surf(X,Time,TrueDataSurf)
xlabel('x')
ylabel('time')
zlabel('Displacement')


% Generate Simulation Data
Npointssim = 50;
Datanormsim = lhsdesign(Npointssim,Ndesign+Ncalib);
Datasim = zeros(Npointssim,Ncalib);

for i = 1:Npointssim
    for j = 1:Ndesign+Ncalib
        Datasim(i,j) = Datanormsim(i,j)*(Bounds(j,2) - Bounds(j,1)) + Bounds(j,1);
    end
end



f=fopen('sim_doe.txt','w+');
fprintf(f,'x\tc\n');
fprintf(f,'%f\t%f\n',[Datasim(:,1),Datasim(:,2)]');
fclose(f);



cd ./sim-data
for i=1:Npointssim

    clear U
    for timec = 1:size(Time,2)
    
        U(timec,1) = Time(timec);
        
        t = Time(timec);
             
        U(timec,2) = sin(pi*(Datasim(i,1) - Datasim(i,2)*Time(timec)));
    end
    
    clear name
    name = sprintf('sim_output_%d.dat',i);

    
    
    
f=fopen(name,'w+');
fprintf(f,'Time\tu\n');
fprintf(f,'%f\t%f\n',[U(:,1) U(:,2)]');
fclose(f);

end
cd ..


bayesian_call('input_data.txt')

 
 
 
 
 
 
 
 
 
 
% % 
% % % Generate Simulated Data
% % Npointssim = 50;
% % Datanormsim = lhsdesign(Npointssim,Ndesign+Ncalib);
% % Datasim = zeros(Npointssim,Ndesign+Ncalib);
% % 
% % for i = 1:Npointssim
% %     for j = 1:Ndesign+Ncalib
% %         Datasim(i,j) = Datanormsim(i,j)*(Bounds(j,2) - Bounds(j,1)) + Bounds(j,1);
% %     end
% % end
% % 
% % for i = 1:Npointssim
% % 
% % x = Datasim(i,1);
% % beta = Datasim(i,2);
% % 
% % ShockPos = pi - asin(sqrt(1-beta^2));
% % 
% % if x <= ShockPos
% %     Datasim(i,Ndesign+Ncalib+1) = sin(x);
% % end
% % 
% % if x > ShockPos
% %    Datasim(i,Ndesign+Ncalib+1) = -1*sin(x);
% % end
% %  
% % end
% % 
% % f=fopen('sim.dat','w+');
% % fprintf(f,'x\tbeta\tu\n');
% % fprintf(f,'%f\t%f\t%f\n',[Datasim(:,1) Datasim(:,2) Datasim(:,3)]');
% % fclose(f);
% % 
% % 
% % figure
% % plot(Data(:,1),Data(:,2),'*r');
% % hold on
% % plot(Datasim(:,1),Datasim(:,3),'.');
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % 
% % load pout
% % 
% % 
% % 
% % 
% % 
% % % --Added by Ankur
% % 'Pout saved'
% % 
% % 
% % % showPvals(pout.pvals)
% % % 
% % % % Prediction plot
% % % 
% % % 
% % % BHM = [pout.pyhat_mean,pout.pyhat_std];
% % % 
% % % SIM = [pout.peta_mean,pout.peta_std];
% % % 
% % % 
% % % % for i = 1:Ndesign
% % % %     
% % % %     figure(i)
% % % %     plot(Test(:,i),Test(:,end),'.r');
% % % %     hold on
% % % %     errorbar(Test(:,i),BHM(:,1),BHM(:,2),'.');
% % % %     hold on
% % % %     errorbar(Test(:,i),SIM(:,1),SIM(:,2),'.k');
% % % % 
% % % % end
% % % 
% % % % Sensitivity Analysis
% % 
% % % sens=gSens(pout,'pvec',pout.pvec,'varlist','all');
% % % pout.sens = sens;
% % % save pout pout
% % % % 
% % 
% % 
% % % % Sensitivity Analysis
% % % 
% % % sens=gSens(pout,'pvec',pout.pvec,'varlist','all');
% % % pout.sens = sens;
% % % save pout pout
% % % 
% % % Smain = pout.sens.smePm;
% % % Stotal = pout.sens.stePm;
% % % 
% % % Sjoint = pout.sens.siePm;
% % % 
% % % Allsens1 = [Smain,Sjoint]';
% % % 
% % % [Allsens2,Ind] = sort(Allsens1,'descend');
% % % 
% % % 
% % % Labels = [pout.labels.x;pout.labels.theta];
% % % 
% % % 
% % % c=0;
% % % clear A
% % % for i = 1:(Ndesign+Ntheta)-1
% % %     for j = i+1:(Ndesign+Ntheta)
% % %     
% % %         c=c+1;
% % %         A(c,1) = strcat(Labels(i),',',Labels(j));
% % %     end
% % % end
% % % 
% % % LabelsAll = [Labels;A];
% % % 
% % % LabelsAllsort = LabelsAll(Ind);
% % % 
% % % Pick = 3;
% % % Allsens = Allsens2(1:Pick);
% % % LabelsFinal = LabelsAllsort(1:Pick-1);
% % % LabelsFinal = [LabelsFinal;'Others'];
% % % 
% % % explode = zeros(size(Allsens));
% % % 
% % % clear c;clear offset
% % % [c,offset] = max(Allsens);
% % % explode(offset) = 1;
% % % 
% % % figure
% % % h = pie(Allsens,explode);
% % % colormap cool
% % % 
% % % textObjs = findobj(h,'Type','text');
% % % oldStr = get(textObjs,{'String'});
% % % val = get(textObjs,{'Extent'});
% % % oldExt = cat(1,val{:});
% % % 
% % % % Names = {'Product X: ';'Product Y: ';'Product Z: '};
% % % Names = LabelsFinal;
% % % newStr = strcat(Names,oldStr);
% % % set(textObjs,{'String'},newStr)
% % % 
% % % val1 = get(textObjs, {'Extent'});
% % % newExt = cat(1, val1{:});
% % % offset = sign(oldExt(:,1)).*(newExt(:,3)-oldExt(:,3))/2;
% % % pos = get(textObjs, {'Position'});
% % % textPos =  cat(1, pos{:});
% % % textPos(:,1) = textPos(:,1)+offset;
% % % set(textObjs,{'Position'},num2cell(textPos,[3,2]))
% % % 
% % % 
% % % 
% % % 
% % % 
% % % % Output pdf
% % % clear a1;clear a2;clear a3;
% % % [a1,a2,a3] = size(pout.pyhat);
% % % Pdf = zeros(a1*a3,2);
% % % 
% % % c=0;
% % % for i = 1:a1
% % %     for j = 1:a3
% % %         c=c+1;
% % %     Pdf(c,1) = pout.pyhat(i,1,j);
% % % 
% % %     Pdf(c,2) = pout.peta(i,1,j);
% % %     end
% % % end
% % % 
% % % 
% % %  nbin = 10;
% % % [Ankur1,Ankur2] = hist(Pdf(:,1),nbin);
% % % [Ankur3,Ankur4] = hist(Pdf(:,2),nbin);
% % %                 
% % % figure
% % % plot(Ankur2,Ankur1,'r','linewidth',2)    
% % % hold on
% % % plot(Ankur4,Ankur3,'linewidth',2)    
% % % grid on
% % % legend('BHM output pdf','SIM output pdf');
% % % 
% % % 
% % % 
% % % 
% % % 
% % % 
% % % 
% % % 
% % 
% 
% 
% 
% 
% 
% 
% 
% 
