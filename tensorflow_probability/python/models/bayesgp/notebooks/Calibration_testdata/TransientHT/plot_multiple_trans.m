load pout

ny = length(pout.labels.y);
ntest = length(pout.obsData);

for i = 1:ntest
    figure(i)
    x = pout.simData.orig.time_sim;
    SimLimits = pout.peta_bounds(2*i-1:2*i,:)';
    yhatLimits = pout.pyhat_bounds(2*i-1:2*i,:)';
     patch([x',x(end:-1:1)',x(1)],[SimLimits(:,1)',SimLimits(end:-1:1,2)',SimLimits(1,1)],[0.8 1 0],'LineStyle','none');
    hold on
    patch([x',x(end:-1:1)',x(1)],[yhatLimits(:,1)',yhatLimits(end:-1:1,2)',yhatLimits(1,1)],[0.5 0.5 1],'LineStyle','none');
    hold on
    
    plot(pout.simData.orig.time_sim,pout.peta_mean(i,:),'k-','LineWidth',2);
    plot(pout.simData.orig.time_sim,pout.pyhat_mean(i,:),'b-','LineWidth',2);
    hold on
    plot(pout.obsData(i).orig.time_obs,pout.obsData(i).orig.y,'ro','MarkerFaceColor','r');
%    plot(pout.simData.orig.time_sim,pout.pdelta_mean,'b-','LineWidth',2);
    h = legend('Calibrated simulator bounds','Simulator + discrepancy bounds','Calibrated simulator mean','Simulator + discrepancy mean','Test Data');
    xlabel('TIME');ylabel(pout.labels.y);
    set(gca,'FontSize',14,'FontWeight','bold','XMinorTick','on','YMinorTick','on');
    set(h,'FontSize',12,'FontWeight','normal');
    set(get(gca,'XLabel'),'FontSize',14,'FontWeight','bold');
    set(get(gca,'YLabel'),'FontSize',14,'FontWeight','bold');
    
    clear name
    name = sprintf('simdelta%d.jpg',i);
    saveas(h,name);
end
