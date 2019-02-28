function bayesian_call(filename)

%filename = 'input_data.txt';

f=fopen(filename,'r');
while ~feof(f)
    
    line = fgetl(f);
    [head,rem]=strtok(line,'=');
    
    if ~isempty(findstr(upper(head),'PATH'))
        mcmc_path=rem(2:end);
        break;
    end   
end
fclose(f);

addpath(mcmc_path);
runmcmc(filename);

return;