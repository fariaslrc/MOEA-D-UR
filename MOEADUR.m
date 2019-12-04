function MOEADUR(Global)
% <algorithm> <M>
% MOEA/D with Update when Required

%------------------------------- Reference --------------------------------
% 
% 
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
	
    %% Generate the weight vectors
    [W,Global.N] = UniformlyRandomlyPoint(Global.N,Global.M);
    W_URP=W;
    W = 1./W./repmat(sum(1./W,2),1,size(W,2)); % WS-Transformation on W
	
    %% Parameter setting
    delta = 0.8; % The probability of choosing parents locally
    nr = round(ceil(Global.N/50)); % Maximum number of solutions replaced by each offspring
    T = ceil(Global.N/10); % Size of neighborhood
    divisoes_k_means=Global.M+1;
    mini_generation=2;    
	start_adaptation=0.21;
	end_adaptation=0.94;
	period=29;
	rho=0.01;
    rate_update_weight=0.05; % Ratio of updated weight vectors
    nEP = ceil(Global.N*2); % EP Size  
    
    %% Detect the neighbours of each solution
    B = pdist2(W,W);
    [~,B] = sort(B,2);
    B = B(:,1:T);

    %% Generate random population
    Population = Global.Initialization();
    Z          = min(Population.objs,[],1);    
    
    % external population
    EP = updateEP([],Population,nEP);
    
    %% Optimization
    I_old = max(abs((Population.objs-repmat(Z,Global.N,1)).*W),[],2);
    while Global.NotTermination(Population)
	
	    % For each solution	
	    Offsprings(1:Global.N) = INDIVIDUAL();
	    for i = 1 : Global.N
            % Choose the parents
            if rand < delta
				P = B(i,randperm(size(B,2)));
			else
                P = randperm(Global.N);
            end

            % Generate an offspring
            Offsprings(i) = GAhalf(Population(P(1:2)));
			
            % Update the ideal point
            Z = min(Z,Offsprings(i).obj);

			% Update the solutions in P by Tchebycheff approach
			g_old = max(abs(Population(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
			g_new = max(repmat(abs(Offsprings(i).obj-Z),length(P),1).*W(P,:),[],2);
			Population(P(find(g_old>=g_new,nr))) = Offsprings(i);
        end

	% EP update
	if Global.gen/Global.maxgen <= end_adaptation
		EP = updateEP(EP,Offsprings,nEP);			
	end
		
	if ~mod(Global.gen,period) && Global.gen/Global.maxgen >= start_adaptation && Global.gen/Global.maxgen <= end_adaptation
            % Improvement Metric
            I_new    = max(abs((Population.objs-repmat(Z,Global.N,1)).*W),[],2);
            improvement_Metric = mean(1-(I_new./I_old));
            I_old=I_new;			
            if abs(improvement_Metric)<=rho				
				% adaptive weight adjustment				
				[Population,W,B,I_old] = updateWeight(Global,Population,W,Z,T,EP,round(rate_update_weight*Global.N));              
                % division of objective space    
				[Population,Z,EP]=space_divide(Global,EP,Population,Z,W_URP,W,divisoes_k_means,mini_generation,nEP);   		
            end
        end
    end
end
