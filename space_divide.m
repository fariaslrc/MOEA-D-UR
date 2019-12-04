function [Population,Z,EP]=space_divide(Global,EP,Population,Z,W_URP,W,divisions_in_space,mini_generation,nEP)
	divisoes_k_means=divisions_in_space;
	[idx,C] = kmeans(Population.objs,divisoes_k_means); 
	
	allInd=[Population EP];
	%% For each group
	for indice_grupo_atual=1:max(idx)		
		%% Definir o grupo corrente
		grupoAtual=Population(idx(:)==indice_grupo_atual);
		group_size=length(grupoAtual);
		
		%% Generate the normalized weight vectors
		W_normalizado=W_URP(idx(:)==indice_grupo_atual,:);
		w_maximos=max(W_normalizado);
		w_minimos=min(W_normalizado);
		W_normalizado= (W_URP.*repmat((w_maximos-w_minimos),length(W_URP),1)) + repmat(w_minimos,length(W),1); %normalization in variable range(x,y)
		
		%% Associar individuos aos pesos
		grupoAtual=associate_newW2allInd(allInd,W_normalizado,Z);
		
		[Pretendentes,Z,EP]=mini_moead(Global,EP,Population,grupoAtual,Z,W_normalizado,group_size,mini_generation,nEP);
		allInd=[allInd Pretendentes];
	end
	
	Population = combine_allInd2Population(Global,Population,allInd,W,Z);
end

function Population = associate_newW2allInd(Pretendentes,W,Z)
   
	Combine = Pretendentes;
    CombineObj = abs(Combine.objs-repmat(Z,length(Combine),1));
    g = zeros(length(Combine),size(W,1));
    for i = 1 : size(W,1)
        g(:,i) = max(CombineObj.*repmat(W(i,:),length(Combine),1),[],2);
    end
    % Choose the best solution for each subproblem
    [~,best]   = min(g,[],1);
    Population = Combine(best);
end

function [grupoAtual,Z,EP]=mini_moead(Global,EP,Population,grupoAtual,Z,W,group_size,mini_generation,nEP)

	generation = 1;
	plotallpop=[Population grupoAtual];
    flag_saida=0;
	while Global.NotTermination(plotallpop) && generation <= mini_generation
		Offsprings(1:length(group_size)) = INDIVIDUAL();
        for i=1:length(group_size)
			% Choose the parents
			parents=unique(grupoAtual);
            if length(parents) > 2
                idx_parents = randperm(length(parents));
                P = randperm(Global.N);

                % Generate an offspring
				Offsprings(i) = GAhalf(parents(idx_parents(1:2)));
				%Offsprings(i) = DE(Population(idx_parents(1)),Population(idx_parents(2)),Population(idx_parents(3)));
				
                % Update the ideal point
                Z = min(Z,Offsprings(i).obj);

                % Update the solutions in P by Tchebycheff approach
                g_old = max(abs(grupoAtual(P).objs-repmat(Z,length(P),1)).*W(P,:),[],2);
                g_new = max(repmat(abs(Offsprings(i).obj-Z),length(P),1).*W(P,:),[],2);
                grupoAtual(P(find(g_old>=g_new,length(P)))) = Offsprings(i);
            else
                flag_saida=1;
                break;
            end
        end
        if flag_saida==0
            % EP update
            EP = updateEP(EP,Offsprings,nEP);
            
            % aux functions
            plotallpop=[Population grupoAtual];
        else %flag_saida==1;
            break;
        end
        generation=generation+1;
	end
end

function Population = combine_allInd2Population(Global,Population,Pretendentes,W,Z)
   
    Combine=unique(Pretendentes);
    if length(Combine) < Global.N
        Combine = Pretendentes;
    end
    CombineObj = abs(Combine.objs-repmat(Z,length(Combine),1));
    g = zeros(length(Combine),size(W,1));
    for i = 1 : size(W,1)
        g(:,i) = max(CombineObj.*repmat(W(i,:),length(Combine),1),[],2);
    end
    % Choose the best solution for each subproblem
    [~,best]   = min(g,[],1);
    Pretendentes = Combine(best);
    Pretendentes=unique(Pretendentes);
	
    %fprintf("%d  ",length(unique(Population)));
    %melhorias=0;
    %iguais=0;
    for i=1:length(Pretendentes)        
        % Global Replacement
        all_g_TCH=max(abs((Pretendentes(i).obj-repmat(Z,Global.N,1)).*W),[],2);
        best_g_TCH=min(all_g_TCH);
        Chosen_one = find(all_g_TCH(:,1)==best_g_TCH);            
        % Update the solutions in P by Tchebycheff approach        
        if Population(Chosen_one) ~= Pretendentes(i)
            g_old = max(abs(Population(Chosen_one).objs-repmat(Z,length(Chosen_one),1)).*W(Chosen_one,:),[],2);
            g_new = max(repmat(abs(Pretendentes(i).obj-Z),length(Chosen_one),1).*W(Chosen_one,:),[],2);
            %if g_old>=g_new
            %    melhorias=melhorias+1;
            %end
            Population(Chosen_one(g_old>=g_new)) = Pretendentes(i);
            %iguais=iguais+1;
        end
    end
    %fprintf("%d/%d  ",melhorias,iguais);
    %fprintf("%d\n",length(unique(Population)));
end