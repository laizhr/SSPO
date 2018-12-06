function [ cum_wealth, daily_incre_fact, daily_port_total,prim_res_total,iter_total] = SSPO_run(data,opts)
%{
THIS SOURCE CODE IS SUPPLIED ¡°AS IS¡± WITHOUT WARRANTY OF ANY KIND, AND ITS AUTHOR AND THE JOURNAL OF
MACHINE LEARNING RESEARCH (JMLR) AND JMLR¡¯S PUBLISHERS AND DISTRIBUTORS, DISCLAIM ANY AND ALL WARRANTIES,
INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, AND ANY WARRANTIES OR NON INFRINGEMENT. THE USER
ASSUMES ALL LIABILITY AND RESPONSIBILITY FOR USE OF THIS
SOURCE CODE, AND NEITHER THE AUTHOR NOR JMLR, NOR
JMLR¡¯S PUBLISHERS AND DISTRIBUTORS, WILL BE LIABLE FOR
DAMAGES OF ANY KIND RESULTING FROM ITS USE. Without limiting the generality of the foregoing, neither the author, nor JMLR, nor
JMLR¡¯s publishers and distributors, warrant that the Source Code will be
error-free, will operate without interruption, or will meet the needs of the
user.


This function is the main code for the Short-term Sparse Portfolio Optimization 
based on Alternating Direction Method of Multipliers [1]. It concentrates wealth 
on a small proportion of assets that have good increasing potential according to 
some empirical nancial principles, so as to maximize the cumulative wealth for 
the whole investment.

For any usage of this function, the following paper(s) should be cited as
reference:

[1]Zhao-Rong Lai, Pei-Yi Yang, Liangda Fang and Xiaotian Wu. "Short-term Sparse 
Portfolio Optimization based on Alternating Direction Method of Multipliers", 
Journal of Machine Learning Research, 2018. Accepted.

At the same time, it is encouraged to cite the following papers with previous related works:

[2] B. Li, D. Sahoo, and S. C. H. Hoi. OLPS: a toolbox for on-line portfolio selection. 
Journal of Machine Learning Research, 17(1):1242?1246, 2016.
[3] J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra, ¡°Efficient
projections onto the \ell_1-ball for learning in high dimensions,¡± in
Proceedings of the International Conference on Machine Learning (ICML), 2008.


Inputs:
data                      -data with price relative sequences
opts                      -parameter setting

Outputs:
cum_wealth                -cumulative wealths
daily_incre_fact          -daily increasing factors of SSPO
daily_port_total          -daily selected portfolios of SSPO
prim_res_total            -primal residuals for all the trades
iter_total                -number of iterations for all the trades

Example:
opts=[];
[ cum_wealth, daily_incre_fact, daily_port_total,prim_res_total,iter_total]
= SSPO_run(data,opts);
%}

%% Parameter Setting
if isfield(opts,'tran_cost')
    tran_cost = opts.tran_cost;         % -transaction cost rate
else
    tran_cost = 0;
end
if isfield(opts,'win_size')
    win_size = opts.win_size;           % -window size
else
    win_size = 5;
end

%% Variables Initialization

[T,N]=size(data);
run_ret =1;
cum_wealth = ones(T, 1);            %cumulative wealth S_t
daily_incre_fact = ones(T, 1);      %daily increasing factor
daily_port = ones(N, 1)/N;          %daily portfolio, starting with uniform portfolio
daily_port_total=ones(N, T)/N;      %portfolio matrix
daily_port_o = zeros(N, 1);         %old daily portfolio
data_close = ones(T,N);             %close price
prim_res_total={};
iter_total=[];

%% to get the close price according to relative price
for i=2:T
    data_close(i,:)= data_close(i-1,:).*data(i,:);
end

%% Main
for t = 1:1:T
    %Calculate t's daily increasing factor and cumulative wealth
    daily_port_total(:,t)=daily_port;
    daily_incre_fact(t, 1) = (data(t, :)*daily_port)*(1-tran_cost/2*sum(abs(daily_port-daily_port_o)));
    run_ret = run_ret * daily_incre_fact(t, 1);
    cum_wealth(t, 1) = run_ret;
    
    % Adjust portfolio for the transaction cost issue
    daily_port_o = daily_port.*data(t, :)'/(data(t, :)*daily_port);

    %Update portfolio
     if(t<T)
       [daily_port,prim_res,iter]=SSPO_fun(data_close,data,t+1,daily_port, win_size,opts);
       prim_res_total=[prim_res_total;prim_res];
       iter_total=[iter_total;iter];
     end
end

if cum_wealth(end)<10000
    fprintf('\t %.2f \n',cum_wealth(end));
else
    fprintf('\t %.2e \n',cum_wealth(end));
end
end