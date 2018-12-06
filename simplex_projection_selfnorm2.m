function w = simplex_projection_selfnorm2(v, b)
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


This function is the simplex projection function exploited by the Short-term Sparse Portfolio Optimization 
based on Alternating Direction Method of Multipliers [1]. It originates from [2][3]. 


For any usage of this function, the following paper(s) should be cited as
reference:

[1]Zhao-Rong Lai, Pei-Yi Yang, Liangda Fang and Xiaotian Wu. "Short-term Sparse 
Portfolio Optimization based on Alternating Direction Method of Multipliers", 
Journal of Machine Learning Research, 2018. Accepted.

At the same time, it is encouraged to cite the following paper(s) that
propose this method and the original code:

[2] J. Duchi, S. Shalev-Shwartz, Y. Singer, and T. Chandra, ¡°Efficient
projections onto the \ell_1-ball for learning in high dimensions,¡± in
Proceedings of the International Conference on Machine Learning (ICML), 2008.
[3] B. Li, D. Sahoo, and S. C. H. Hoi. OLPS: a toolbox for on-line portfolio selection. 
Journal of Machine Learning Research, 17(1):1242¨C1246, 2016.


Inputs:
v                  -a d-dimensional vector
b                  -the "size" of the simplex, default=1

Outputs:
w                  -the output vector on the simplex

%}

while(max(abs(v))>1e6)
v=v/10;
end

u = sort(v,'descend');

sv = cumsum(u);
rho = find(u > (sv - b) ./ (1:length(u))', 1, 'last');
theta = (sv(rho) - b) / rho;
w = max(v - theta, 0);
end