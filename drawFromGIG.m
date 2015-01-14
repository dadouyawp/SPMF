function out = drawFromGIG(lam, psi, chi)
% THE GENERALIZED INVERSE GAUSSIAN DISTRIBUTION
%     GIG(lam, chi, psi)
%
% pdf = (psi/chi)^(lam/2)*y.^(lam-1)/(2*besselk(lam, sqrt(chi*psi))) .* exp(-1/2*(chi./y + psi*y));  y > 0 
%
% Mean = sqrt( chi / psi ) * besselk(lam+1,sqrt(chi*psi),1)/besselk(lam,sqrt(chi*psi),1);
% Variance = chi/psi * besselk(lam+2,sqrt(chi*psi),1)/besselk(lam,sqrt(chi*psi),1) - Mean^2;
%
% PARAMETERS:
%   chi>0,  psi>=0  if lam<0;
%   chi>0,  psi>0   if lam=0;
%   chi>=0, psi>0   if lam>0;
%
% USAGE:
%   drawFromGIG(lam, chi, psi) - generate same size number variates of chi 
%         and psi from the Generalized Inverse Gaussian distribution with 
%         parameters 'lam', 'chi' and 'psi'
%
% EXAMPLES:
%  y = drawFromGIG(-1, [1, 2]', [3, 4]');
%
% AUTHER: WangPeng (wangpeng@bjtu.edu.cn)
% DATE:   Jan, 12, 2015
% Beijing Jiaotong University

% First we check parameters

h = lam;
b = sqrt( chi .* psi );
out = zeros(size(psi));

m = ( h-1+sqrt((h-1)^2 + b .^ 2) ) ./ b;  % Mode
log_1_over_pm = -(h-1)/2 .* log(m) + (b./4) .* (m + (1./m));

r = (6 .* m + (2*h) .* m - b .* m .^ 2 + b) ./ (4 .* m .^ 2);
s = (1 + h - b .* m) ./ (2 .* m .^ 2);
p = (3 .* s - r .^ 2) ./ 3;
q = (2 .* r .^ 3) ./ 27 - (r .* s) ./ 27 + b ./ (-4 .* m .^ 2);
eta = sqrt(-(p .^ 3) ./ 27);

y1  = 2 .* exp(log(eta) ./ 3) .* cos(acos(-q ./ (2 .* eta)) ./ 3) - r ./ 3;
y2  = 2 .* exp(log(eta) ./ 3) .* cos(acos(-q ./ (2 .* eta)) ./ 3  + 2 ./ 3 .* pi) - r ./ 3;

temp = (h<=1 & b<=1) | abs(q ./ eta)>2 | y1<0 | y2>0;
list = find(temp);

% without shifting by m                        
ym = (-h-1 + sqrt((h+1)^2 + b(list) .^ 2)) ./ b(list);

% a = vplus/uplus
a = exp(-0.5*h .* log(m(list) .* ym) + 0.5*log(m(list) ./ ym) + b(list) ./ 4 .* (m(list) + 1 ./ m(list) - ym - 1.0 ./ ym));

u = rand( size(ym) );
v = rand( size(ym) );
out(list) = a .* (v./u);
indxs = find( log(u) > (h-1)/2 .* log(out(list)) - b(list) ./ 4 .* (out(list) + 1 ./ out(list)) + log_1_over_pm(list) );
while ~isempty( indxs )
    indxsSize = size( indxs );
    u = rand( indxsSize );
    v = rand( indxsSize );
    outNew = a(indxs) .* (v./u);
    l = log(u) <= (h-1)/2 .* log(outNew) - b(list(indxs)) ./ 4 .* (outNew + 1 ./ outNew ) + log_1_over_pm(list(indxs));
    out( indxs( l ) ) = outNew(l);
    indxs = indxs( ~l );
end                      

% with shifting by m                         
list = find(~temp);
vplus = exp( log_1_over_pm(list) + log(1 ./ y1(list)) + (h-1)/2 .* log(1 ./ y1(list) + m(list)) - ...
    b(list) ./ 4 .* (1 ./ y1(list) + m(list) + 1 ./ (1 ./ y1(list) + m(list))) );
vminus = -exp( log_1_over_pm(list) + log(-1 ./ y2(list)) + (h-1)/2 .* log(1 ./ y2(list) + m(list)) - ...
    b(list) ./ 4 .* (1 ./ y2(list) + m(list) + 1 ./ (1 ./ y2(list) + m(list))) );  

u = rand( size(vplus) );
v = vminus + (vplus - vminus) .* rand( size(vplus) );
z = v ./ u;
clear('v');
indxs = find( z < -m(list) );

while ~isempty(indxs),
    indxsSize = size( indxs );
    uNew = rand( indxsSize );
    vNew = vminus(indxs) + (vplus(indxs) - vminus(indxs)) .* rand( indxsSize );
    zNew = vNew ./ uNew;
    l = (zNew >= -m(list(indxs)));
    z( indxs( l ) ) = zNew(l);
    u( indxs( l ) ) = uNew(l);
    indxs = indxs( ~l );
end

out(list) = z + m(list);
indxs = find( log(u) > (log_1_over_pm(list) + (h-1)/2 .* log(out(list)) - b(list) ./ 4 .* (out(list) + 1./out(list))) );

iter = 0;
total = length(chi) * 20;
while ~isempty(indxs) & iter <= total
    iter = iter + 1;
    if iter > total
        fprintf('Warning: too many circulation \n');
    end
    indxsSize = size( indxs );                             
    u = rand( indxsSize );
    v = vminus(indxs) + (vplus(indxs) - vminus(indxs)) .* rand( indxsSize );
    z = v ./ u;
    clear('v');
    indxs1 = find( z < -m(list(indxs)) );
    while ~isempty(indxs1),
        indxsSize1 = size( indxs1 );
        uNew = rand( indxsSize1 );
        vNew = vminus(indxs(indxs1)) + (vplus(indxs(indxs1)) - vminus(indxs(indxs1))) .* rand( indxsSize1 );
        zNew = vNew ./ uNew;
        l = (zNew >= -m(list(indxs(indxs1))));
        z( indxs1( l ) ) = zNew(l);
        u( indxs1( l ) ) = uNew(l);
        indxs1 = indxs1( ~l );
    end

    outNew = z + m(list(indxs));
    l = ( log(u) <= (log_1_over_pm(indxs) + (h-1)/2 .* log(outNew) - b(indxs) ./ 4 .* (outNew + 1 ./ outNew)) );
    out( indxs(l) ) = outNew( l );
    indxs = indxs( ~l );
end

out = sqrt( chi ./ psi ) .* out;
end