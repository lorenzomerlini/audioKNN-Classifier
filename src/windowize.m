function [M,nf]=windowize(x,wl,hop,W)
% [M,Nframes]=WINDOWIZE(Signal,<WindowSize,HopSize,WeigthWindow>)
%
% Windowizes a signal, (last frame is cut if not complete)
%

if nargin<2 wl=256; end                              % default 256
if nargin<3 hop=fix(wl./2); end                       % default wl/2


if size(x,2) ~= 1
    x=x';
end

xl=size(x,1);
nf=floor((xl-wl)/hop)+1;
lines=(1:wl)';
cols=0:hop:(nf-1)*hop;
M=x(lines(:,ones(1,nf))+cols(ones(wl,1),:));

if nargin==4
    if size(W,2) ~= 1
        W=W';
    end
    M=W(:,ones(1,nf)).*M;
end
