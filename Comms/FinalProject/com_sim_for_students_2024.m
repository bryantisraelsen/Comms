% Digital communications simulation
% Original Author:  Jake Gunther
% Date  :  January 2024
% Class :  ECE 5660 (Utah State University)

clear all; close all; fclose all;

% Set parameters
f_eps        = 0.0; % Carrier frequency offset percentage (0 = no offset)
Phase_Offset = 0.0; % Carrier phase offset (0 = no offset)
t_eps        = 0.0; % Clock freqency offset percentage (0 = no offset)
T_offset     = 0.0; % Clock phase offset (0 = no offset)
Ts = 1; % Symbol period
N = 4; % Samples per symbol period
fname = 'kitten1_8bit.dat';

% Select modulation type
% Use 8 bits per symbol and 256 square QAM
B = 8; % Bits per symbol (B should be even: 8, 6, 4, 2)
bits2index = 2.^[B-1:-1:0].';
M = 2^B; % Number of symbols in the constellation
Mroot = 2^(B/2);
a = [-Mroot+1:2:Mroot-1].'; b = ones(Mroot,1);
LUT = [kron([-Mroot+1:2:Mroot-1].',ones(Mroot,1)), ...
       kron(ones(Mroot,1),[-Mroot+1:2:Mroot-1].')];
% Scale the constellation to have unit energy
Eave = sum(sum(LUT.^2))/M;
LUT = LUT/sqrt(Eave);
Eave = 1;
Eb = Eave/B;
EbN0dB = 30; % SNR in dB
N0 = Eb*10^(-EbN0dB/10);
nstd = sqrt(N0/2); % Noise standard deviation
% Note: nstd is set to 0 below so there is no noise in this test

if 0
% Plot the constellation
figure;
plot(LUT(:,1),LUT(:,2),'o');
for i=1:M
  text(LUT(i,1),LUT(i,2),[' ',dec2bin(i-1,B)]);
end
grid on; axis((max(axis)+0.1/B)*[-1 1 -1 1]); axis square;
xlabel('In-Phase','FontSize',16,'interpreter','latex');
ylabel('Quadrature','FontSize',16,'interpreter','latex');
title(['Constellation Diagram for ',int2str(M),'-QAM'],...
      'FontSize',16,'interpreter','latex');
set(gca,'FontSize',16);
end

% Unique word (randomly generated)
uw = [162
    29
    92
    47
    16
   112
    63
   234
    50
     7
    15
   211
   109
   124
   239
   255
   243
   134
   119
    40
   134
   158
   182
     0
   101
    62
   176
   152
   228
    36];
uw = uw(:);
uw_len = length(uw);
uwsym = LUT(uw+1,:); % Add 1 for Matlab's base 1 indexing


% Build the list of four possible UW rotations
angles = 2*pi*[0:3]/4;
uwrotsyms = zeros(length(uw),2,4);
for i=1:length(angles)
  C = cos(angles(i)); S = -sin(angles(i));
  G = [C -S; S C];
  uwrot = uwsym*G; % Rotate the UW symbols
  uwrotsyms(:,:,i) = uwrot; % Save the rotated version
end

% Load and display the image
disp(fname);
fid=fopen(fname,'rb');
imsize = fread(fid,2,'int');
rows = imsize(1);
cols = imsize(2);
[x,xlen] = fread(fid,rows*cols,'uint8');
fclose(fid);
x = reshape(x,rows,cols);
fprintf('Rows = %d, Columns = %d, Pixels = %d, Input Length = %d\n',rows,cols,rows*cols,xlen);
figure;
imagesc(x); axis image;
colormap gray; title('Original','FontSize',24); drawnow;

% Periodically insert the unique word
x = [x;repmat(uw,1,cols)];
x = x + 1; % Add 1 for Matlab's 1-based indexing (x is used to
           % index symbols directly
figure
imagesc(x); axis image;
colormap gray; title('With UW','FontSize',24); drawnow;

rows = rows + uw_len;
x = x(:); % Column scan
sym_stream = LUT(x,:);
sym_keep = sym_stream;
num_syms = length(sym_stream);

% Generate received signal with a clock frequency offset
tic;
fprintf('Generating transmitted I/Q waveforms ... ');
EB = 0.7; % Excess bandwidth
To = (1+t_eps);
if(t_eps == 0) % No clock skew
  Lp = 12;
  t = [-Lp*N:Lp*N].'/N+1e-8;
  tt = t + T_offset;
  srrc = (sin(pi*(1-EB)*tt)+4*EB*tt.*cos(pi*(1+EB)*tt))./((pi*tt).*(1-(4*EB*tt).^2));
  srrc = srrc/sqrt(N);
  Isu = zeros(num_syms*N,1); Isu(1:N:end) = sym_stream(:,1);
  Qsu = zeros(num_syms*N,1); Qsu(1:N:end) = sym_stream(:,2);
  I = conv(srrc,Isu);
  Q = conv(srrc,Qsu);
else % Implement clock skew
  t = [0:num_syms*N-1].'/N+1e-8;
  I = zeros(size(t)); % In-phase pulse train
  Q = zeros(size(t)); % Quadrature pulse train
  for i=1:num_syms
    tt = t-i*To + T_offset;
    srrc = (sin(pi*(1-EB)*tt)+4*EB*tt.*cos(pi*(1+EB)*tt))./((pi*tt).*(1-(4*EB*tt).^2));
    srrc = srrc/sqrt(N);
    I = I + srrc*sym_stream(i,1);
    Q = Q + srrc*sym_stream(i,2);
  end
end
fprintf('done.\n');
toc;

% Modulate the pulse trains
tic;
fprintf('Modulating I/Q waveforms ... ');
Omega0 = pi/2*(1+f_eps);
n = [0:length(I)-1].';
C =  sqrt(2)*cos(Omega0*n + Phase_Offset);
S = -sqrt(2)*sin(Omega0*n + Phase_Offset);
nstd = 0   % Override the SNR, so now there is no noise for testing
r = I.*C + Q.*S + nstd*randn(size(I)); % Noisy received signal
fprintf('done.\n');
toc;

save test_2021 -ascii -double r
