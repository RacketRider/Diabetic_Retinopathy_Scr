function Iout = padToSquare(I)
% PADTOSQUARE Zero-pad the shorter spatial dimension without distortion.

[height, width, channels] = size(I);
side = max(height, width);
Iout = zeros(side, side, channels, "like", I);
y0 = floor((side - height) / 2) + 1;
x0 = floor((side - width) / 2) + 1;
Iout(y0:y0 + height - 1, x0:x0 + width - 1, :) = I;
end
