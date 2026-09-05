function Iout = preprocessFundus(input, targetSize)
% PREPROCESSFUNDUS FOV-crop, square-pad, and resize an RGB fundus image.

if ischar(input) || isstring(input)
    I = imread(input);
else
    I = input;
end
if size(I, 3) == 1
    I = repmat(I, 1, 1, 3);
end
[I, ~] = cropFundusFOV(I);
I = padToSquare(I);
Iout = im2single(imresize(I, [targetSize targetSize]));
end
