function img_resized = image_resizing(img, target_size_side)
    % Define the desired dimension
    target_size = [target_size_side, target_size_side];

    % Resize the image
    img_resized = imresize(img, target_size, 'bilinear');
    % 'bilinear' is needed for interpolation
end
