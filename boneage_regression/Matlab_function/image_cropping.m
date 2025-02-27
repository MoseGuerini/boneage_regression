function img_cropped = image_cropping(img)
    % Switch to greyscale if the image is not in this mode
    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    img = double(img); % Convert to double for Python comaptibility

    % Normalize the image
    max_intensity = prctile(img(:), 99.5);
    min_intensity = prctile(img(:), 0.5);
    img_norm = (img - min_intensity) / (max_intensity - min_intensity);
    img_norm = max(0, min(1, img_norm));

    % Create the mask used to discard as much background as possible
    lower_threshold = 0.1;
    upper_threshold = 1;
    mask = (img_norm > lower_threshold) & (img_norm < upper_threshold);

    % Search hand bounding box
    [row, col] = find(mask);

    x_min = min(col);
    x_max = max(col);
    y_min = min(row);
    y_max = max(row);
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min];

    % Crop the image
    img_cropped = imcrop(img_norm, bbox);

end