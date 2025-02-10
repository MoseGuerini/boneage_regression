function img_cropped = image_cropping(img)
    % Converti in grayscale se necessario
    if size(img, 3) == 3
        img = rgb2gray(img);
    end

    img = double(img); % Converti in double per compatibilità con Python

    % Normalizza l'immagine
    max_intensity = prctile(img(:), 99.5);
    min_intensity = prctile(img(:), 0.5);
    img_norm = (img - min_intensity) / (max_intensity - min_intensity);
    img_norm = max(0, min(1, img_norm));

    % Crea la maschera per trovare la mano
    lower_threshold = 0.1;
    upper_threshold = 1;
    mask = (img_norm > lower_threshold) & (img_norm < upper_threshold);

    % Trova il bounding box della mano
    [row, col] = find(mask);

    x_min = min(col);
    x_max = max(col);
    y_min = min(row);
    y_max = max(row);
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min];

    % Ritaglia l'immagine
    img_cropped = imcrop(img_norm, bbox);

end