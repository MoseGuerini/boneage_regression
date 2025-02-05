function img_resized = image_resizing(img, target_size_side)
    % Definisci la nuova dimensione desiderata
    target_size = [target_size_side, target_size_side];

    % Ridimensiona l'immagine
    img_resized = imresize(img, target_size, 'bilinear'); % Usa 'bilinear' o 'bicubic' per interpolazione migliore
end
