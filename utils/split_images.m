function split_images(type)
    if nargin == 0
        type = 'carcinoma';
    end
    
    patients = dir(type);
    
    counter = 0;
    file_mapping = containers.Map();
    bad = 0;
    for i = 1 : numel(patients)
        if ~isempty(patients(i).name) && patients(i).name(1) == '.'
            continue
        end

        one_patient = dir(fullfile(type, patients(i).name, '*.tif'));

        for j = 1 : numel(one_patient)
            if ~isempty(one_patient(j).name) && one_patient(j).name(1) == '.'
                continue
            end

            one_im = imread(fullfile(...
                type, patients(i).name, one_patient(j).name));
            
            if size(one_im, 3) == 4
                if all(one_im(:,:,4) == 255, 'all')
                    one_im(:,:,4) = [];
                else
                    warning(['input:', one_patient(j).name]);
                    continue;
                end
            end
            
            [tiled_im, tiled_r, tiled_c] = tile_crop_image(one_im, [500 500]);
            
            names = cell(size(tiled_im,4), 1);
            rows = num2cell(tiled_r');
            cols = num2cell(tiled_c');
            
            for k = 1 : size(tiled_im,4)
                if sum(rgb2gray(tiled_im(:,:,:,k)) > 230, 'all') > 500*500/2
                    bad = bad + 1;
                    continue
                end
                fname = sprintf('%08d.tif', counter);
                imwrite(tiled_im(:,:,:,k), fullfile([type(1), '2'], fname));
                names{k} = fname;
                %rows{k} = tiled_r(k);
                %cols{k} = tiled_c(k);
                counter = counter + 1;
            end
            
            file_mapping(...
                fullfile(type, patients(i).name, one_patient(j).name)) = ...
                struct('fname', names, 'row', rows, 'col', cols);
        end
    end
    bad
    
    fd = fopen(sprintf('%s_file_mapping.json', type), 'w');
    fprintf(fd,jsonencode(file_mapping));
    fclose(fd);

end

function [out_im, out_r, out_c] = tile_crop_image(im_in, tile_sz)
    [rs, cs, num_im] = tile_crop_dim(size(im_in, 1:2), tile_sz);
    out_im = zeros([tile_sz, size(im_in, 3), prod(num_im)], 'like', im_in);
    out_r = zeros([1, prod(num_im)]);
    out_c = zeros([1, prod(num_im)]);
    i = 1;
    for r = rs
        for c = cs
            out_im(:,:,:,i) = im_in(...
                r : r + tile_sz(1) - 1, c : c + tile_sz(2) - 1, :);
            out_r(i) = r;
            out_c(i) = c;
            i = i + 1;
        end
    end
end


function [row_start, col_start, num_im] = tile_crop_dim(im_sz, tile_sz)
    num_im = floor(im_sz ./ tile_sz);
    out_sz = num_im .* tile_sz;
    crop_on_one_side = floor((im_sz - out_sz) ./ 2);
    
    first_start = ones(size(im_sz)) + crop_on_one_side;
    ult_start = first_start + (num_im - 1) .* tile_sz;
    
    row_start = first_start(1):tile_sz(1):ult_start(1);
    col_start = first_start(2):tile_sz(2):ult_start(2);
end
