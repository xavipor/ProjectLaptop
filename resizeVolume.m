function V_ = resizeVolume(V, volumeSize)


[r, c, s] = size(V);

% crop row
if r > volumeSize(1)
    dr = r-volumeSize(1);
    if mod(dr,2)==1
        hdr = ceil(dr/2);
    else
        hdr = dr/2;
    end
    V1 = V(hdr+1:end-(dr-hdr), :, :);
end

% pad row
if r < volumeSize(1)
    V1 = zeros(volumeSize(1), size(V,2), size(V,3));
    V1(1:size(V,1), :, :) = V;
end

% keep row
if r == volumeSize(1)
    V1 = V;
end

clearvars V

% crop colume
if c > volumeSize(2)
    dc = c - volumeSize(2);
    if mod(dc,2)==1
        hdc = ceil(dc/2);
    else
        hdc = dc/2;
    end
    V2 = V1(:, hdc+1:end-(dc-hdc), :);
end

% pad colume
if c < volumeSize(2)
    V2 = zeros(size(V1,1), volumeSize(2), size(V1,3));
    V2(:, 1:size(V1,2), :) = V1;
end

% keep colume
if c == volumeSize(2)
    V2 = V1;
end

clearvars V1

% crop slice
if s > volumeSize(3)
    ds = s - volumeSize(3);
    if mod(ds,2)==1
        hds = ceil(ds/2);
    else
        hds = ds/2;
    end
    V_ = V2(:, :, hds+1:end-(ds-hds));
end

% pad slice
if s < volumeSize(3)
    V_ = zeros(size(V2,1), size(V2,2), volumeSize(3));
    V_(:, :, 1:size(V2,3)) = V2;
end

% keep colume
if s == volumeSize(3)
    V_ = V2;
end

clearvars V2

end





