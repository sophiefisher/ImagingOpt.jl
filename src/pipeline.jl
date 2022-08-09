# Implementations of differentiable functions that form optimization pipeline

function geoms_to_far(geoms, surrogate, incident, n2f_kernel, freq)
    gridL, _ = size(incident)
    
    to_trans = (geom, freq, surrogate) -> surrogate(geom, freq)
    geomstmp = dropdims(geoms, dims=1)

    freqstmp = repeat([freq,], gridL, gridL)

    surtmp = Zygote.ignore(() -> repeat([surrogate], inner=(gridL, gridL)) ) #TODO why Zygote.ignore?
    trans = map(to_trans, geomstmp, freqstmp, surtmp); #TODO pmap
    
    near = incident .* trans #broadcasting
    
    to_far = (near_field, kernel) -> convolve(near_field, kernel)
    
    far = to_far(near, n2f_kernel);
    (;far, near, trans)
end

function far_to_PSFs(far, psfL, binL)
    gridL, _ = size(far)
    cropL = (gridL - psfL * binL) ÷ 2 # require binL * (objL + imgL) <= gridL
    farcrop = far[cropL + 1:gridL - cropL, cropL + 1:gridL - cropL];
    farcropbin = reshape(farcrop, (binL, psfL, binL, psfL))
    farcropbinmag = (abs.(farcropbin)).^2
    PSFsbin = sum(farcropbinmag, dims=(1, 3))
    PSFs = dropdims(PSFsbin, dims=(1,3))
    PSFs = PSFs ./ mean(PSFs) # Normalize PSF values, allowing for different calibration values for different channels
    PSFs
end

function PSFs_to_G(PSFs, objL, imgL, nF, iF, lbfreq, ubfreq)
    psfL, _ = size(PSFs)
    PSFsC = complex.(PSFs) # needed because adjoint of fft does not project correctly
    
    fftPSFs = planned_fft(PSFsC) 
    
    G = Gop(fftPSFs, objL, imgL, nF, iF, lbfreq, ubfreq)

    (;G, fftPSFs)
end
