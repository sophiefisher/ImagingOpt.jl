function ChainRules.rrule(
    config::RuleConfig{>:HasReverseMode}, ::typeof(ThreadsX.sum), f, xs::AbstractArray)
    fx_and_pullbacks = ThreadsX.map(x->rrule_via_ad(config, f, x), xs)
    y = ThreadsX.sum(first, fx_and_pullbacks)

    pullbacks = ThreadsX.map(last, fx_and_pullbacks)

    project = ProjectTo(xs)

    function sum_pullback(ȳ)
        call(f, x) = f(x)
        # if dims is :, then need only left-handed only broadcast
        # broadcast_ȳ = dims isa Colon  ? (ȳ,) : ȳ
        broadcast_ȳ = ȳ
        f̄_and_x̄s = ThreadsX.map(f->f(ȳ), pullbacks)
        # no point thunking as most of work is in f̄_and_x̄s which we need to compute for both
        f̄ = if fieldcount(typeof(f)) === 0 # Then don't need to worry about derivative wrt f
            NoTangent()
        else
            ThreadsX.sum(first, f̄_and_x̄s)
        end
        x̄s = ThreadsX.map(unthunk ∘ last, f̄_and_x̄s) # project does not support receiving InplaceableThunks
        return NoTangent(), f̄, project(x̄s)
    end
    return y, sum_pullback
end

# Implementations of differentiable functions that form optimization pipeline
#=
function nearfield(incident, surrogate, geoms)
    return incident .* surrogate.(geoms)
end


function ChainRulesCore.rrule(::typeof(nearfield), incident, surrogate, geoms)
    out = map(incident, geoms) do i,g 
        s, sderiv = chebgradient(surrogate,g)
        s * i, sderiv * i
    end
    project_geom = ProjectTo(geoms)
    pullback(y) = NoTangent(), NoTangent(), NoTangent(), project_geom(last.(out) .* y )
    return first.(out), pullback
end
=#

function geoms_to_far(geoms::Matrix{Float64}, surrogate::FastChebInterp.ChebPoly{1, ComplexF64, Float64}, incident::Matrix{ComplexF64}, n2f_kernel::Matrix{ComplexF64}, plan_nearfar::FFTW.cFFTWPlan, parallel::Bool=true)
    #=gridL, _ = size(incident)
    
    to_trans = (geom, surrogate) -> surrogate(geom)
    geomstmp = dropdims(geoms, dims=1)

    surtmp = Zygote.ignore(() -> repeat([surrogate], inner=(gridL, gridL)) ) #TODO why Zygote.ignore?
    trans = map(to_trans, geomstmp, surtmp); #TODO pmap
    =#
    if parallel == true
        near = incident .* ThreadsX.map(surrogate,geoms)
    else
        near = incident .* map(surrogate,geoms)
    end
  
    
    #to_far = (near_field, kernel) -> convolve(near_field, kernel)
    far = convolve(near, n2f_kernel, plan_nearfar);
    far
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

function PSFs_to_G(PSFs, objL, imgL, nF, iF, lbfreq, ubfreq, plan_PSF)
    psfL, _ = size(PSFs)
    PSFsC = complex.(PSFs) # needed because adjoint of fft does not project correctly
    
    fftPSFs = plan_PSF * PSFsC
    
    G = Gop(fftPSFs, objL, imgL, nF, iF, lbfreq, ubfreq, plan_PSF)

    G
end

#=
function make_images(pp, imgp, Bs::Vector, freqs, surrogates, geoms)
    psfL = imgp.objL + imgp.imgL
    nF = pp.orderfreqPSF + 1
    ys = [zeros(imgp.imgL, imgp.imgL) for _ in 1:imgp.objN]
    ys_noiseless = [zeros(imgp.imgL, imgp.imgL) for _ in 1:imgp.objN]
    
    for iF in 1:nF
        freq = freqs[iF]
        surrogate = surrogates[iF]
        incident, n2f_kernel =  prepare_physics(pp, freq) 
        far, _ = geoms_to_far(geoms, surrogate, incident, n2f_kernel)
        PSF = far_to_PSFs(far, psfL, imgp.binL)
        G, _ = PSFs_to_G(PSF, imgp.objL, imgp.imgL, nF, iF, freqs[1], freqs[end])
        for iO in 1:imgp.objN
            y_temp = G * Bs[iO][:,:,iF][:]
            y_temp = reshape(y_temp, (imgp.imgL,imgp.imgL))
            ys_noiseless[iO] = ys_noiseless[iO] + y_temp
        end
    end
    #add nosie 
    for iO in 1:imgp.objN
        if imgp.noise_abs == true
            ys[iO] = abs.(ys_noiseless[iO] .+ mean(ys_noiseless[iO])* imgp.noise_level .*randn.() )
        else
            ys[iO] = ys_noiseless[iO] .+ mean(ys_noiseless[iO])* imgp.noise_level .*randn.()
        end
    end
    ys
end
=#

function make_images(pp, imgp, B::Array, freqs, surrogates, geoms, plan_nearfar, plan_PSF)
    psfL = imgp.objL + imgp.imgL
    nF = pp.orderfreq + 1
    y = zeros(imgp.imgL, imgp.imgL) 
    y_noiseless = zeros(imgp.imgL, imgp.imgL) 
    
    if imgp.emiss_noise_level != 0
        emiss_noise = rand(imgp.objL^2) * imgp.emiss_noise_level
    end
    
    for iF in 1:nF
        freq = freqs[iF]
        surrogate = surrogates[iF]
        incident =  prepare_incident(pp,freq) 
        n2f_kernel =  prepare_n2f_kernel(pp,freq, plan_nearfar) 
        far = geoms_to_far(geoms, surrogate, incident, n2f_kernel, plan_nearfar)
        PSF = far_to_PSFs(far, psfL, imgp.binL)
        G = PSFs_to_G(PSF, imgp.objL, imgp.imgL, nF, iF, freqs[1], freqs[end], plan_PSF)
        if imgp.emiss_noise_level != 0
            y_temp = G * (B[:,:,iF][:] .* (1 .- emiss_noise) )
        else
            y_temp = G * B[:,:,iF][:] 
        end
        y_temp = reshape(y_temp, (imgp.imgL,imgp.imgL))
        y_noiseless = y_noiseless + y_temp
    end
    #add nosie 
    
    if imgp.noise_abs == true
        y = abs.(y_noiseless .+ mean(y_noiseless)* imgp.noise_level .*randn.() )
    else
        y = y_noiseless .+ mean(y_noiseless)* imgp.noise_level .*randn.()
    end
    y
end


#reconstruction
function reconstruct_object(ygrid, Tmap, Tinit_flat, pp, imgp, optp, recp, freqs, surrogates, geoms, plan_nearfar, plan_PSF, α, save::Bool=true, parallel::Bool=true)
    nF = pp.orderfreq + 1
    psfL = imgp.objL + imgp.imgL
    noise_level = imgp.noise_level
    xtol_rel = recp.tol
    yflat = ygrid[:]
    
    objective(Tflat::Vector) = reconstruction_objective(Tflat, yflat, pp, imgp, freqs, surrogates, geoms, plan_nearfar, plan_PSF, α, parallel)

    
    function myfunc(Tflat::Vector, grad::Vector)
        if length(grad) > 0
            grad[:] = gradient( objective, Tflat )[1]
        end
        obj = objective(Tflat)
        println(obj)
        flush(stdout)
        obj
    end
    
    opt = Opt(:LD_LBFGS, imgp.objL^2)
    opt.lower_bounds = repeat([3.0,], imgp.objL^2)
    opt.xtol_rel = xtol_rel
    opt.min_objective = myfunc
    
    (minf,minT,ret) = optimize(opt, Tinit_flat)
    
    if save==true
        Tmaps_filename = @sprintf("ImagingOpt.jl/recdata/Tmap_%s_%s_%d_%d_%.2e_%.2f_%s_%.2e_%.2f.csv", imgp.object_data[4], optp.geoms_init_data[2],  imgp.objL, imgp.imgL, α, noise_level, string(imgp.noise_abs), xtol_rel, imgp.emiss_noise_level);
        writedlm( Tmaps_filename,  hcat(minT, Tmap[:]),',')
    end
    (minf,minT,ret)
end


function reconstruction_objective(Tflat::Vector, yflat, pp, imgp, freqs, surrogates, geoms, plan_nearfar, plan_PSF, α, parallel::Bool=true, )
    nF = pp.orderfreq + 1
    psfL = imgp.objL + imgp.imgL
    Tgrid = reshape(Tflat, imgp.objL, imgp.objL)
    Bgrid = prepare_blackbody(Tgrid, freqs, imgp, pp)
    

    function forward(iF)
        freq = ChainRulesCore.ignore_derivatives( freqs[iF])
        surrogate = ChainRulesCore.ignore_derivatives(()-> surrogates[iF])
        incident = ChainRulesCore.ignore_derivatives( ()-> prepare_incident(pp,freq) )
        n2f_kernel = ChainRulesCore.ignore_derivatives( ()-> prepare_n2f_kernel(pp,freq, plan_nearfar) )
        far = ChainRulesCore.ignore_derivatives( ()-> geoms_to_far(geoms, surrogate, incident, n2f_kernel, plan_nearfar, parallel) ) 
        PSF = ChainRulesCore.ignore_derivatives( ()-> far_to_PSFs(far, psfL, imgp.binL) )
        G = ChainRulesCore.ignore_derivatives( ()-> PSFs_to_G(PSF, imgp.objL, imgp.imgL, nF, iF, freqs[1], freqs[end], plan_PSF) )
        G * Bgrid[:,:,iF][:]
    end

    if parallel == true
        yflat_noiseless  = ThreadsX.sum(iF->forward(iF), 1:nF)
    else
        yflat_noiseless  = sum(iF->forward(iF), 1:nF)
    end

    (yflat - yflat_noiseless)'*(yflat - yflat_noiseless) + α*Tflat'*Tflat
end
