# Implementations of differentiable functions that form optimization pipeline and their derivatives
function nearfield(incident, surrogate, geoms, parallel)
    if parallel
        #near = incident .* ThreadsX.map(surrogate,geoms, basesize = div(length(geoms), 2) )
        near = incident .* ThreadsX.map(surrogate,geoms)
    else
        near = incident .* surrogate.(geoms)
    end
    near
end

function geoms_to_far(geoms, surrogate, incident, n2f_kernel, plan_nearfar::FFTW.cFFTWPlan, parallel::Bool=true)
    near = nearfield(incident, surrogate, geoms, parallel)
    far = convolve(near, n2f_kernel, plan_nearfar)
end

function far_to_PSF(far, psfL, binL, scaling, freq)
    #gridL, _ = size(far)
    #cropL = (gridL - psfL * binL) ÷ 2 # require binL * (objL + imgL) <= gridL
    #farcrop = far[cropL + 1:gridL - cropL, cropL + 1:gridL - cropL];
    farcropbin = reshape(far, (binL, psfL, binL, psfL))
    farcropbinmag = (abs.(farcropbin)).^2
    PSFbin = sum(farcropbinmag, dims=(1, 3))
    #multiply PSFs by arbitrary scaling factor and divide by freqency to get photon count
    PSF = scaling .* dropdims(PSFbin, dims=(1,3)) ./ freq 
end
    

function PSF_to_fftPSF(PSF, plan_PSF)
    fftPSF = plan_PSF * complex.(PSF) # needed because adjoint of fft does not project correctly
end

#helper functions 
function get_far(freq, surrogate, pp, imgp, geoms, plan_nearfar, parallel)
    incident =  ChainRulesCore.ignore_derivatives( ()-> prepare_incident(pp,freq) ) 
    n2f_kernel =  ChainRulesCore.ignore_derivatives( ()-> prepare_n2f_kernel(pp,imgp,freq, plan_nearfar) )
    far = geoms_to_far(geoms, surrogate, incident, n2f_kernel, plan_nearfar, parallel)
end

function get_PSF(freq, surrogate, pp, imgp, geoms, plan_nearfar, parallel)
    far = get_far(freq, surrogate, pp, imgp, geoms, plan_nearfar, parallel)
    PSF = far_to_PSF(far, imgp.objL + imgp.imgL, imgp.binL, pp.PSF_scaling, freq)
end

function get_fftPSF(freq, surrogate, pp, imgp, geoms, plan_nearfar, plan_PSF, parallel)
    PSF = get_PSF(freq, surrogate, pp, imgp, geoms, plan_nearfar, parallel)
    fftPSF = PSF_to_fftPSF(PSF, plan_PSF)
end

function get_fftPSF_from_far(far, freq, pp, imgp, plan_nearfar, plan_PSF)
    PSF = far_to_PSF(far, imgp.objL + imgp.imgL, imgp.binL, pp.PSF_scaling, freq)
    fftPSF = PSF_to_fftPSF(PSF, plan_PSF)
end
#end helper functions

function make_image(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, noise, plan_nearfar, plan_PSF, parallel)
    nF = pp.orderfreq + 1
    
    floattype = typeof(freqs[1])
        
    if imgp.emiss_noise_level != 0
        emiss_noise = rand(floattype, imgp.objL, imgp.objL) * imgp.emiss_noise_level
    end
    
    function image_Tmap_grid_noiseless_iF(iF)
        if imgp.emiss_noise_level != 0
            y_temp = G(B_Tmap_grid[:,:,iF] .* (1 .- emiss_noise), fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF)
        else
            y_temp = G(B_Tmap_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF)
        end
        y_temp = reshape(y_temp, (imgp.imgL,imgp.imgL))
    end
    
    if parallel
        image_Tmap_grid_noiseless = ThreadsX.sum(iF->image_Tmap_grid_noiseless_iF(iF), 1:nF)
    else
        image_Tmap_grid_noiseless = sum(iF->image_Tmap_grid_noiseless_iF(iF), 1:nF)
    end
    
    #add nosie 
    if imgp.noise_abs
        image_Tmap_grid = abs.(image_Tmap_grid_noiseless .+ mean(image_Tmap_grid_noiseless)* imgp.noise_level .*randn.(floattype) )
    else
        image_Tmap_grid = image_Tmap_grid_noiseless .+ mean(image_Tmap_grid_noiseless)*noise
    end
    image_Tmap_grid
end


function get_image_diff_flat(Test_flat, image_Tmap_flat, pp, imgp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel::Bool=true)
    nF = pp.orderfreq + 1
    Test_grid = reshape(Test_flat, imgp.objL, imgp.objL)
    B_Test_grid = prepare_blackbody(Test_grid, freqs, imgp, pp)
    
    if parallel
        image_Test_flat  = ThreadsX.sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    else
        image_Test_flat  = sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    end
    image_Tmap_flat - image_Test_flat
end


function reconstruction_objective_simplified(Test_flat, α, image_diff_flat, subtract_reg, print_objvals::Bool=false)
    term1 = image_diff_flat'*image_diff_flat
    term2 = α*( (Test_flat .- subtract_reg)'*(Test_flat .- subtract_reg) )
        
    #println("obj:$( round(term1+term2,sigdigits=8) ) \t \t term1:$(round(term1,sigdigits=8) ) \t \t term2:$(round(term2,sigdigits=8) ) ")
    if print_objvals
        @printf("obj: %30.8f term1: %30.8f term2: %30.8f \n" ,term1+term2, term1, term2)
    end
    term1, term2
end

function gradient_reconstruction_T(Test_flat, image_diff_flat, α, pp, imgp, recp, fftPSFs, freqs, plan_nearfar, plan_PSF, weights, parallel)
    nF = pp.orderfreq + 1
    image_diff_grid = reshape(image_diff_flat, imgp.imgL, imgp.imgL)
    
    function term2(iF)
        temp_term2 = Gtranspose( image_diff_grid,  fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF)
        dB_dT.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) .* temp_term2
    end
    
    if parallel
        term2  = ThreadsX.sum(iF->term2(iF), 1:nF)
    else
        term2  = sum(iF->term2(iF), 1:nF)
    end

    2 .* α .* (Test_flat .- recp.subtract_reg ) .+ (-2 .* term2)
    
end
        
function gradient_reconstruction_T_autodiff(Test_flat::Vector, image_Tmap_flat, pp, imgp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, parallel)
    function obj(Test_flat)
        image_diff_flat = get_image_diff_flat(Test_flat, image_Tmap_flat, pp, imgp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
        term1, term2 = reconstruction_objective_simplified(Test_flat, α, image_diff_flat, recp.subtract_reg)
        term1 + term2
    end
    
    Zygote.gradient(obj, Test_flat)[1]
end

#reconstruction
function reconstruct_object(image_Tmap_grid, Tmap, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, save_Tmaps::Bool=false, save_objvals::Bool=false, parallel::Bool=true, print_objvals::Bool=false)
    rec_id = Dates.now()
        
    nF = pp.orderfreq + 1
    image_Tmap_flat = image_Tmap_grid[:]

    if save_objvals
        objvals_filename = "ImagingOpt.jl/recdata/objvals_$(rec_id).csv"
    end
        
    image_diff_flat = similar(image_Tmap_flat)
    function myfunc(Test_flat::Vector, grad::Vector)
        image_diff_flat[:] = get_image_diff_flat(Test_flat, image_Tmap_flat, pp, imgp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
        if length(grad) > 0
            #grad[:] = gradient( objective, Test_flat )[1]
            grad[:] = gradient_reconstruction_T(Test_flat, image_diff_flat, α, pp, imgp, recp, fftPSFs, freqs, plan_nearfar, plan_PSF, weights, parallel)
        end
        #obj = objective(Test_flat)
        term1, term2 = reconstruction_objective_simplified(Test_flat, α, image_diff_flat,recp.subtract_reg, print_objvals)

        if save_objvals
            open(objvals_filename, "a") do io
                writedlm(io, [term1+term2 term1 term2], ',')
            end
        end
            
        term1+term2
    end
    
    opt = Opt(:LD_LBFGS, imgp.objL^2)
    opt.lower_bounds = repeat([ convert(typeof(pp.lbfreq),3.0),], imgp.objL^2)
    #opt.xtol_rel = recp.xtol_rel
    opt.ftol_rel = recp.ftol_rel
    opt.min_objective = myfunc

    (minf,minT,ret) = NLopt.optimize!(opt, Tinit_flat)
    minT = convert.(typeof(freqs[1]),minT)
    
    if save_Tmaps
        Tmaps_filename = "ImagingOpt.jl/recdata/Tmaps_$(rec_id).csv"
        writedlm( Tmaps_filename,  hcat( minT, Tmap[:]),',')
    end
    #println(@sprintf("minf is %.2f", minf))
    println(@sprintf("reconstruction return value is %s", ret))
    flush(stdout) 
    minT
end
