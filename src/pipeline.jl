#chain rule for threaded sum
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

#chain rule for threaded map
function ChainRulesCore.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(ThreadsX.map), f, X::AbstractArray)
    hobbits = ThreadsX.map(X) do x  # this makes an array of tuples
        y, back = rrule_via_ad(config, f, x)
    end
    Y = map(first, hobbits)
    function map_pullback(dY_raw)
        dY = unthunk(dY_raw)
        # Should really do these in the reverse order
        backevals = ThreadsX.map(hobbits, dY) do (y, back), dy
            dx, dx = back(dy)
        end
        df = ProjectTo(f)(ThreadsX.sum(first, backevals))
        dX = ThreadsX.map(last, backevals)
        return (NoTangent(), df, dX)
    end
    return Y, map_pullback
end


# Implementations of differentiable functions that form optimization pipeline and their derivatives
function nearfield(incident, surrogate, geoms, parallel)
    if parallel == true
        #near = incident .* ThreadsX.map(surrogate,geoms, basesize = div(length(geoms), 2) )
        near = incident .* ThreadsX.map(surrogate,geoms)
    else
        near = incident .* surrogate.(geoms)
    end
    near
end


function ChainRulesCore.rrule(::typeof(nearfield), incident, surrogate, geoms, parallel)
    if parallel==true
        out = ThreadsX.map(incident, geoms) do i,g 
            s, sderiv = chebgradient(surrogate,g)
            s * i, sderiv * i
        end
    else
        out = map(incident, geoms) do i,g 
            s, sderiv = chebgradient(surrogate,g)
            s * i, sderiv * i
        end
    end
    project_geom = ProjectTo(geoms)
    pullback(y) = NoTangent(), NoTangent(), NoTangent(), project_geom( conj(last.(out)) .* y ), NoTangent()
    return first.(out), pullback
end



function geoms_to_far(geoms, surrogate, incident, n2f_kernel, plan_nearfar::FFTW.cFFTWPlan, parallel::Bool=true)
    near = nearfield(incident, surrogate, geoms, parallel)
    far = convolve(near, n2f_kernel, plan_nearfar)
end

function far_to_PSF(far, psfL, binL, scaling, freq)
    gridL, _ = size(far)
    cropL = (gridL - psfL * binL) ÷ 2 # require binL * (objL + imgL) <= gridL
    farcrop = far[cropL + 1:gridL - cropL, cropL + 1:gridL - cropL];
    farcropbin = reshape(farcrop, (binL, psfL, binL, psfL))
    farcropbinmag = (abs.(farcropbin)).^2
    PSFbin = sum(farcropbinmag, dims=(1, 3))
    #multiply PSFs by arbitrary scaling factor and divide by freqency to get photon count
    PSF = scaling .* dropdims(PSFbin, dims=(1,3)) ./ freq 
end
    

function PSF_to_fftPSF(PSF, plan_PSF)
    fftPSF = plan_PSF * complex.(PSF) # needed because adjoint of fft does not project correctly
end

function get_PSF(freq, surrogate, pp, imgp, geoms, plan_nearfar, parallel)
    incident =  ChainRulesCore.ignore_derivatives( ()-> prepare_incident(pp,freq) ) 
    n2f_kernel =  ChainRulesCore.ignore_derivatives( ()-> prepare_n2f_kernel(pp,freq, plan_nearfar) )
    far = geoms_to_far(geoms, surrogate, incident, n2f_kernel, plan_nearfar, parallel)
    PSF = far_to_PSF(far, imgp.objL + imgp.imgL, imgp.binL, pp.PSF_scaling, freq)
end

function get_fftPSF(freq, surrogate, pp, imgp, geoms, plan_nearfar, plan_PSF, parallel)
    PSF = get_PSF(freq, surrogate, pp, imgp, geoms, plan_nearfar, parallel)
    fftPSF = PSF_to_fftPSF(PSF, plan_PSF)
end

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
    
    if parallel == true
        image_Tmap_grid_noiseless = ThreadsX.sum(iF->image_Tmap_grid_noiseless_iF(iF), 1:nF)
    else
        image_Tmap_grid_noiseless = sum(iF->image_Tmap_grid_noiseless_iF(iF), 1:nF)
    end
    
    #add nosie 
    if imgp.noise_abs == true
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
    
    if parallel == true
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


function dB_dT(T, freq, wavcen, scaling)
    c = convert(typeof(freq),299792458)
    h = convert(typeof(freq),6.62607015e-34)
    kb = convert(typeof(freq),1.380649e-23)
    
    A = (h * c * 10^6)/(wavcen * kb)
    numerator = (2 * A * freq^4 * exp(A * freq / T) ) 
    denominator = T^2 * (exp(A * freq / T) - 1)^2
    scaling * numerator/denominator
end

function d2B_dT2(T, freq, wavcen, scaling)
    c = convert(typeof(freq),299792458)
    h = convert(typeof(freq),6.62607015e-34)
    kb = convert(typeof(freq),1.380649e-23)
    
    A = (h * c * 10^6)/(wavcen * kb)
    numerator = A * freq^4 * (-2*T + A * freq * coth(A*freq/(2*T) ) )
    denominator = T^4 * (-1 + cosh(A*freq/T) )
    scaling * numerator/denominator
end

function gradient_reconstruction_T(Test_flat, image_diff_flat, α, pp, imgp, recp, fftPSFs, freqs, plan_nearfar, plan_PSF, weights, parallel)
    nF = pp.orderfreq + 1
    image_diff_grid = reshape(image_diff_flat, imgp.imgL, imgp.imgL)
    
    function term2(iF)
        dB_dT_diag = LinearAlgebra.Diagonal([ dB_dT(T, freqs[iF], pp.wavcen, pp.blackbody_scaling) for T in Test_flat])
        temp_term2 = Gtranspose( image_diff_grid,  fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF)
        dB_dT_diag * temp_term2
    end
    
    if parallel == true
        term2  = ThreadsX.sum(iF->term2(iF), 1:nF)
    else
        term2  = sum(iF->term2(iF), 1:nF)
    end

    2 * α * (Test_flat .- recp.subtract_reg ) + -2*term2
    
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
    
    get_image_diff_flat2(Test_flat) = get_image_diff_flat(Test_flat, image_Tmap_flat, pp, imgp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
    
    reconstruction_objective_simplified2(Test_flat, image_diff_flat, subtract_reg, print_objvals) = reconstruction_objective_simplified(Test_flat, α, image_diff_flat, subtract_reg, print_objvals)
    
    gradient_reconstruction_T2(Test_flat, image_diff_flat) = gradient_reconstruction_T(Test_flat, image_diff_flat, α, pp, imgp, recp, fftPSFs, freqs, plan_nearfar, plan_PSF, weights, parallel)
    
    #objective(Test_flat::Vector) = reconstruction_objective(Test_flat, image_Tmap_flat, pp, imgp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, parallel)

    if save_objvals
        objvals_filename = "ImagingOpt.jl/recdata/objvals_$(rec_id).csv"
    end

    function myfunc(Test_flat::Vector, grad::Vector)
        Test_flat = convert.(typeof(freqs[1]),Test_flat)
        image_diff_flat = get_image_diff_flat2(Test_flat)
        if length(grad) > 0
            #grad[:] = gradient( objective, Test_flat )[1]
            grad[:] = gradient_reconstruction_T2(Test_flat, image_diff_flat)
        end
        #obj = objective(Test_flat)
        term1, term2 = reconstruction_objective_simplified2(Test_flat, image_diff_flat, recp.subtract_reg, print_objvals)

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
    
    println(@sprintf("f tolerance is %e", recp.ftol_rel))
    println(@sprintf("α is %e", α))
    flush(stdout) 
        
    (minf,minT,ret) = optimize(opt, Tinit_flat)
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

function term1plusterm2_hes(α, pp, imgp, fftPSFs, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, parallel)
    nF = pp.orderfreq + 1
    Test_grid = reshape(Test_flat, imgp.objL, imgp.objL)
    B_Test_grid = prepare_blackbody(Test_grid, freqs, imgp, pp)
    
    term1hes = I*2*α
    
    if parallel == true
        image_Test_flat  = ThreadsX.sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    else
        image_Test_flat  = sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    end
    
    image_diff_grid = image_Tmap_grid - reshape(image_Test_flat, imgp.imgL, imgp.imgL)
    
    
    function term2hes_iF(iF)
        Diagonal( d2B_dT2.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) .* Gtranspose(image_diff_grid, fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF) )
    end
    
    if parallel == true
        term2hes = -2 * ThreadsX.sum(iF->term2hes_iF(iF), 1:nF)
    else
        term2hes = -2 * sum(iF->term2hes_iF(iF), 1:nF)
    end

    term1hes + term2hes
end

struct Hes{FloatType, IntType} <: LinearMap{FloatType}
    nF::IntType
    wavcen::FloatType
    objL::IntType
    imgL::IntType
    term1plusterm2_hessian::Diagonal{FloatType, Vector{FloatType}}
    fftPSFs::Vector{Matrix{Complex{FloatType}}}
    freqs::Vector{FloatType}
    Test_flat::Vector{FloatType}
    plan_nearfar::FFTW.cFFTWPlan{Complex{FloatType}, -1, true, 2, UnitRange{IntType}}
    plan_PSF::FFTW.cFFTWPlan{Complex{FloatType}, -1, true, 2, UnitRange{IntType}}
    weights::Vector{FloatType}
    blackbody_scaling::FloatType
    parallel::Bool
end

Base.size(H::Hes) = (H.objL^2, H.objL^2) 

function LinearMaps._unsafe_mul!(vectorout::AbstractVecOrMat, H::Hes, vectorin::AbstractVector)

    function  term3_rhs_iF(iF)
        out = dB_dT.(H.Test_flat, H.freqs[iF], H.wavcen, H.blackbody_scaling) .* vectorin
        G( reshape(out, H.objL, H.objL), H.fftPSFs[iF], H.weights[iF], H.freqs[1], H.freqs[end], H.plan_PSF)
    end
    
    if H.parallel == true
        term3_rhs  = ThreadsX.sum(iF->term3_rhs_iF(iF), 1:H.nF)
    else
        term3_rhs  = sum(iF->term3_rhs_iF(iF), 1:H.nF)
    end
    
    
    function term3_iF(iF)
        out = Gtranspose( reshape(term3_rhs, H.imgL, H.imgL), H.fftPSFs[iF], H.weights[iF], H.freqs[1], H.freqs[end], H.plan_PSF )
         (Diagonal([dB_dT(i, H.freqs[iF], H.wavcen, H.blackbody_scaling) for i in H.Test_flat]) * out )
    end
    
    if H.parallel == true
        vectorout .= 2 * ThreadsX.sum(iF->term3_iF(iF), 1:H.nF)
    else
        vectorout .= 2 * sum(iF->term3_iF(iF), 1:H.nF)
    end
    vectorout .= (H.term1plusterm2_hessian * vectorin) .+ vectorout
    
    
end
    
function build_hessian(α, pp, imgp, fftPSFs, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, parallel)
    nF = pp.orderfreq + 1

    term1plusterm2 = term1plusterm2_hes(α, pp, imgp, fftPSFs, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, parallel)
        
    function term3rhs_iF(iF)
        term3rhs_temp = zeros(typeof(freqs[1]), imgp.imgL^2, imgp.objL^2)
        for i = 1:imgp.objL^2
            input = zeros(typeof(freqs[1]), imgp.objL^2)
            input[i] = dB_dT(Test_flat[i], freqs[iF], pp.wavcen, pp.blackbody_scaling)
            # can this be done faster?
            out = G( reshape(input, imgp.objL, imgp.objL), fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF)
            term3rhs_temp[:, i] .= term3rhs_temp[:, i] .+ out
        end
        term3rhs_temp
    end
        
    if parallel == true
        term3rhs =  ThreadsX.sum(iF->term3rhs_iF(iF), 1:nF)
    else
        term3rhs = sum(iF->term3rhs_iF(iF), 1:nF)
    end

    term3 = 2 * transpose(term3rhs) * term3rhs
    
    term1plusterm2 + term3

end

function jacobian_vp_undiff(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, B_Tmap_grid, noise, parallel)
    nF = pp.orderfreq + 1
    #fftPSFs = [get_fftPSF(freqs[iF], surrogates[iF], weights[iF], pp, imgp, geoms, plan_nearfar, plan_PSF, parallel) for iF in 1:pp.orderfreq+1]
    function get_fftPSF_iF(iF)
        get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, plan_PSF, parallel)
    end
    fftPSFs = ThreadsX.map(get_fftPSF_iF, Array(1:nF))

    
    Test_grid = reshape(Test_flat, imgp.objL, imgp.objL)
    B_Test_grid = prepare_blackbody(Test_grid, freqs, imgp, pp)
    
    if parallel == true
        image_Test_flat  = ThreadsX.sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    else
        image_Test_flat  = sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    end
    image_diff_grid =  make_image(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, noise, plan_nearfar, plan_PSF, parallel) - reshape(image_Test_flat, imgp.imgL, imgp.imgL)

    function term2_iF(iF)
        -2 * lambda' * Diagonal(  dB_dT.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) ) * Gtranspose( image_diff_grid,  fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF)
    end
    
    if parallel == true
        jvp_undiff  = ThreadsX.sum(iF->term2_iF(iF), 1:nF)
    else
        jvp_undiff  = sum(iF->term2_iF(iF), 1:nF)
    end

    jvp_undiff
end

function jacobian_vp_autodiff(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, B_Tmap_grid, noise, parallel)
    gradient(geoms -> jacobian_vp_undiff(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, B_Tmap_grid, noise, parallel), geoms)[1]

end
        
function uvector_dot_Gtransposedg_vvector(pp, imgp, freq, surrogate, incident, n2f_kernel, geoms, weight, freqend, freq1, uflat, vgrid, plan_nearfar, plan_PSF, parallel)
    psfL = imgp.imgL + imgp.objL
    geoms_flat = geoms[:]
        
    far = geoms_to_far(geoms, surrogate, incident, n2f_kernel, plan_nearfar, parallel)
    #PSF_nomean = far_to_PSFsnomean(far, psfL, imgp.binL)[:]
        
    temp1 = plan_PSF * [reshape(uflat, imgp.objL, imgp.objL) zeros(typeof(freq),  imgp.objL, imgp.imgL); zeros(typeof(freq), imgp.imgL, imgp.objL + imgp.imgL) ]
    temp2 = plan_PSF \ [zeros(typeof(freq), imgp.objL, imgp.objL + imgp.imgL); zeros(typeof(freq), imgp.imgL, imgp.objL) vgrid]
    
    xvector = real.( plan_PSF * (temp1 .* temp2) .* weight .* (freqend - freq1) )[:]
    #dmean = (xvector * (psfL^2) / (sum(PSF_nomean)) ) .- ( (psfL^2) * (xvector' * PSF_nomean) / ( sum(PSF_nomean) )^2 )
    
    dmeantemp = repeat(reshape(xvector, psfL, psfL), inner=(imgp.binL,imgp.binL))
            
    dim = (pp.gridL - (psfL * imgp.binL) ) ÷ 2 
    dfarcrop = [zeros(typeof(freq), dim, pp.gridL); zeros(typeof(freq), (psfL * imgp.binL) , dim) dmeantemp zeros(typeof(freq), (psfL * imgp.binL) , dim); zeros(typeof(freq), dim, pp.gridL) ][:]
    
    objective_out = pp.PSF_scaling .* conj.(far)[:] ./ freq
    zeropadded = plan_nearfar \ [zeros(typeof(freq), pp.gridL, 2*pp.gridL); zeros(typeof(freq), pp.gridL, pp.gridL) reshape(dfarcrop .* objective_out,pp.gridL, pp.gridL) ]
     
    dsur_dg_function(geom) =  chebgradient(surrogate, geom)[2]
    dsur_dg = ThreadsX.map(dsur_dg_function, geoms)
            
     2*real( dsur_dg .* incident .* (plan_nearfar * (zeropadded .* n2f_kernel))[1:pp.gridL, 1:pp.gridL] );

end
        
function jacobian_vp_manual(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, B_Tmap_grid, noise, fftPSFs, parallel)
    nF = pp.orderfreq + 1
    psfL = imgp.imgL + imgp.objL
    freqend = freqs[end]
    freq1 = freqs[1]
    
    Test_grid = reshape(Test_flat, imgp.objL, imgp.objL)
    B_Test_grid = prepare_blackbody(Test_grid, freqs, imgp, pp)
    
    if parallel == true
        image_Test_flat  = ThreadsX.sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    else
        image_Test_flat  = sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    end
    image_diff_grid =  image_Tmap_grid - reshape(image_Test_flat, imgp.imgL, imgp.imgL)
   
            
    function term1_iF(iF)
        freq = freqs[iF]
        weight = weights[iF]
        surrogate = surrogates[iF]
        incident = prepare_incident(pp,freq)  
        n2f_kernel = prepare_n2f_kernel(pp,freq, plan_nearfar);
        uvector = (-2 * lambda .* dB_dT.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) ) 
        uvector_dot_Gtransposedg_vvector(pp, imgp, freq, surrogate, incident, n2f_kernel, geoms, weight, freqend, freq1, uvector, image_diff_grid, plan_nearfar, plan_PSF, parallel)
    end
            
    if parallel == true
        term1  = ThreadsX.sum(iF->term1_iF(iF), 1:nF)
    else
        term1  = sum(iF->term1_iF(iF), 1:nF)
    end      
    
        
    if parallel == true
        vgrid2  = ThreadsX.sum(iF->G( reshape(dB_dT.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) .* lambda , imgp.objL, imgp.objL)  , fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    else
        vgrid2  = sum(iF->G(reshape(dB_dT.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) .* lambda , imgp.objL, imgp.objL), fftPSFs[iF], weights[iF], freq1, freqend, plan_PSF), 1:nF)
    end
    vgrid2 = reshape(vgrid2, imgp.imgL, imgp.imgL)
        
        

    function term2_iF(iF)
        freq = freqs[iF]
        weight = weights[iF]
        surrogate = surrogates[iF]
        incident = prepare_incident(pp,freq)  
        n2f_kernel = prepare_n2f_kernel(pp,freq, plan_nearfar);
        uvector = -2*( B_Tmap_grid[:,:,iF]  -  B_Test_grid[:,:,iF])[:]
        uvector_dot_Gtransposedg_vvector(pp, imgp, freq, surrogate, incident, n2f_kernel, geoms, weight, freqend, freq1, uvector, vgrid2, plan_nearfar, plan_PSF, parallel)
    end
            
    if parallel == true
        term2  = ThreadsX.sum(iF->term2_iF(iF), 1:nF)
    else
        term2  = sum(iF->term2_iF(iF), 1:nF)
    end
        
    function term3constant_iF(iF)
        (-2 / imgp.imgL^2 ) * lambda' * (dB_dT.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) .* Gtranspose(noise, fftPSFs[iF], weights[iF], freq1, freqend, plan_PSF) )
    end
    
    if parallel == true
        term3constant  = ThreadsX.sum(iF->term3constant_iF(iF), 1:nF)
    else
        term3constant  = sum(iF->term3constant_iF(iF), 1:nF)
    end
    
        
    function term3_iF(iF)
        freq = freqs[iF]
        weight = weights[iF]
        surrogate = surrogates[iF]
        incident = prepare_incident(pp,freq)  
        n2f_kernel = prepare_n2f_kernel(pp,freq, plan_nearfar);
        uvector_dot_Gtransposedg_vvector(pp, imgp, freq, surrogate, incident, n2f_kernel, geoms, weight, freqend, freq1, B_Tmap_grid[:,:,iF], ones(imgp.imgL, imgp.imgL), plan_nearfar, plan_PSF, parallel)
    end
        
    if parallel == true
        term3  = term3constant * ThreadsX.sum(iF->term3_iF(iF), 1:nF)
    else
        term3  = term3constant  * sum(iF->term3_iF(iF), 1:nF)
    end
    
    term1 + term2 + term3
    
end
    
    
function dloss_dparams(pp, imgp, optp, recp, geoms, α, Tmap, B_Tmap_grid, Test_flat, image_Tmap_grid, noise, fftPSFs, surrogates, freqs, plan_nearfar, plan_PSF, weights, parallel)
    if optp.optimize_alpha
        grad = Vector{Float64}(undef,pp.gridL^2 + 1) 
    else
        grad = Vector{Float64}(undef,pp.gridL^2 )
    end
        
    term1plusterm2_hessian = term1plusterm2_hes(α, pp, imgp, fftPSFs, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, parallel);
    H = Hes(pp.orderfreq + 1, pp.wavcen, imgp.objL, imgp.imgL, term1plusterm2_hessian, fftPSFs, freqs, Test_flat, plan_nearfar, plan_PSF, weights, pp.blackbody_scaling, parallel)
    b = 2 * (Tmap[:] - Test_flat) / (Tmap[:]' * Tmap[:])

    lambda = zeros(typeof(freqs[1]), imgp.objL^2)
    lambda, ch = cg!(lambda, H, b, log=true, maxiter = optp.cg_maxiter_factor * imgp.objL^2);
    println(ch)
    flush(stdout)
        
    jacobian_vp = jacobian_vp_manual(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, B_Tmap_grid, noise, fftPSFs, parallel)[:]
        
    if optp.optimize_alpha
        grad[1:end-1] = jacobian_vp
        grad[end] = 2 * (1/optp.α_scaling) * (lambda' * (Test_flat .- recp.subtract_reg ) )
    else
        grad[1:end] = jacobian_vp
    end
    grad, length(ch[:resnorm])
end
    
#open(file_save_cg_iters, "a") do io
#    writedlm(io, length(ch[:resnorm]), ',')
#end
    
#(1/imgp.objN)