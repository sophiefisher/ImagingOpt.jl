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

function ChainRules.rrule(config::RuleConfig{>:HasReverseMode}, ::typeof(ThreadsX.map), f::F, xs::Tuple...) where {F}
    println("HI")
    flush(stdout)
    length_y = minimum(length, xs)
    hobbits = ntuple(length_y) do i
        args = getindex.(xs, i)
        rrule_via_ad(config, f, args...)
    end
    y = ThreadsX.map(first, hobbits)
    num_xs = Val(length(xs))
    paddings = ThreadsX.map(x -> ntuple(Returns(NoTangent()), (length(x) - length_y)), xs)
    all(isempty, paddings) || @error """map(f, xs::Tuple...) does not allow mistmatched lengths!
        But its `rrule` does; when JuliaLang/julia #42216 is fixed this warning should be removed."""
    function map_pullback(dy_raw)
        dy = unthunk(dy_raw)
        # We want to call the pullbacks in `rrule_via_ad` in reverse sequence to the forward pass:
        backevals = ntuple(length_y) do i
            rev_i = length_y - i + 1
            last(hobbits[rev_i])(dy[rev_i])
        end |> reverse
        # This df doesn't infer, could test Base.issingletontype(F), but it's not the only inference problem.
        df = ProjectTo(f)(ThreadsX.sum(first, backevals))
        # Now unzip that. Because `map` like `zip` should when any `x` stops, some `dx`s may need padding.
        # Although in fact, `map(+, (1,2), (3,4,5))` is an error... https://github.com/JuliaLang/julia/issues/42216
        dxs = ntuple(num_xs) do k
            dx_short = ThreadsX.map(bv -> bv[k+1], backevals)
            ProjectTo(xs[k])((dx_short..., paddings[k]...))  # ProjectTo makes the Tangent for us
        end
        return (NoTangent(), df, dxs...)
    end
    map_back(dy::AbstractZero) = (NoTangent(), NoTangent(), ntuple(Returns(NoTangent()), num_xs)...)
    return y, map_pullback
end

# Implementations of differentiable functions that form optimization pipeline
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

function PSFs_to_fftPSFs(PSFs, plan_PSF)
    PSFsC = complex.(PSFs) # needed because adjoint of fft does not project correctly
    fftPSFs = plan_PSF * PSFsC
end

function get_PSF(freq, surrogate, weight, pp, imgp, geoms, plan_nearfar, parallel)
    incident =  ChainRulesCore.ignore_derivatives( ()-> prepare_incident(pp,freq) ) 
    n2f_kernel =  ChainRulesCore.ignore_derivatives( ()-> prepare_n2f_kernel(pp,freq, plan_nearfar) )
    far = geoms_to_far(geoms, surrogate, incident, n2f_kernel, plan_nearfar, parallel)
    PSF = far_to_PSFs(far, imgp.objL + imgp.imgL, imgp.binL)
end

function get_fftPSF(freq, surrogate, weight, pp, imgp, geoms, plan_nearfar, plan_PSF, parallel)
    PSF = get_PSF(freq, surrogate, weight, pp, imgp, geoms, plan_nearfar, parallel)
    fftPSF = PSFs_to_fftPSFs(PSF, plan_PSF)
end

function make_images(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
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
        image_Tmap_grid = image_Tmap_grid_noiseless .+ mean(image_Tmap_grid_noiseless)* imgp.noise_level .*randn.(floattype)
    end
    image_Tmap_grid
end

#=
function reconstruction_objective(Test_flat::Vector, image_Tmap_flat, pp, imgp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, parallel::Bool=true)
    nF = pp.orderfreq + 1
    Test_grid = reshape(Test_flat, imgp.objL, imgp.objL)
    B_Test_grid = prepare_blackbody(Test_grid, freqs, imgp, pp)

    function forward(iF)
        
    end
    
    if parallel == true
        image_Test_flat  = ThreadsX.sum(iF-> G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    else
        image_Test_flat  = sum(iF-> G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    end
    
    (image_Tmap_flat - image_Test_flat)'*(image_Tmap_flat - image_Test_flat) + α*Test_flat'*Test_flat
end
=#


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


function reconstruction_objective_simplified(Test_flat, α, image_diff_flat)
    image_diff_flat'*image_diff_flat + α*Test_flat'*Test_flat
end


function dB_dT(T, freq, wavcen)
    c = convert(typeof(freq),299792458)
    h = convert(typeof(freq),6.62607015e-34)
    kb = convert(typeof(freq),1.380649e-23)
    
    A = (h * c * 10^6)/(wavcen * kb)
    numerator = (2 * A * freq^4 * exp(A * freq / T) ) 
    denominator = T^2 * (exp(A * freq / T) - 1)^2
    numerator/denominator
end

function d2B_dT2(T, freq, wavcen)
    c = convert(typeof(freq),299792458)
    h = convert(typeof(freq),6.62607015e-34)
    kb = convert(typeof(freq),1.380649e-23)
    
    A = (h * c * 10^6)/(wavcen * kb)
    numerator = A * freq^4 * (-2*T + A * freq * coth(A*freq/(2*T) ) )
    denominator = T^4 * (-1 + cosh(A*freq/T) )
    numerator/denominator
end

function gradient_reconstruction_T(Test_flat, image_diff_flat, α, pp, imgp, fftPSFs, freqs, plan_nearfar, plan_PSF, weights, parallel)
    nF = pp.orderfreq + 1
    image_diff_grid = reshape(image_diff_flat, imgp.imgL, imgp.imgL)
    
    function term2(iF)
        dB_dT_diag = LinearAlgebra.Diagonal([ dB_dT(T, freqs[iF], pp.wavcen) for T in Test_flat])
        temp_term2 = Gtranspose( image_diff_grid,  fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF)
        dB_dT_diag * temp_term2
    end
    
    if parallel == true
        term2  = ThreadsX.sum(iF->term2(iF), 1:nF)
    else
        term2  = sum(iF->term2(iF), 1:nF)
    end

    2 * α * Test_flat + -2*term2
    
end

#reconstruction
function reconstruct_object(image_Tmap_grid, Tmap, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, save::Bool=true, parallel::Bool=true)
    nF = pp.orderfreq + 1
    image_Tmap_flat = image_Tmap_grid[:]
    
    get_image_diff_flat2(Test_flat) = get_image_diff_flat(Test_flat, image_Tmap_flat, pp, imgp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
    
    reconstruction_objective_simplified2(Test_flat, image_diff_flat) = reconstruction_objective_simplified(Test_flat, α, image_diff_flat)
    
    gradient_reconstruction_T2(Test_flat, image_diff_flat) = gradient_reconstruction_T(Test_flat, image_diff_flat, α, pp, imgp, fftPSFs, freqs, plan_nearfar, plan_PSF, weights, parallel)
    
    #objective(Test_flat::Vector) = reconstruction_objective(Test_flat, image_Tmap_flat, pp, imgp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, parallel)


    function myfunc(Test_flat::Vector, grad::Vector)
        Test_flat = convert.(typeof(freqs[1]),Test_flat)
        image_diff_flat = get_image_diff_flat2(Test_flat)
        if length(grad) > 0
            #grad[:] = gradient( objective, Test_flat )[1]
            grad[:] = gradient_reconstruction_T2(Test_flat, image_diff_flat)
        end
        #obj = objective(Test_flat)
        obj = reconstruction_objective_simplified2(Test_flat, image_diff_flat)
        #println(obj)
        flush(stdout)
        obj
    end
    
    opt = Opt(:LD_LBFGS, imgp.objL^2)
    opt.lower_bounds = repeat([ convert(typeof(pp.lbfreq),3.0),], imgp.objL^2)
    opt.xtol_rel = recp.xtol_rel
    opt.min_objective = myfunc
    
    println(@sprintf("x tolerance is %e", recp.xtol_rel))
    println(@sprintf("α is %e", α))
    flush(stdout) 
        
    (minf,minT,ret) = optimize(opt, Tinit_flat)
    minT = convert.(typeof(freqs[1]),minT)
    
    if save==true
        Tmaps_filename = @sprintf("ImagingOpt.jl/recdata/Tmap_%s_%s_%d_%d_%.2e_%.2f_%s_%.2e_%.2f.csv", imgp.object_savefilestring, optp.geoms_init_savefilestring,  imgp.objL, imgp.imgL, α, imgp.noise_level, string(imgp.noise_abs), recp.xtol_rel, imgp.emiss_noise_level);
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
        Diagonal( d2B_dT2.(Test_flat, freqs[iF], pp.wavcen) .* Gtranspose(image_diff_grid, fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF) )
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
    parallel::Bool
end

Base.size(H::Hes) = (H.objL^2, H.objL^2) 

function LinearMaps._unsafe_mul!(vectorout::AbstractVecOrMat, H::Hes, vectorin::AbstractVector)

    function  term3_rhs_iF(iF)
        out = dB_dT.(H.Test_flat, H.freqs[iF], H.wavcen) .* vectorin
        G( reshape(out, H.objL, H.objL), H.fftPSFs[iF], H.weights[iF], H.freqs[1], H.freqs[end], H.plan_PSF)
    end
    
    if H.parallel == true
        term3_rhs  = ThreadsX.sum(iF->term3_rhs_iF(iF), 1:H.nF)
    else
        term3_rhs  = sum(iF->term3_rhs_iF(iF), 1:H.nF)
    end
    
    
    function term3_iF(iF)
        out = Gtranspose( reshape(term3_rhs, H.imgL, H.imgL), H.fftPSFs[iF], H.weights[iF], H.freqs[1], H.freqs[end], H.plan_PSF )
         (Diagonal([dB_dT(i, H.freqs[iF], H.wavcen) for i in H.Test_flat]) * out )
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
            input[i] = dB_dT(Test_flat[i], freqs[iF], pp.wavcen)
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

function jacobian_vp_undiff(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, Tmap, parallel)
    nF = pp.orderfreq + 1
    #fftPSFs = [get_fftPSF(freqs[iF], surrogates[iF], weights[iF], pp, imgp, geoms, plan_nearfar, plan_PSF, parallel) for iF in 1:pp.orderfreq+1]
    function hi(iF)
        get_fftPSF(freqs[iF], surrogates[iF], weights[iF], pp, imgp, geoms, plan_nearfar, plan_PSF, parallel)
    end
    fftPSFs = ThreadsX.map(hi, Tuple(1:nF))

    
    Test_grid = reshape(Test_flat, imgp.objL, imgp.objL)
    B_Test_grid = prepare_blackbody(Test_grid, freqs, imgp, pp)
    
    if parallel == true
        image_Test_flat  = ThreadsX.sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    else
        image_Test_flat  = sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    end
    image_diff_grid =  image_Tmap_grid - reshape(image_Test_flat, imgp.imgL, imgp.imgL)

    function term2_iF(iF)
        -2 * lambda' * Diagonal(  dB_dT.(Test_flat, freqs[iF], pp.wavcen) ) * Gtranspose( image_diff_grid,  fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF)
    end
    
    if parallel == true
        jvp_undiff  = ThreadsX.sum(iF->term2_iF(iF), 1:nF)
    else
        jvp_undiff  = sum(iF->term2_iF(iF), 1:nF)
    end

    jvp_undiff
end

function jacobian_vp(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, Tmap, parallel)
    gradient(geoms -> jacobian_vp_undiff(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, Tmap, parallel), geoms)[1]

end
    