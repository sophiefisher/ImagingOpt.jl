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

#chain rule for nearfield 
function ChainRulesCore.rrule(::typeof(nearfield), incident, surrogate, geoms, parallel)
    if parallel
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


function term1plusterm2_hes(α, pp, imgp, fftPSFs, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, parallel)
    nF = pp.orderfreq + 1
    Test_grid = reshape(Test_flat, imgp.objL, imgp.objL)
    B_Test_grid = prepare_blackbody(Test_grid, freqs, imgp, pp)
    
    term1hes = I*2*α
    
    if parallel
        image_Test_flat  = ThreadsX.sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    else
        image_Test_flat  = sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    end
    
    image_diff_grid = image_Tmap_grid - reshape(image_Test_flat, imgp.imgL, imgp.imgL)
    
    
    function term2hes_iF(iF)
        Diagonal( d2B_dT2.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) .* Gtranspose(image_diff_grid, fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF) )
    end
    
    if parallel
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
    
    if H.parallel
        term3_rhs  = ThreadsX.sum(iF->term3_rhs_iF(iF), 1:H.nF)
    else
        term3_rhs  = sum(iF->term3_rhs_iF(iF), 1:H.nF)
    end
    
    
    function term3_iF(iF)
        out = Gtranspose( reshape(term3_rhs, H.imgL, H.imgL), H.fftPSFs[iF], H.weights[iF], H.freqs[1], H.freqs[end], H.plan_PSF )
         (Diagonal([dB_dT(i, H.freqs[iF], H.wavcen, H.blackbody_scaling) for i in H.Test_flat]) * out )
    end
    
    if H.parallel
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
        
    if parallel
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
    
    if parallel
        image_Test_flat  = ThreadsX.sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    else
        image_Test_flat  = sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    end
    image_diff_grid =  make_image(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, noise, plan_nearfar, plan_PSF, parallel) - reshape(image_Test_flat, imgp.imgL, imgp.imgL)

    function term2_iF(iF)
        -2 * lambda' * Diagonal(  dB_dT.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) ) * Gtranspose( image_diff_grid,  fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF)
    end
    
    if parallel
        jvp_undiff  = ThreadsX.sum(iF->term2_iF(iF), 1:nF)
    else
        jvp_undiff  = sum(iF->term2_iF(iF), 1:nF)
    end

    jvp_undiff
end

function jacobian_vp_autodiff(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, B_Tmap_grid, noise, parallel)
    gradient(geoms -> jacobian_vp_undiff(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, B_Tmap_grid, noise, parallel), geoms)[1]

end
    
function get_dsur_dg_times_incident(pp, freq, surrogate, geoms, parallel)
    incident = prepare_incident(pp,freq) 
    dsur_dg_function(geom) =  chebgradient(surrogate, geom)[2]
    if parallel
        dsur_dg = ThreadsX.map(dsur_dg_function, geoms)
    else
        dsur_dg = map(dsur_dg_function, geoms)
    end
    dsur_dg .* incident
end
        
function uvector_dot_Gtransposedg_vvector(pp, imgp, freq, far, dsur_dg_times_incident, n2f_kernel, geoms, weight, freqend, freq1, uflat, vgrid, plan_nearfar, plan_PSF, parallel)
    psfL = imgp.imgL + imgp.objL
    geoms_flat = geoms[:]
        
    #far = geoms_to_far(geoms, surrogate, incident, n2f_kernel, plan_nearfar, parallel)
    #PSF_nomean = far_to_PSFsnomean(far, psfL, imgp.binL)[:]
        
    temp1 = plan_PSF * [reshape(uflat, imgp.objL, imgp.objL) zeros(typeof(freq),  imgp.objL, imgp.imgL); zeros(typeof(freq), imgp.imgL, imgp.objL + imgp.imgL) ]
    temp2 = plan_PSF \ [zeros(typeof(freq), imgp.objL, psfL); zeros(typeof(freq), imgp.imgL, imgp.objL) vgrid]
    xvector = real.( plan_PSF * (temp1 .* temp2) .* weight .* (freqend - freq1) )[:]
    #dmean = (xvector * (psfL^2) / (sum(PSF_nomean)) ) .- ( (psfL^2) * (xvector' * PSF_nomean) / ( sum(PSF_nomean) )^2 )
    
    dbinPSF = repeat(reshape(xvector, psfL, psfL), inner=(imgp.binL,imgp.binL))
            
    #dim = (pp.gridL - (psfL * imgp.binL) ) ÷ 2 
    #dfarcrop = [zeros(typeof(freq), dim, pp.gridL); zeros(typeof(freq), (psfL * imgp.binL) , dim) dmeantemp zeros(typeof(freq), (psfL * imgp.binL) , dim); zeros(typeof(freq), dim, pp.gridL) ][:]
    
    objective_out = pp.PSF_scaling .* conj.(far) ./ freq
     
    #dsur_dg_function(geom) =  chebgradient(surrogate, geom)[2]
    #if parallel
    #    dsur_dg = ThreadsX.map(dsur_dg_function, geoms)
    #else
    #    dsur_dg = map(dsur_dg_function, geoms)
    #end

    2*real( dsur_dg_times_incident .* convolveT(  dbinPSF .* objective_out, n2f_kernel, plan_nearfar) );

end
        
function jacobian_vp_manual(lambda, pp, imgp,  geoms, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, B_Tmap_grid, noise, fftPSFs, dsur_dg_times_incidents, far_fields, parallel)
    nF = pp.orderfreq + 1
    psfL = imgp.imgL + imgp.objL
    freqend = freqs[end]
    freq1 = freqs[1]
    
    Test_grid = reshape(Test_flat, imgp.objL, imgp.objL)
    B_Test_grid = prepare_blackbody(Test_grid, freqs, imgp, pp)
    
    if parallel
        image_Test_flat  = ThreadsX.sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    else
        image_Test_flat  = sum(iF->G(B_Test_grid[:,:,iF], fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    end
    image_diff_grid =  image_Tmap_grid - reshape(image_Test_flat, imgp.imgL, imgp.imgL)
   
            
    function term1_iF(iF)
        freq = freqs[iF]
        weight = weights[iF]
        far = far_fields[iF]
        dsur_dg_times_incident = dsur_dg_times_incidents[iF]
        n2f_kernel = prepare_n2f_kernel(pp,imgp,freq, plan_nearfar);
        uvector = (-2 * lambda .* dB_dT.(Test_flat, freq, pp.wavcen, pp.blackbody_scaling) ) 
        uvector_dot_Gtransposedg_vvector(pp, imgp, freq, far, dsur_dg_times_incident, n2f_kernel, geoms, weight, freqend, freq1, uvector, image_diff_grid, plan_nearfar, plan_PSF, parallel)
    end
            
    if parallel
        term1  = ThreadsX.sum(iF->term1_iF(iF), 1:nF)
    else
        term1  = sum(iF->term1_iF(iF), 1:nF)
    end  
        
        
    if parallel
        vgrid2  = ThreadsX.sum(iF->G( reshape(dB_dT.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) .* lambda , imgp.objL, imgp.objL)  , fftPSFs[iF], weights[iF], freqs[1], freqs[end], plan_PSF), 1:nF)
    else
        vgrid2  = sum(iF->G(reshape(dB_dT.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) .* lambda , imgp.objL, imgp.objL), fftPSFs[iF], weights[iF], freq1, freqend, plan_PSF), 1:nF)
    end
    vgrid2 = reshape(vgrid2, imgp.imgL, imgp.imgL)
        
        

    function term2_iF(iF)
        freq = freqs[iF]
        weight = weights[iF]
        dsur_dg_times_incident = dsur_dg_times_incidents[iF]
        far = far_fields[iF]
        n2f_kernel = prepare_n2f_kernel(pp,imgp,freq, plan_nearfar);
        uvector = -2*( B_Tmap_grid[:,:,iF]  -  B_Test_grid[:,:,iF])[:]
        uvector_dot_Gtransposedg_vvector(pp, imgp, freq, far, dsur_dg_times_incident, n2f_kernel, geoms, weight, freqend, freq1, uvector, vgrid2, plan_nearfar, plan_PSF, parallel)
    end
            
    if parallel
        term2  = ThreadsX.sum(iF->term2_iF(iF), 1:nF)
    else
        term2  = sum(iF->term2_iF(iF), 1:nF)
    end
        
    function term3constant_iF(iF)
        (-2 / imgp.imgL^2 ) * lambda' * (dB_dT.(Test_flat, freqs[iF], pp.wavcen, pp.blackbody_scaling) .* Gtranspose(noise, fftPSFs[iF], weights[iF], freq1, freqend, plan_PSF) )
    end
    
    if parallel
        term3constant  = ThreadsX.sum(iF->term3constant_iF(iF), 1:nF)
    else
        term3constant  = sum(iF->term3constant_iF(iF), 1:nF)
    end
    
        
    function term3_iF(iF)
        freq = freqs[iF]
        weight = weights[iF]
        dsur_dg_times_incident = dsur_dg_times_incidents[iF]
        far = far_fields[iF]
        n2f_kernel = prepare_n2f_kernel(pp,imgp,freq, plan_nearfar);
        uvector_dot_Gtransposedg_vvector(pp, imgp, freq, far, dsur_dg_times_incident, n2f_kernel, geoms, weight, freqend, freq1, B_Tmap_grid[:,:,iF], ones(imgp.imgL, imgp.imgL), plan_nearfar, plan_PSF, parallel)
    end
        
    if parallel
        term3  = term3constant * ThreadsX.sum(iF->term3_iF(iF), 1:nF)
    else
        term3  = term3constant  * sum(iF->term3_iF(iF), 1:nF)
    end
    
    #=
    figure(figsize=(8,3))
    subplot(1,2,1)
    imshow(reshape(term1, pp.gridL, pp.gridL))
    colorbar()
        
    subplot(1,2,2)
    imshow(reshape(term2, pp.gridL, pp.gridL))
    colorbar()
    =#
    
    term1 + term2 + term3
    
end
    
    
function dloss_dparams(pp, imgp, optp, recp, geoms, α, Tmap, B_Tmap_grid, Test_flat, image_Tmap_grid, noise, fftPSFs, dsur_dg_times_incidents, far_fields, freqs, plan_nearfar, plan_PSF, weights, parallel)
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
    #println(ch)
    #flush(stdout)
        
    jacobian_vp = jacobian_vp_manual(lambda, pp, imgp,  geoms, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, B_Tmap_grid, noise, fftPSFs, dsur_dg_times_incidents, far_fields, parallel)[:]
        
    if optp.optimize_alpha
        grad[1:end-1] = jacobian_vp
        grad[end] = 2 * (1/optp.α_scaling) * (lambda' * (Test_flat .- recp.subtract_reg ) )
    else
        grad[1:end] = jacobian_vp
    end
    grad, length(ch[:resnorm])
end