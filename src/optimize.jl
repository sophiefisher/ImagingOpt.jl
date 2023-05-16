# collection of useful jobs to run on a params file

### Projecting a tuple to SMatrix leads to ChainRulesCore._projection_mismatch by default, so overloaded here
function (project::ChainRulesCore.ProjectTo{<:Tangent{<:Tuple}})(dx::SArray)
    dy = reshape(dx, axes(project.elements))  # allows for dx::OffsetArray
    dz = ntuple(i -> project.elements[i](dy[i]), length(project.elements))
    return ChainRulesCore.project_type(project)(dz...)
end

### Project SArray to SArray
function ChainRulesCore.ProjectTo(x::SArray{S,T}) where {S, T}
    return ChainRulesCore.ProjectTo{SArray}(; element=ChainRulesCore._eltype_projectto(T), axes=S)
end

function (project::ChainRulesCore.ProjectTo{SArray})(dx::AbstractArray{S,M}) where {S,M}
    return SArray{project.axes}(dx)
end

### Adjoint for SArray constructor

function ChainRulesCore.rrule(::Type{T}, x::Tuple) where {T<:SArray}
    project_x = ProjectTo(x)
    Array_pullback(ȳ) = (NoTangent(), project_x(ȳ))
    return T(x), Array_pullback
end

StructTypes.StructType(::Type{PhysicsParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{ImagingParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{OptimizeParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{ReconstructionParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{JobParams}) = StructTypes.Struct()

const PARAMS_DIR = "ImagingOpt.jl/params"

function get_params(pname, presicion)
    
    if presicion == "double"
        floattype = Float64
        inttype = Int64
    elseif presicion == "single"
        floattype = Float32
        inttype = Int32
    end
    
    jsonread = JSON3.read(read("$PARAMS_DIR/$pname.json", String))
    pp_temp = jsonread.pp
    pp = PhysicsParams{floattype, inttype}([pp_temp[key] for key in keys(pp_temp)]...)
    
    imgp_temp = jsonread.imgp
    imgp = ImagingParams{floattype, inttype}([imgp_temp[key] for key in keys(imgp_temp)]...)
    
    optp_temp = jsonread.optp
    optp = OptimizeParams{floattype, inttype}([optp_temp[key] for key in keys(optp_temp)]...)
    
    recp_temp = jsonread.recp
    recp = ReconstructionParams{floattype}([recp_temp[key] for key in keys(recp_temp)]...)
    params = JobParams(pp, imgp, optp, recp)
end

function design_minimax_lens(pname, presicion="double", parallel=true, opt_name=:LD_MMA, num_iters=1000, opt_xtol_rel=1e-6, inequal_tol=1e-8, save_objective_data=true)
    #assumptions: starting from a uniform metasurface
    #not sure if single presicion works

    params = get_params(pname, presicion)
    pp = params.pp
    imgp = params.imgp
    surrogates, freqs = prepare_surrogate(pp)
    geoms_init = fill((pp.lbwidth + pp.ubwidth)/2, pp.gridL, pp.gridL);
    #geoms_init =  rand(pp.lbwidth:eps():pp.ubwidth,pp.gridL, pp.gridL)
    plan_nearfar = plan_fft!(zeros(Complex{typeof(freqs[1])}, (2*pp.gridL, 2*pp.gridL)), flags=FFTW.MEASURE)
    plan_PSF = plan_fft!(zeros(Complex{typeof(freqs[1])}, (imgp.objL + imgp.imgL, imgp.objL + imgp.imgL)), flags=FFTW.MEASURE)
    weights = convert.( typeof(freqs[1]), ClenshawCurtisQuadrature(pp.orderfreq + 1).weights)

    objective_iter_filename = @sprintf("geomsdata/minimax_objectivedata_%s_%d_%.2e_%d.csv",string(opt_name),pp.gridL,opt_xtol_rel,num_iters )

    psfL = imgp.objL + imgp.imgL
    middle = div(psfL,2)
    nF = pp.orderfreq + 1

    t_init = minimum(1:nF) do iF
        PSF = get_PSF(freqs[iF], surrogates[iF], weights[iF], pp, imgp, geoms_init, plan_nearfar, parallel)
        PSF[middle,middle] + PSF[middle+1,middle] + PSF[middle,middle+1] + PSF[middle+1,middle+1]
    end
    x_init = [geoms_init[:]; t_init]

    function myfunc(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1:end-1] = zeros(typeof(freqs[1]), pp.gridL^2)
            grad[end] = 1
        end
        println(x[end])
        flush(stdout)

        if save_objective_data == true
            open(objective_iter_filename, "a") do io
                writedlm(io, x[end], ',')
            end
        end

        x[end]
    end

    function design_broadband_lens_objective(freq, surrogate, weight, geoms)
        PSF = get_PSF(freq, surrogate, weight, pp, imgp, geoms, plan_nearfar, parallel)
        PSF[middle,middle] + PSF[middle+1,middle] + PSF[middle,middle+1] + PSF[middle+1,middle+1]
    end

    function myconstraint(x::Vector, grad::Vector, iF)
        geoms_grid = reshape(x[1:end-1], pp.gridL, pp.gridL)
        freq = freqs[iF]
        surrogate = surrogates[iF]
        weight = weights[iF]
        constraint = g -> design_broadband_lens_objective(freq, surrogate, weight, g)
        if length(grad) > 0
            grad[end] = 1
            grad_grid = -1 * gradient( constraint, geoms_grid )[1]
            grad[1:end-1] = grad_grid[:]
        end
        x[end] - constraint(geoms_grid)
    end

    opt = Opt(opt_name, pp.gridL^2 + 1)
    opt.lower_bounds = [fill(pp.lbwidth,pp.gridL^2); -Inf]
    opt.upper_bounds = [fill(pp.ubwidth,pp.gridL^2); Inf]
    opt.max_objective = myfunc
    for iF = 1:nF
        inequality_constraint!(opt, (x,grad) -> myconstraint(x, grad, iF), inequal_tol)
    end
    opt.xtol_rel = opt_xtol_rel
    opt.maxeval = num_iters

    (maxf,maxx,ret) = optimize(opt, x_init)
    geoms_filename = @sprintf("geomsdata/minimax_geoms_%s_%d_%.2e_%d.csv",string(opt_name),pp.gridL,xtol_rel,num_iters )
    writedlm( geoms_filename,  maxx[1:end-1],',')
    println(ret)
end


#function design_singlefreq_lens()
#end


function run_opt(pname, presicion, parallel)
    params = get_params(pname, presicion)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    surrogates, freqs = prepare_surrogate(pp)
    Tinit_flat = prepare_reconstruction(recp, imgp)
    Tmaps = prepare_objects(imgp, pp)
    
    plan_nearfar = plan_fft!(zeros(Complex{typeof(freqs[1])}, (2*pp.gridL, 2*pp.gridL)), flags=FFTW.MEASURE)
    plan_PSF = plan_fft!(zeros(Complex{typeof(freqs[1])}, (imgp.objL + imgp.imgL, imgp.objL + imgp.imgL)), flags=FFTW.MEASURE)
    weights = convert.( typeof(freqs[1]), ClenshawCurtisQuadrature(pp.orderfreq + 1).weights)
    
    function myfunc(parameters::Vector, grad::Vector)
        start = time()
        #parameters has geoms first, then reconstruction parameter alpha
        geoms = reshape(parameters[1:end-1], pp.gridL, pp.gridL)
        α = parameters[end]
        
        objective = convert(typeof(freqs[1]), 0)
        grad[:] = zeros(typeof(freqs[1]), pp.gridL^2 + 1)
        
        for obji = 1:imgp.objN
            Tmap = Tmaps[obji]
            B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
            
            #shouldn't this be outside the loop?
            fftPSFs = [get_fftPSF(freqs[iF], surrogates[iF], weights[iF], pp, imgp, geoms, plan_nearfar, plan_PSF, parallel) for iF in 1:pp.orderfreq+1]
            image_Tmap_grid = make_images(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel);
            Test_flat = reconstruct_object(image_Tmap_grid, Tmap, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, parallel)
            
            objective = objective +  (1/imgp.objN) * ( (Tmap[:] - Test_flat)'*(Tmap[:] - Test_flat) ) / (Tmap[:]' * Tmap[:])

            term1plusterm2_hessian = term1plusterm2_hes(α, pp, imgp, fftPSFs, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, parallel);
            H = Hes(pp.orderfreq + 1, pp.wavcen, imgp.objL, imgp.imgL, term1plusterm2_hessian, fftPSFs, freqs, Test_flat, plan_nearfar, plan_PSF, weights, parallel)
            b = 2 * (Tmap[:] - Test_flat) / (Tmap[:]' * Tmap[:])

            lambda = zeros(typeof(freqs[1]), imgp.objL^2)
            cg!(lambda, H, b);
            if length(grad) > 0
                grad[1:end-1] = grad[1:end-1] + ( (1/imgp.objN) * jacobian_vp(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, Tmap, parallel)[:] )
                grad[end] = grad[end] + ( (1/imgp.objN) * 2 * lambda' * Test_flat )
            end
        end
        elapsed = time() - start
        println()
        println(@sprintf("time elapsed = %f",elapsed))
        println(@sprintf("OBJECTIVE VAL IS %f",objective) )
        println()
        flush(stdout)
        objective
    end
    
    opt = Opt(:LD_LBFGS, pp.gridL^2 + 1)
    opt.min_objective = myfunc
    
    opt.lower_bounds = [repeat([ pp.lbwidth,], pp.gridL^2); eps()]
    opt.upper_bounds = [repeat([ pp.ubwidth,], pp.gridL^2); Inf]
    
    opt.xtol_rel = optp.xtol_rel
    opt.maxeval = optp.maxeval
    
    geoms_init = prepare_geoms(params)[:]
    parameters_init = [geoms_init; optp.αinit]
    
    (minobj,minparams,ret) = optimize(opt, parameters_init)
end


#=
function test_init(pname)
    params = get_params(pname)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    
    surrogates, freqs = prepare_surrogate(pp)
    geoms = prepare_geoms(params)
    
    Tmaps = prepare_objects(imgp, pp)
    Bs = prepare_blackbody(Tmaps, freqs, imgp, pp)
    
    ys = [zeros(imgp.imgL, imgp.imgL) for _ in 1:imgp.objN]
    (;pp, imgp, optp, Bs, surrogates, freqs, geoms, ys)
end

function test_forwardmodel_perfreq(pp, imgp, Bs, surrogate, freq, iF, geoms, ys)
    psfL = imgp.objL + imgp.imgL
    nF = pp.orderfreq + 1
    
    incident, n2f_kernel = prepare_physics(pp, freq)
    far, _ = geoms_to_far(geoms, surrogate, incident, n2f_kernel)
    PSFs = far_to_PSFs(far, psfL, imgp.binL)
    G, _ = PSFs_to_G(PSFs, imgp.objL, imgp.imgL, nF, iF, pp.lbfreq, pp.ubfreq)

    for iO in 1:imgp.objN
        y_temp = G * Bs[iO][:,:,iF][:]
        y_temp = reshape(y_temp, (imgp.imgL,imgp.imgL))
        ys[iO] = ys[iO] + y_temp
    end
    ys
end

function test_forwardmodel(pp, imgp, Bs, surrogates, freqs, geoms, ys)
    nF = pp.orderfreq + 1
    for iF in 1:nF
        println(iF)
        flush(stdout)
        freq = freqs[iF]
        surrogate = surrogates[iF]
        ys = test_forwardmodel_perfreq(pp, imgp, Bs, surrogate, freq, iF, geoms, ys)
    end
    ys
end


function design_broadband_lens_objective_average(pp, imgp, surrogates, freqs, geoms)
    psfL = imgp.objL + imgp.imgL
    middle = div(psfL,2)
    nF = pp.orderfreqPSF + 1

    sum(1:nF) do iF
        freq = freqs[iF]
        surrogate = surrogates[iF]
        incident, n2f_kernel = ChainRulesCore.ignore_derivatives( ()-> prepare_physics(pp, freq) )
        far, _ = geoms_to_far(geoms, surrogate, incident, n2f_kernel)
        PSF = far_to_PSFs(far, psfL, imgp.binL)
        PSF[middle,middle] + PSF[middle+1,middle] + PSF[middle,middle+1] + PSF[middle+1,middle+1]
    end
end

function design_broadband_lens_objective(pp, imgp, surrogate, incident, n2f_kernel, geoms)
    psfL = imgp.objL + imgp.imgL
    middle = div(psfL,2)
    far, _ = geoms_to_far(geoms, surrogate, incident, n2f_kernel)
    PSF = far_to_PSFs(far, psfL, imgp.binL)
    PSF[middle,middle] + PSF[middle+1,middle] + PSF[middle,middle+1] + PSF[middle+1,middle+1]
end


function test_design_broadband_lens_average(pp, imgp, surrogates, freqs, geoms)
    #uniform metasurface
    geoms = fill((pp.lbwidth + pp.ubwidth)/2, 1, pp.gridL, pp.gridL)
    gradient(g -> design_broadband_lens_objective(pp, imgp, surrogates, freqs, g), geoms)
end
=#
