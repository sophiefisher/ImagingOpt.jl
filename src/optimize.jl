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


function get_params(pname, presicion, dir=PARAMS_DIR )
    if presicion == "double"
        floattype = Float64
        inttype = Int64
    elseif presicion == "single"
        floattype = Float32
        inttype = Int32
    end
    
    jsonread = JSON3.read(read("$dir/$pname.json", String))
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

function get_α(image_Tmap_flat, pp, imgp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
    Trand_flat = rand(imgp.lbT:eps(typeof(pp.lbfreq)):imgp.ubT,imgp.objL^2)
    image_diff_flat = get_image_diff_flat(Trand_flat, image_Tmap_flat, pp, imgp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
    (image_diff_flat'*image_diff_flat)/( (Trand_flat .- recp.subtract_reg )'*(Trand_flat .- recp.subtract_reg) )
end

function print_params(pp, imgp, optp, recp)
    println("######################### printing physics params #########################")
    lblambda = pp.wavcen / pp.ubfreq 
    ublambda = pp.wavcen / pp.lbfreq 
    println("wavelengths: $(round(lblambda,digits=4)) to $(round(ublambda,digits=4)) μm")
    println("chebyshev order in wavelength: $(pp.orderfreq) [$(pp.orderfreq+1) points]")
    unit_cell_length = pp.wavcen * pp.cellL
    println("unit cell length: $(round(unit_cell_length, digits=4)) μm")
    println("unit cells: $(pp.gridL) x $(pp.gridL)")
    println()
    println("######################### printing image params #########################")
    println("Tmap pixels: $(imgp.objL) x $(imgp.objL)")
    println("image pixels: $(imgp.imgL) x $(imgp.imgL)")
    println("binning: $(imgp.binL)")
    println("Tmaps to train on: $(imgp.objN)")
    println("Tmap lower bound: $(imgp.lbT) Kelvin")
    println("Tmap upper bound: $(imgp.ubT) Kelvin")
    println("differentiate noise?: $(imgp.differentiate_noise)")
    println("noise level: $(imgp.noise_level)")
    println()
    println("######################### printing optimization params #########################")
    println("initializing metasurface as: $(optp.geoms_init_type)")
    println("initializing α as: $(optp.αinit)")
    println("maximum evaluations: $(optp.maxeval)")
    println()
end

function compute_system_params(pp, imgp)
    println("######################### printing more system params #########################")
    println()
    
    image_pixel_size = imgp.binL * pp.wavcen * pp.cellL
    println("image pixel size: $(round(image_pixel_size,digits=4)) μm")
    
    object_pixel_size = image_pixel_size * pp.depth / pp.F
    println("object pixel size: $( round(object_pixel_size,digits=4)) μm")
    
    diameter = pp.gridL * pp.cellL * pp.wavcen
    println("diameter: $(round(diameter,digits=4)) μm [$( round(diameter / 1e4,digits=4)) cm]")
    
    NA = sin(atan( pp.gridL * pp.cellL / (2 * pp.F) ))
    println("NA: $( round(NA,digits=4 ))")
    
    f_number = (pp.F)/(pp.gridL * pp.cellL)
    println("f number: $(round(f_number,digits=4))")
    
    lblambda = pp.wavcen / pp.ubfreq 
    ublambda = pp.wavcen / pp.lbfreq 
    midlambda = (ublambda + lblambda)/2
    difflimmiddle = midlambda / (2*NA)
    difflimupper = (ublambda) / (2*NA)
    difflimlower = (lblambda ) / (2*NA)
    println("diffraction limit for λ = $(round(lblambda,digits=4)) μm: $(round(difflimlower,digits=4)) μm")
    println("diffraction limit for λ = $(round(midlambda,digits=4)) μm: $(round(difflimmiddle,digits=4)) μm")
    println("diffraction limit for λ = $(round(ublambda,digits=4)) μm: $(round(difflimupper,digits=4)) μm")
    println()
    
    Dict("object_pixel_size_μm" => object_pixel_size, "image_pixel_size_μm" => image_pixel_size, "NA" => NA, "diameter" => diameter, "f_number" => f_number, "diff_lim_middle_μm" => difflimmiddle, "diff_lim_lower_μm" => difflimlower, "diff_lim_upper_μm" => difflimupper)
end

#=
#need to update
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
=#

function design_singlefreq_lens(pname, presicion, parallel, opt_date)
    params = get_params(pname, presicion)
    println("######################### params loaded #########################")
    println()
    flush(stdout)
    pp = params.pp
    imgp = params.imgp
    
    opt_id = @sprintf("%s_singlefreq_lens", opt_date)
    directory = @sprintf("ImagingOpt.jl/geomsoptdata/%s", opt_id)
    Base.Filesystem.mkdir( directory )
    
    #save input parameters in json file
    jsonread = JSON3.read(read("$PARAMS_DIR/$pname.json", String))
    open("$directory/$opt_id.json", "w") do io
        JSON3.pretty(io, jsonread)
    end

    #file for saving objective vals
    file_save_objective_vals = "$directory/objective_vals_$opt_date.csv"   

    surrogates, freqs = prepare_surrogate(pp)
    middle_freq_idx = (length(freqs) + 1) ÷ 2
    surrogate = surrogates[middle_freq_idx]
    freq = freqs[middle_freq_idx]

    plan_nearfar, _ = prepare_fft_plans(pp, imgp)

    geoms_init = fill((pp.lbwidth + pp.ubwidth)/2, pp.gridL^2)

    psfL = imgp.objL + imgp.imgL
    middle = div(psfL,2)

    function objective(geoms_flat)
        geoms_grid = reshape(geoms_flat, pp.gridL, pp.gridL)
        PSF = get_PSF(freq, surrogate, pp, imgp, geoms_grid, plan_nearfar, parallel)
        obj = -1*(PSF[middle,middle] + PSF[middle+1,middle] + PSF[middle,middle+1] + PSF[middle+1,middle+1])
    
        @ignore_derivatives open(file_save_objective_vals, "a") do io
            writedlm(io, obj, ',')
        end
    
        obj
    end

    function grad!(grad::Vector, parameters::Vector)
        grad[1:end] = Zygote.gradient(objective, parameters)[1]
    end

    options = Optim.Options(f_tol=1e-8, iterations=5000, f_calls_limit=5000)
    method = Fminbox(Optim.LBFGS(m=10, linesearch=LineSearches.HagerZhang() ))
    ret_optim = Optim.optimize(objective, grad!, [pp.lbwidth for _ in 1:pp.gridL^2], [pp.ubwidth for _ in 1:pp.gridL^2], geoms_init, method, options)

    println(ret_optim)
    flush(stdout)

    mingeoms = Optim.minimizer(ret_optim)
    f_converged = Optim.f_converged(ret_optim)
    iteration_limit_reached = Optim.iteration_limit_reached(ret_optim)
    iterations = Optim.iterations(ret_optim)

    #save output data in json file
    dict_output = Dict("f_converged?" => f_converged, "iteration_limit_reached?" => iteration_limit_reached, "num_iterations" => iterations)
    output_data_filename = "$directory/output_data_$opt_date.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output)
    end

    #save optimized metasurface parameters (geoms)
    geoms_filename = "$directory/geoms_singlefreq_lens_$opt_date.csv"
    writedlm( geoms_filename,  mingeoms,',')

    #process opt
    geoms = reshape(mingeoms, pp.gridL, pp.gridL)

    #plot objective values
    objdata = readdlm(file_save_objective_vals,',')
    figure(figsize=(20,6))
    suptitle("objective data")
    subplot(1,2,1)
    plot(objdata,".-")

    subplot(1,2,2)
    semilogy(abs.(objdata),".-")
    tight_layout()
    savefig("$directory/objective_vals_$opt_date.png")

    #plot geoms
    figure(figsize=(16,5))
    subplot(1,3,1)
    imshow( reshape(geoms_init, pp.gridL, pp.gridL) , vmin = pp.lbwidth, vmax = pp.ubwidth)
    colorbar()
    title("initial metasurface \n parameters")
    subplot(1,3,2)
    imshow(geoms, vmin = pp.lbwidth, vmax = pp.ubwidth)
    colorbar()
    title("optimized metasurface \n parameters")
    subplot(1,3,3)
    imshow(geoms)
    colorbar()
    title("optimized metasurface \n parameters")
    savefig("$directory/geoms_singlefreq_lens_$opt_date.png")

    #plot PSFs (make sure there are only 21 of them)
    if parallel
        PSFs = ThreadsX.map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
    else
        PSFs = map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
    end
    figure(figsize=(20,9))
    for i = 1:21
        subplot(3,7,i)
        imshow(PSFs[i])
        colorbar()
        title("PSF for ν = $(round(freqs[i],digits=3) )")
    end
    tight_layout()
    savefig("$directory/PSFs_$opt_date.png")

    geoms
end


function compute_obj_and_grad(params_opt, params_init, freqs, surrogates, Tinit_flat, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel)
    pp = params_init.pp
    imgp = params_init.imgp
    optp = params_init.optp
    recp = params_init.recp
    
    Tmaps = prepare_objects(imgp, pp)
    noises = prepare_noises(imgp)

    if optp.optimize_alpha
        geoms = reshape(params_opt[1:end-1], pp.gridL, pp.gridL)
        α = params_opt[end] / optp.α_scaling
    else
        geoms = reshape(params_opt, pp.gridL, pp.gridL)
        α = optp.αinit
    end

    if parallel
        far_fields = ThreadsX.map(iF->get_far(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
        fftPSFs = ThreadsX.map(iF->get_fftPSF_from_far(far_fields[iF], freqs[iF], pp, imgp, plan_nearfar, plan_PSF),1:pp.orderfreq+1)
        dsur_dg_times_incidents = ThreadsX.map(iF->get_dsur_dg_times_incident(pp, freqs[iF], surrogates[iF], geoms, parallel),1:pp.orderfreq+1)

    else
        far_fields = map(iF->get_far(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
        fftPSFs = map(iF->get_fftPSF_from_far(far_fields[iF], freqs[iF], pp, imgp, plan_nearfar, plan_PSF),1:pp.orderfreq+1)
        dsur_dg_times_incidents = map(iF->get_dsur_dg_times_incident(pp, freqs[iF], surrogates[iF], geoms, parallel),1:pp.orderfreq+1)
    end

    objective = convert(typeof(freqs[1]), 0)
    if optp.optimize_alpha
        grad = zeros(typeof(freqs[1]), pp.gridL^2 + 1)
    else
        grad = zeros(typeof(freqs[1]), pp.gridL^2)
    end

    num_cg_iters_list = Matrix{Float64}(undef, 1, imgp.objN)
    ret_vals = Matrix{String}(undef, 1, imgp.objN)
    num_evals_list = Matrix{Float64}(undef, 1, imgp.objN)
    for obji = 1:imgp.objN
        Tmap = Tmaps[obji]
        noise = noises[obji]
        B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)

        image_Tmap_grid = make_image(pp, imgp, imgp.differentiate_noise, B_Tmap_grid, fftPSFs, freqs, weights, noise, noise_multiplier, plan_nearfar, plan_PSF, parallel);

        Test_flat, ret_val, num_evals = reconstruct_object(image_Tmap_grid, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel, false, false)
        ret_vals[obji] = ret_val
        num_evals_list[obji] = num_evals

        MSE = sum((Tmap[:] .- Test_flat).^2) / sum(Tmap.^2)
        objective = objective +  (1/imgp.objN) * MSE

        grad_obji, num_cg_iters = dloss_dparams(pp, imgp, optp, recp, geoms, α, Tmap, B_Tmap_grid, Test_flat, image_Tmap_grid, noise, fftPSFs, dsur_dg_times_incidents, far_fields, freqs, plan_nearfar, plan_PSF, weights, parallel)
        num_cg_iters_list[obji] = num_cg_iters
        
        grad = grad + (1/imgp.objN) * grad_obji
    end
    objective, grad, num_cg_iters_list, ret_vals, num_evals_list
end


function run_opt(pname, presicion, parallel, opt_date)
    params_init = get_params(pname, presicion)
    println("######################### params loaded #########################")
    println()
    flush(stdout)
    pp = params_init.pp
    imgp = params_init.imgp
    optp = params_init.optp
    recp = params_init.recp
    
    #check that parameters are consistent
    if imgp.noise_level == 0
        error("noise level is zero; change noise to nonzero")
    end
    if ! optp.optimize_alpha 
        error("optimize_alpha is set to false; change to true")
    end
    if optp.xtol_rel != 0
        error("xtol_rel is nonzero; change to zero")
    end
    
    #print params info
    print_params(pp, imgp, optp, recp)
    
    #prepare physics
    surrogates, freqs = prepare_surrogate(pp)
    Tinit_flat = prepare_reconstruction(recp, imgp)
    plan_nearfar, plan_PSF = prepare_fft_plans(pp, imgp)
    weights = prepare_weights(pp)
    geoms_init = prepare_geoms(params_init)
    
    if imgp.differentiate_noise
        noise_multiplier = 0
    else
        noise_multiplier = prepare_noise_multiplier(pp, imgp, surrogates, freqs, weights, plan_nearfar, plan_PSF, parallel)
    end
        
    if optp.optimize_alpha
        params_opt = [geoms_init[:]; optp.α_scaling * optp.αinit]
    else
        params_opt = geoms_init[:]
    end
    
    #prepare opt files
    lblambda = pp.wavcen / pp.ubfreq 
    ublambda = pp.wavcen / pp.lbfreq 
    unit_cell_length = pp.wavcen * pp.cellL
    opt_id = "$(round(lblambda,digits=4))_$(round(ublambda,digits=4))_$(pp.orderfreq)_$(round(unit_cell_length,digits=4))_$(pp.gridL)_$(imgp.objL)_$(imgp.imgL)_$(imgp.binL)_$(imgp.objN)_$(imgp.lbT)_$(imgp.ubT)_$(imgp.differentiate_noise)_$(imgp.noise_level)_$(optp.geoms_init_type)_$(optp.αinit)_$(optp.maxeval)"
    #opt_id = @sprintf("%s_geoms_%s_%d_%d_%d_batchsize_%d_alphainit_%.1e_maxeval_%d_diffnoise_%s", opt_date, optp.geoms_init_type, imgp.objL, imgp.imgL, pp.gridL, imgp.objN, optp.αinit, optp.maxeval, imgp.differentiate_noise)
    directory = "ImagingOpt.jl/optdata/$(opt_id)"
    Base.Filesystem.mkdir( directory )
    
    #save input parameters in json file
    jsonread = JSON3.read(read("$PARAMS_DIR/$pname.json", String))
    open("$directory/$(pname)_$(opt_date).json", "w") do io
        JSON3.pretty(io, jsonread)
    end

    #save additional system parameters in json file
    extra_params = compute_system_params(pp, imgp)
    extra_params_filename = "$directory/extra_params_$opt_date.json"
    open(extra_params_filename,"w") do io
        JSON3.pretty(io, extra_params)
    end
    
    #file for saving objective vals
    file_save_objective_vals = "$directory/objective_vals_$opt_date.csv"  
    file_save_best_objective_vals = "$directory/best_objective_vals_$opt_date.csv"  

    #file for saving alpha vals
    if optp.optimize_alpha
        file_save_alpha_vals = "$directory/alpha_vals_$opt_date.csv"   
    end

    #file for saving return values of reconstructions
    file_reconstruction_ret_vals = "$directory/reconstruction_return_vals_$opt_date.csv"

    #file for saving number of function evals for reconstructions
    file_reconstruction_num_evals = "$directory/reconstruction_num_function_evals_$opt_date.csv"

    #file for saving conjugate gradient solve iterations
    file_save_cg_iters = "$directory/cg_iters_$opt_date.csv"
    open(file_save_cg_iters, "a") do io
        write(io, "Default maxiter = $(imgp.objL^2); set to maxiter = $(optp.cg_maxiter_factor * imgp.objL^2) \n")
    end
    
    opt = Optimisers.ADAM(optp.η)
    setup = Optimisers.setup(opt, params_opt)

    params_opt_best = params_opt
    obj_best = Inf

    println("######################### beginning optimization #########################")
    println()
    for iter in 1:optp.maxeval
        @time objective, grad, num_cg_iters_list, ret_vals, num_evals_list = compute_obj_and_grad(params_opt, params_init, freqs, surrogates, Tinit_flat, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel)
    
        if objective < obj_best
            obj_best = objective
            params_opt_best[:] = params_opt
        end
        
        #save objective val
        open(file_save_objective_vals, "a") do io
            writedlm(io, objective, ',')
        end
    
        #save best objective val
        open(file_save_best_objective_vals, "a") do io
            writedlm(io, obj_best, ',')
        end

        #save best alpha val
        if optp.optimize_alpha
            α = params_opt_best[end] / optp.α_scaling
            open(file_save_alpha_vals, "a") do io
                writedlm(io, α, ',')
            end
        end
    
        #save return value of reconstruction
        open(file_reconstruction_ret_vals, "a") do io
            writedlm(io, ret_vals,',')
        end
    
        #save number of function evals of reconstruction
        open(file_reconstruction_num_evals, "a") do io
            writedlm(io, num_evals_list,',')
        end
    
        #save number of conjugate gradient solve iterations 
        open(file_save_cg_iters, "a") do io
            writedlm(io, num_cg_iters_list,',')
        end
    
        setup, params_opt = Optimisers.update(setup, params_opt, grad)
        params_opt[1:end-1] = (x -> clamp(x, pp.lbwidth, pp.ubwidth)).(params_opt[1:end-1])
        params_opt[end] = clamp(params_opt[end], 0,Inf)
    end
    println()

    mingeoms = params_opt_best[1:end-1]
    println("######################### MAX EVAL REACHED #########################")
    dict_output = Dict("return_val" => "MAXEVAL_REACHED")
    #save output data in json file
    output_data_filename = "$directory/output_data_$opt_date.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output)
    end
    
    #save optimized metasurface parameters (geoms)
    geoms_filename = "$directory/geoms_$opt_date.csv"
    writedlm( geoms_filename,  mingeoms,',')

    opt_id
end


function process_opt(presicion, parallel, opt_date, opt_id, pname)
    directory = "ImagingOpt.jl/optdata/$opt_id"
    
    params = get_params("$(pname)_$(opt_date)", presicion, directory)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    surrogates, freqs = prepare_surrogate(pp)
    Tinit_flat = prepare_reconstruction(recp, imgp)
    plan_nearfar, plan_PSF = prepare_fft_plans(pp, imgp)
    weights = prepare_weights(pp)
    iqi = SSIM(KernelFactors.gaussian(1.5, 11), (1,1,1)) #standard parameters for SSIM
    if imgp.differentiate_noise
        noise_multiplier = 0
    else
        noise_multiplier = prepare_noise_multiplier(pp, imgp, surrogates, freqs, weights, plan_nearfar, plan_PSF, parallel)
    end
    
    Tmaps_random = prepare_objects(imgp, pp) #assuming random Tmaps
    noises_random = prepare_noises(imgp)
    
    object_loadfilename_MIT = "MIT$(imgp.objL).csv"
    filename_MIT = @sprintf("ImagingOpt.jl/objdata/%s",object_loadfilename_MIT)
    lbT = imgp.lbT
    ubT = imgp.ubT
    diff = ubT - lbT
    Tmap_MIT = readdlm(filename_MIT,',',typeof(freqs[1])).* diff .+ lbT
    Tmaps_MIT = [ Tmap_MIT ]
    noises_MIT = [ imgp.noise_level .* randn(imgp.imgL, imgp.imgL); ]
    
    ################################# plot objective and alpha values throughout opt #################################
    file_save_objective_vals = "$directory/objective_vals_$opt_date.csv"
    objdata = readdlm(file_save_objective_vals,',')
    figure(figsize=(22,10))
    suptitle(L"\mathrm{objective \  data } ,  \langle \frac{|| T - T_{est} ||^2}{  || T ||^2} \rangle_{T}")
    subplot(2,2,1)
    plot(objdata,".-")

    subplot(2,2,2)
    semilogy(objdata,".-")

    file_save_best_objective_vals = "$directory/best_objective_vals_$opt_date.csv"
    objdata_best = readdlm(file_save_best_objective_vals,',')
    subplot(2,2,3)
    plot(objdata_best,".-",color="orange")

    subplot(2,2,4)
    semilogy(objdata_best,".-",color="orange")
    tight_layout()
    savefig("$directory/objective_vals_$opt_date.png")
    
    if optp.optimize_alpha
        file_save_alpha_vals = "$directory/alpha_vals_$opt_date.csv"   
        alpha_vals = readdlm(file_save_alpha_vals,',')
        
        figure(figsize=(20,6))
        suptitle(L"\alpha \ \mathrm{values }")
        subplot(1,2,1)
        plot(alpha_vals,".-")

        subplot(1,2,2)
        semilogy(alpha_vals,".-")
        tight_layout()
        savefig("$directory/alpha_vals_$opt_date.png")
    end
    
    ################################# INITIAL geoms, alpha, and PSFs #################################
    #TO DO: if starting from random metasurface, save and then reload geoms here
    geoms_init = prepare_geoms(params)
    if parallel
        fftPSFs_init = ThreadsX.map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    else
        fftPSFs_init = map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    end
    α_init = optp.αinit
    
    #plot initial PSFs
    plot_PSFs(opt_date, directory, params, freqs, surrogates, plan_nearfar, geoms_init, parallel, "initial")
    
    #save initial random reconstruction
    plot_reconstruction(opt_date, directory, params, freqs, Tinit_flat, Tmaps_random, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_init, α_init, parallel, iqi, "initial", "random")
    
    #save initial MIT reconstruction
    plot_reconstruction(opt_date, directory, params, freqs, Tinit_flat, Tmaps_MIT, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_init, α_init, parallel, iqi, "initial", "MIT")
    
    ################################# OPTIMIZED geoms, alpha, and PSFs #################################
    geoms_filename = "$directory/geoms_$opt_date.csv"
    geoms_optimized = reshape(readdlm(geoms_filename,','),pp.gridL, pp.gridL )
    if parallel
        fftPSFs_optimized = ThreadsX.map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_optimized, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    else
        fftPSFs_optimized = map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_optimized, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    end
    if optp.optimize_alpha
        α_optimized = alpha_vals[end]
    else
        α_optimized = optp.αinit
    end
    
    #plot optimized PSFs
    plot_PSFs(opt_date, directory, params, freqs, surrogates, plan_nearfar, geoms_optimized, parallel, "optimized")
    
    #save optimized random reconstruction
    plot_reconstruction(opt_date, directory, params, freqs, Tinit_flat, Tmaps_random, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "random")
    
    #save optimized MIT reconstruction
    plot_reconstruction(opt_date, directory, params, freqs, Tinit_flat, Tmaps_MIT, noises_random, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs_optimized, α_optimized, parallel, iqi, "optimized", "MIT")
    
    #reconstructions for different noise levels, random and MIT Tmaps
    plot_reconstruction_fixed_noise_levels(opt_date, directory, params, freqs, Tinit_flat, Tmaps_random[1], [0.01; 0.02; 0.05; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "random")
    
    plot_reconstruction_fixed_noise_levels(opt_date, directory, params, freqs, Tinit_flat, Tmap_MIT, [0.01; 0.02; 0.05; 0.10], plan_nearfar, plan_PSF, weights, fftPSFs_optimized, α_optimized, parallel, iqi, "MIT")
    
    ################################# plot geoms init and optimized side by side #################################
    figure(figsize=(16,5))
    subplot(1,3,1)
    imshow( geoms_init , vmin = pp.lbwidth, vmax = pp.ubwidth)
    colorbar()
    title("initial metasurface \n parameters")
    subplot(1,3,2)
    imshow(geoms_optimized, vmin = pp.lbwidth, vmax = pp.ubwidth)
    colorbar()
    title("optimized metasurface \n parameters")
    subplot(1,3,3)
    imshow(geoms_optimized)
    colorbar()
    title("optimized metasurface \n parameters")
    savefig("$directory/geoms_$opt_date.png")
    
    ################################# get transmissions for initial and optimized metasurface #################################
    get_fftPSF_freespace_iF = iF->get_fftPSF_freespace(freqs[iF], surrogates[iF], pp, imgp, plan_nearfar, plan_PSF)
    if parallel
        fftPSFs_freespace = ThreadsX.map(get_fftPSF_freespace_iF,1:pp.orderfreq+1)
    else
        fftPSFs_freespace = map(get_fftPSF_freespace_iF,1:pp.orderfreq+1)
    end
    
    transmission_initial_random = get_transmission(pp, imgp, Tmaps_random[1], fftPSFs_freespace, fftPSFs_init, freqs, weights,  plan_nearfar, plan_PSF, parallel)
    transmission_initial_MIT = get_transmission(pp, imgp, Tmap_MIT, fftPSFs_freespace, fftPSFs_init, freqs, weights, plan_nearfar, plan_PSF, parallel)
    transmission_optimized_random = get_transmission(pp, imgp, Tmaps_random[1], fftPSFs_freespace, fftPSFs_optimized, freqs, weights, plan_nearfar, plan_PSF, parallel)
    transmission_optimized_MIT = get_transmission(pp, imgp, Tmap_MIT, fftPSFs_freespace, fftPSFs_optimized, freqs, weights, plan_nearfar, plan_PSF, parallel)
    
    dict_output = Dict("transmission_initial_random" => transmission_initial_random, "transmission_initial_MIT" => transmission_initial_MIT, "transmission_optimized_random" => transmission_optimized_random, "transmission_optimized_MIT" => transmission_optimized_MIT)
    #save output data in json file
    output_data_filename = "$(directory)/transmission.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output)
    end
    
    ################################# figure plots #################################
    figure_plots_intial_directory = "ImagingOpt.jl/optdata/$opt_id/figure_plots_initial"
    if ! isdir(figure_plots_intial_directory)
        Base.Filesystem.mkdir( figure_plots_intial_directory )
    end
    plot_geoms_figure(geoms_init, figure_plots_intial_directory)
    plot_PSFs_figure(pp, imgp, freqs, surrogates, geoms_init, plan_nearfar, parallel, figure_plots_intial_directory)
    
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmap_MIT, α_init, fftPSFs_init, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_intial_directory, "MIT")
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmaps_random[1], α_init, fftPSFs_init, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_intial_directory, "random")
    
    
    figure_plots_optimized_directory = "ImagingOpt.jl/optdata/$opt_id/figure_plots_optimized"
    if ! isdir(figure_plots_optimized_directory)
        Base.Filesystem.mkdir( figure_plots_optimized_directory )
    end
    plot_geoms_figure(geoms_optimized, figure_plots_optimized_directory)
    plot_PSFs_figure(pp, imgp, freqs, surrogates, geoms_optimized, plan_nearfar, parallel, figure_plots_optimized_directory)
    
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmap_MIT, α_optimized, fftPSFs_optimized, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_optimized_directory, "MIT")
    plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmaps_random[1], α_optimized, fftPSFs_optimized, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, figure_plots_optimized_directory, "random")
    
end

function plot_reconstruction(opt_date, directory, params, freqs, Tinit_flat, Tmaps, noises, plan_nearfar, plan_PSF, weights, noise_multiplier, fftPSFs, α, parallel, iqi, geoms_type, Tmap_type)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp

    num_Tmaps = length(Tmaps)
    
    figure(figsize=(18,3.5*num_Tmaps))
    suptitle("$(geoms_type) reconstruction")
    for obji = 1:num_Tmaps
        Tmap = Tmaps[obji]
        noise = noises[obji]
        B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)

        image_Tmap_grid = make_image(pp, imgp, imgp.differentiate_noise, B_Tmap_grid, fftPSFs, freqs, weights, noise, noise_multiplier, plan_nearfar, plan_PSF, parallel);
        Test_flat, _ = reconstruct_object(image_Tmap_grid, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel)
        Test = reshape(Test_flat, imgp.objL, imgp.objL)
        
        subplot(num_Tmaps, 5, obji*5 - 4)
        imshow(Tmap, vmin = imgp.lbT, vmax = imgp.ubT)
        colorbar()
        title(L"T(x,y) \ %$obji")
    
        subplot(num_Tmaps, 5, obji*5 - 3)
        imshow(Test, vmin = imgp.lbT, vmax = imgp.ubT)
        colorbar()
        title(L"T_{est}(x,y) \ %$obji")
    
        subplot(num_Tmaps, 5, obji*5 - 2 )
        imshow( (Test .- Tmap)./Tmap .* 100)
        colorbar()
        MSE = sum((Tmap .- Test).^2) / sum(Tmap.^2)
        title("% difference \n MSE = $(round(MSE, digits=6))")
        
        ssim_map = ImageQualityIndexes._ssim_map(iqi, Tmap, Test )
        subplot(num_Tmaps, 5, obji*5 - 1  )
        imshow( ssim_map )
        colorbar()
        title("SSIM \n mean = $( round(mean(ssim_map),digits=6)  )")
        
        subplot(num_Tmaps, 5, obji*5)
        imshow(image_Tmap_grid)
        colorbar()
        # calculate new noise level
        if imgp.differentiate_noise
            noise_level = imgp.noise_level
        else
            img_noiseless = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
            noise_level = imgp.noise_level * noise_multiplier / mean(img_noiseless)
        end
        title("image \n noise level is $( round(noise_level * 100,digits=4) )%")
    end
    tight_layout()
    savefig("$directory/$(Tmap_type)_reconstruction_$(geoms_type)_$(opt_date).png")
end

function plot_reconstruction_fixed_noise_levels(opt_date, directory, params, freqs, Tinit_flat, Tmap, noise_levels, plan_nearfar, plan_PSF, weights, fftPSFs, α, parallel, iqi, Tmap_type)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    num_noise_levels = length(noise_levels)
    
    figure(figsize=(18,3.5*num_noise_levels))
    suptitle("$(Tmap_type) reconstruction at fixed noise levels")
    for i = 1:num_noise_levels
        noise_level = noise_levels[i]
        noise = noise_level .* randn(imgp.imgL, imgp.imgL)
        image_Tmap_grid = make_image(pp, imgp, true, B_Tmap_grid, fftPSFs, freqs, weights, noise, 0, plan_nearfar, plan_PSF, parallel);
        Test_flat, _ = reconstruct_object(image_Tmap_grid, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel)
        Test = reshape(Test_flat, imgp.objL, imgp.objL)
        
        subplot(num_noise_levels, 5, i*5 - 4)
        imshow(Tmap, vmin = imgp.lbT, vmax = imgp.ubT)
        colorbar()
        title(L"T(x,y)")
    
        subplot(num_noise_levels, 5, i*5 - 3)
        imshow(Test, vmin = imgp.lbT, vmax = imgp.ubT)
        colorbar()
        title(L"T_{est}(x,y)")
    
        subplot(num_noise_levels, 5, i*5 - 2 )
        imshow( (Test .- Tmap)./Tmap .* 100)
        colorbar()
        MSE = sum((Tmap .- Test).^2) / sum(Tmap.^2)
        title("% difference \n MSE = $(round(MSE, digits=6))")
        
        ssim_map = ImageQualityIndexes._ssim_map(iqi, Tmap, Test )
        subplot(num_noise_levels, 5, i*5 - 1  )
        imshow( ssim_map )
        colorbar()
        title("SSIM \n mean = $( round(mean(ssim_map),digits=6)  )")
        
        subplot(num_noise_levels, 5, i*5)
        imshow(image_Tmap_grid)
        colorbar()
        title("image \n noise level is $(noise_level * 100)%")
    end
    tight_layout()
    savefig("$directory/$(Tmap_type)_reconstruction_optimized_fixed_noises_$(opt_date).png")
end

function plot_PSFs(opt_date, directory, params, freqs, surrogates, plan_nearfar, geoms, parallel, geoms_type)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    
    if parallel
        PSFs = ThreadsX.map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
    else
        PSFs = map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
    end
    
    figure(figsize=(20,9))
    #assumes 21 PSFs
    for i = 1:21
        subplot(3,7,i)
        imshow(PSFs[i])
        colorbar()
        title("PSF for ν = $(round(freqs[i],digits=3) )")
    end
    tight_layout()
    savefig("$directory/PSFs_$(geoms_type)_$opt_date.png")
end

function get_transmission(pp, imgp, Tmap, fftPSFs_freespace, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    image_Tmap_grid_freespace = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs_freespace, freqs, weights, plan_nearfar, plan_PSF, parallel)
    image_Tmap_grid_nonoise = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
    transmission = round(sum(image_Tmap_grid_nonoise) / sum(image_Tmap_grid_freespace) * 100,digits=2)
end

function plot_geoms_figure(geoms, directory)
    figure(figsize=(5,5))
    imshow(geoms, cmap="Greys_r")
    axis("off")
    savefig("$(directory)/geoms.png")
end

function plot_PSFs_figure(pp, imgp, freqs, surrogates, geoms, plan_nearfar, parallel, directory)
    #plot PSFs (make sure there are only 21 of them)
    if parallel
        PSFs = ThreadsX.map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
    else
        PSFs = map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
    end
    
    figure(figsize=(12,6))
    for i = 1:21
        subplot(3,7,i)
        imshow(PSFs[i])
        wavelength = pp.wavcen ./ freqs[i]
        title("λ = $(round(wavelength,digits=2) ) μm",fontsize=15)
        axis("off")
    end
    tight_layout()
    savefig("$(directory)/PSFs.png")
end

function plot_Tmap_figure(imgp, Tmap, directory, Tmap_type)
    figure(figsize=(4,4))
    imshow(Tmap, vmin = imgp.lbT, vmax = imgp.ubT, cmap="Reds")
    clb = colorbar()
    axis("off")
    clb.set_label("Temperature (°K)",fontsize=15)
    clb.ax.tick_params(labelsize=15)
    savefig("$(directory)/$(Tmap_type)_Tmap.png")
end

function plot_image_figure(pp, imgp, Tmap, fftPSFs, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, directory, Tmap_type, noise_model_type)
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    
    if noise_model_type == "default"
        noise = imgp.noise_level .* randn(imgp.imgL, imgp.imgL);
        image_Tmap_grid = make_image(pp, imgp, imgp.differentiate_noise, B_Tmap_grid, fftPSFs, freqs, weights, noise, noise_multiplier, plan_nearfar, plan_PSF, parallel);
        if imgp.differentiate_noise
            noise_level = imgp.noise_level * 100
        else
            img_noiseless = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
            noise_level = imgp.noise_level * noise_multiplier / mean(img_noiseless) * 100
        end
    #otherwise, noise_model_type is equal to noise_level 
        noise_model_type = "$(noise_model_type)_$(round(noise_level ,digits=4))"
    else
        noise_level = parse(Float64,noise_model_type) / 100
        noise = noise_level .* randn(imgp.imgL, imgp.imgL);
        image_Tmap_grid = make_image(pp, imgp, true, B_Tmap_grid, fftPSFs, freqs, weights, noise, 0, plan_nearfar, plan_PSF, parallel)
    end
    
    figure(figsize=(4,4))
    imshow(image_Tmap_grid)
    #colorbar()
    axis("off")
    savefig("$(directory)/$(Tmap_type)_image_$(noise_model_type).png")
    image_Tmap_grid, noise_model_type
end

function plot_reconstruction_figure(pp, imgp, optp, recp, image_Tmap_grid, Tinit_flat, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, parallel, directory, Tmap_type, noise_model_type)
    Test_flat, _ = reconstruct_object(image_Tmap_grid, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel)
    Test = reshape(Test_flat, imgp.objL, imgp.objL)
    figure(figsize=(4,4))
    imshow(Test, vmin = imgp.lbT, vmax = imgp.ubT, cmap="Reds")
    clb = colorbar()
    axis("off")
    clb.set_label("Temperature (°K)",fontsize=15)
    clb.ax.tick_params(labelsize=15)
    savefig("$(directory)/$(Tmap_type)_Tmap_reconstructed_$(noise_model_type).png")
    Test
end

function plot_percent_error_figure(Test, Tmap, directory, Tmap_type, noise_model_type)
    figure(figsize=(4,4))
    pd = (Test .- Tmap)./Tmap .* 100
    imshow( pd, cmap = "Greys")
    clb = colorbar()
    title("|% error| < $( round(maximum(abs.(pd)),digits=2) )",fontsize=15)
    clb.ax.tick_params(labelsize=15)
    axis("off")
    savefig("$(directory)/$(Tmap_type)_Tmap_percent_error_$(noise_model_type).png")
end

function plot_Tmap_image_reconstruction_figures(pp, imgp, optp, recp, Tmap, α, fftPSFs, Tinit_flat, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, directory, Tmap_type)
    plot_Tmap_figure(imgp, Tmap, directory, Tmap_type)
    image_Tmap_grid, noise_model_type = plot_image_figure(pp, imgp, Tmap, fftPSFs, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, directory, Tmap_type, "default")
    Test = plot_reconstruction_figure(pp, imgp, optp, recp, image_Tmap_grid, Tinit_flat, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, parallel, directory, Tmap_type, noise_model_type)
    plot_percent_error_figure(Test, Tmap, directory, Tmap_type, noise_model_type)
    
    noise_levels = ["1.0", "2.0", "5.0", "10.0"]
    for noise_level in noise_levels
        plot_Tmap_figure(imgp, Tmap, directory, Tmap_type)
        image_Tmap_grid, noise_model_type = plot_image_figure(pp, imgp, Tmap, fftPSFs, freqs, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel, directory, Tmap_type, noise_level)
        Test = plot_reconstruction_figure(pp, imgp, optp, recp, image_Tmap_grid, Tinit_flat, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, parallel, directory, Tmap_type, noise_model_type)
        plot_percent_error_figure(Test, Tmap, directory, Tmap_type, noise_model_type)
    end
end