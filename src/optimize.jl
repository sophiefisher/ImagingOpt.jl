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

function compute_system_params(pp, imgp)
    image_pixel_size = imgp.binL * pp.wavcen * pp.cellL
    println("image pixel size is $image_pixel_size μm")
    object_pixel_size = image_pixel_size * pp.depth / pp.F
    println("object pixel size is $object_pixel_size μm")
    NA = sin(atan( pp.gridL * pp.cellL / (2 * pp.F) ))
    println("the NA is $NA")
    lblambda = pp.wavcen / pp.ubfreq 
    ublambda = pp.wavcen / pp.lbfreq 
    midlambda = (ublambda + lblambda)/2
    difflimmiddle = midlambda / (2*NA)
    difflimupper = (ublambda) / (2*NA)
    difflimlower = (lblambda ) / (2*NA)
    println("the diffraction limit for λ = $midlambda μm is $difflimmiddle μm ($difflimlower μm to $difflimupper for full bandwidth)")
    Dict("object_pixel_size_μm" => object_pixel_size, "image_pixel_size_μm" => image_pixel_size, "NA" => NA, "diff_lim_middle_μm" => difflimmiddle, "diff_lim_lower_μm" => difflimlower, "diff_lim_upper_μm" => difflimupper)
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
    println("params loaded")
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

    options = Optim.Options(f_tol=1e-8, iterations=5000)
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
end


function compute_obj_and_grad(params_opt, params_init, freqs, surrogates, Tinit_flat, weights, plan_nearfar, plan_PSF, parallel)
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
    for obji = 1:imgp.objN
        Tmap = Tmaps[obji]
        noise = noises[obji]
        B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)

        image_Tmap_grid = make_image(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, noise, plan_nearfar, plan_PSF, parallel);
        #TO DO: save reconstruction data in csv (save data from one optimization iter in one row): return value and # of iterations
        Test_flat = reconstruct_object(image_Tmap_grid, Tmap, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel)

        MSE = sum((Tmap[:] .- Test_flat).^2) / sum(Tmap.^2)
        objective = objective +  (1/imgp.objN) * MSE

        grad_obji, num_cg_iters = dloss_dparams(pp, imgp, optp, recp, geoms, α, Tmap, B_Tmap_grid, Test_flat, image_Tmap_grid, noise, fftPSFs, dsur_dg_times_incidents, far_fields, freqs, plan_nearfar, plan_PSF, weights, parallel)
        num_cg_iters_list[obji] = num_cg_iters
        
        grad = grad + (1/imgp.objN) * grad_obji
    end
    objective, grad, num_cg_iters_list
end


function run_opt(pname, presicion, parallel, opt_date)
    params_init = get_params(pname, presicion)
    println("params loaded")
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
    
    #prepare physics
    surrogates, freqs = prepare_surrogate(pp)
    Tinit_flat = prepare_reconstruction(recp, imgp)
    plan_nearfar, plan_PSF = prepare_fft_plans(pp, imgp)
    weights = convert.( typeof(freqs[1]), ClenshawCurtisQuadrature(pp.orderfreq + 1).weights)
    geoms_init = prepare_geoms(params_init)
    
    if optp.optimize_alpha
        params_opt = [geoms_init[:]; optp.α_scaling * optp.αinit]
    else
        params_opt = geoms_init[:]
    end
    
    #prepare opt files
    opt_id = @sprintf("%s_geoms_%s_%d_%d_%d_alphainit_%.1e_maxeval_%d_xtolrel_%.1e", opt_date, optp.geoms_init_type, imgp.objL, imgp.imgL, pp.gridL, optp.αinit, optp.maxeval, optp.xtol_rel)
    directory = @sprintf("ImagingOpt.jl/optdata/%s", opt_id)
    Base.Filesystem.mkdir( directory )
    
    #save input parameters in json file
    jsonread = JSON3.read(read("$PARAMS_DIR/$pname.json", String))
    open("$directory/$(pname)_$(opt_date).json", "w") do io
        JSON3.pretty(io, jsonread)
    end
    #TO DO: add total diameter of lens; fstop 
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

    #file for saving conjugate gradient solve iterations
    file_save_cg_iters = "$directory/cg_iters_$opt_date.csv"
    open(file_save_cg_iters, "a") do io
        write(io, "Default maxiter = $(imgp.objL^2); set to maxiter = $(optp.cg_maxiter_factor * imgp.objL^2) \n")
    end
    
    opt = Optimisers.ADAM(optp.η)
    setup = Optimisers.setup(opt, params_opt)

    params_opt_best = params_opt
    obj_best = Inf

    for iter in 1:optp.maxeval
        @time objective, grad, num_cg_iters_list = compute_obj_and_grad(params_opt, params_init, freqs, surrogates, Tinit_flat, weights, plan_nearfar, plan_PSF, parallel)
    
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
    
        #save number of conjugate gradient solve iterations 
        open(file_save_cg_iters, "a") do io
            writedlm(io, num_cg_iters_list,',')
        end
    
        setup, params_opt = Optimisers.update(setup, params_opt, grad)
        params_opt[1:end-1] = (x -> clamp(x, pp.lbwidth, pp.ubwidth)).(params_opt[1:end-1])
        params_opt[end] = clamp(params_opt[end], 0,Inf)
    end

    mingeoms = params_opt_best[1:end-1]
    println("MAX EVAL REACHED")
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
    
    Tmaps = prepare_objects(imgp, pp)
    noises = prepare_noises(imgp)
    
    plan_nearfar, plan_PSF = prepare_fft_plans(pp, imgp)
    weights = convert.( typeof(freqs[1]), ClenshawCurtisQuadrature(pp.orderfreq + 1).weights)
    
    #TO DO: if starting from random metasurface, save and then reload geoms here
    geoms_init = prepare_geoms(params)
    
    #save initial reconstruction
    iqi = SSIM(KernelFactors.gaussian(1.5, 11), (1,1,1)) #standard parameters for SSIM
    if parallel
        fftPSFs = ThreadsX.map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    else
        fftPSFs = map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    end
    
    figure(figsize=(16,10))
    suptitle("initial reconstruction")
    for obji = 1:imgp.objN
        Tmap = Tmaps[obji]
        noise = noises[obji]
        B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)

        image_Tmap_grid = make_image(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, noise, plan_nearfar, plan_PSF, parallel);
        Test = reshape(reconstruct_object(image_Tmap_grid, Tmap, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, optp.αinit, false, false, parallel), imgp.objL, imgp.objL)
        
        subplot(imgp.objN, 4, obji*4 - 3)
        imshow(Tmap, vmin = imgp.lbT, vmax = imgp.ubT)
        colorbar()
        title(L"T(x,y) \ %$obji")
    
        subplot(imgp.objN, 4, obji*4 - 2)
        imshow(Test, vmin = imgp.lbT, vmax = imgp.ubT)
        colorbar()
        title(L"T_{est}(x,y) \ %$obji")
    
        subplot(imgp.objN, 4, obji*4 - 1 )
        imshow( (Test .- Tmap)./Tmap .* 100)
        colorbar()
        MSE = sum((Tmap .- Test).^2) / sum(Tmap.^2)
        title("% difference $obji.\n MSE = $MSE")
        
        ssim_map = ImageQualityIndexes._ssim_map(iqi, Tmap, Test )
        subplot(imgp.objN, 4, obji*4  )
        imshow( ssim_map )
        colorbar()
        title("SSIM.\n mean = $(mean(ssim_map))")
    end
    tight_layout()
    savefig("$directory/reconstruction_initial_$opt_date.png")
    
    #plot PSFs (make sure there are only 21 of them)
    if parallel
        PSFs = ThreadsX.map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, parallel),1:pp.orderfreq+1)
    else
        PSFs = map(iF->get_PSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, parallel),1:pp.orderfreq+1)
    end
    figure(figsize=(20,9))
    for i = 1:21
        subplot(3,7,i)
        imshow(PSFs[i])
        colorbar()
        title("PSF for ν = $(round(freqs[i],digits=3) )")
    end
    tight_layout()
    savefig("$directory/PSFs_initial_$opt_date.png")
    
    #MIT initial reconstruction
    fig_MIT, ax_MIT = subplots(2,5,figsize=(18,8))
    object_loadfilename_MIT = "MIT$(imgp.objL).csv"
    filename_MIT = @sprintf("ImagingOpt.jl/objdata/%s",object_loadfilename_MIT)
    lbT = imgp.lbT
    ubT = imgp.ubT
    diff = ubT - lbT
    Tmap_MIT = readdlm(filename_MIT,',',typeof(freqs[1])).* diff .+ lbT

    B_Tmap_grid_MIT = prepare_blackbody(Tmap_MIT, freqs, imgp, pp)
    image_Tmap_grid_MIT = make_image(pp, imgp, B_Tmap_grid_MIT, fftPSFs, freqs, weights, imgp.noise_level .* randn(imgp.imgL, imgp.imgL), plan_nearfar, plan_PSF, parallel);
    Test_MIT = reshape(reconstruct_object(image_Tmap_grid_MIT, Tmap_MIT, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, optp.αinit, false, false, parallel), imgp.objL, imgp.objL)

    p1 = ax_MIT[1,1].imshow(Tmap_MIT, vmin = imgp.lbT, vmax = imgp.ubT)
    fig_MIT.colorbar(p1, ax=ax_MIT[1,1])
    ax_MIT[1,1].set_title(L"T(x,y) \  \mathrm{initial}")

    p2 = ax_MIT[1,2].imshow(Test_MIT, vmin = imgp.lbT, vmax = imgp.ubT)
    fig_MIT.colorbar(p2, ax=ax_MIT[1,2])
    ax_MIT[1,2].set_title(L"T_{est}(x,y) \  \mathrm{initial}")
    
    p3 = ax_MIT[1,3].imshow( (Test_MIT .- Tmap_MIT)./Tmap_MIT .* 100)
    fig_MIT.colorbar(p3, ax=ax_MIT[1,3])
    MSE = sum((Tmap_MIT .- Test_MIT).^2) / sum(Tmap_MIT.^2)
    ax_MIT[1,3].set_title("% difference initial.\n MSE = $MSE")

    ssim_map = ImageQualityIndexes._ssim_map(iqi, Tmap_MIT, Test_MIT )
    p4 = ax_MIT[1,4].imshow(ssim_map)
    fig_MIT.colorbar(p4, ax=ax_MIT[1,4])
    ax_MIT[1,4].set_title("SSIM initial.\n mean = $(mean(ssim_map))")

    p5 = ax_MIT[1,5].imshow(image_Tmap_grid_MIT)
    fig_MIT.colorbar(p5, ax=ax_MIT[1,5])
    ax_MIT[1,5].set_title("image initial")
    
    #now use optimized geoms and optimized alpha
    if optp.optimize_alpha
        file_save_alpha_vals = "$directory/alpha_vals_$opt_date.csv"   
        alpha_vals = readdlm(file_save_alpha_vals,',')
        α = alpha_vals[end]
    else
        α = optp.αinit
    end
    
    geoms_filename = "$directory/geoms_$opt_date.csv"
    geoms = reshape(readdlm(geoms_filename,','),pp.gridL, pp.gridL )
    
    figure(figsize=(16,5))
    subplot(1,3,1)
    imshow( geoms_init , vmin = pp.lbwidth, vmax = pp.ubwidth)
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
    savefig("$directory/geoms_$opt_date.png")


    #save optimized reconstructions and images
    if parallel
        fftPSFs = ThreadsX.map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    else
        fftPSFs = map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    end
    
    fig1, ax1 = subplots(imgp.objN,4,figsize=(16,10))
    fig1.suptitle("optimized reconstruction")

    fig2, ax2 = subplots(imgp.objN,2,figsize=(8,10))
    fig2.suptitle("optimized images")
    for obji = 1:imgp.objN
        Tmap = Tmaps[obji]
        noise = noises[obji]
        B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)

        image_Tmap_grid = make_image(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, noise, plan_nearfar, plan_PSF, parallel);
        Test = reshape(reconstruct_object(image_Tmap_grid, Tmap, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel), imgp.objL, imgp.objL)
        
        p1 = ax1[obji,1].imshow(Tmap, vmin = imgp.lbT, vmax = imgp.ubT)
        fig1.colorbar(p1, ax=ax1[obji,1])
        ax1[obji,1].set_title(L"T(x,y) \ %$obji")
    
        p2 = ax1[obji,2].imshow(Test, vmin = imgp.lbT, vmax = imgp.ubT)
        fig1.colorbar(p2, ax=ax1[obji,2])
        ax1[obji,2].set_title(L"T_{est}(x,y) \ %$obji")
    
        p3 = ax1[ obji,3 ].imshow( (Test .- Tmap)./Tmap .* 100 )
        fig1.colorbar(p3, ax = ax1[ obji,3 ])
        MSE = sum((Tmap .- Test).^2) / sum(Tmap.^2)
        ax1[obji,3].set_title("% difference $obji.\n MSE = $MSE")
    
        ssim_map = ImageQualityIndexes._ssim_map(iqi, Tmap, Test )
        p4 = ax1[obji,4].imshow( ssim_map )
        fig1.colorbar(p4, ax = ax1[ obji,4 ])
        ax1[obji,4].set_title("SSIM.\n mean = $(mean(ssim_map))")
    
        p5 = ax2[obji,1].imshow(Tmap, vmin = imgp.lbT, vmax = imgp.ubT)
        fig2.colorbar(p5, ax = ax2[obji,1])
        ax2[obji,1].set_title(L"T(x,y) \ %$obji")
    
        p6 = ax2[obji,2].imshow(image_Tmap_grid)
        fig2.colorbar(p6, ax = ax2[obji,2])
        ax2[obji,2].set_title("image $obji" )
    
    end
    fig1.tight_layout()
    fig1.savefig("$directory/reconstruction_optimized_$opt_date.png")

    fig2.tight_layout()
    fig2.savefig("$directory/images_optimized_$opt_date.png")


    #plot objective values
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
    savefig("$directory/PSFs_optimized_$opt_date.png")

    #now try reconstruction on more readable image    
    image_Tmap_grid_MIT = make_image(pp, imgp, B_Tmap_grid_MIT, fftPSFs, freqs, weights, imgp.noise_level .* randn(imgp.imgL, imgp.imgL), plan_nearfar, plan_PSF, parallel);
    Test_MIT = reshape(reconstruct_object(image_Tmap_grid_MIT, Tmap_MIT, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel), imgp.objL, imgp.objL)

    p1 = ax_MIT[2,1].imshow(Tmap_MIT, vmin = imgp.lbT, vmax = imgp.ubT)
    fig_MIT.colorbar(p1, ax=ax_MIT[2,1])
    ax_MIT[2,1].set_title(L"T(x,y) \  \mathrm{optimized}")

    p2 = ax_MIT[2,2].imshow(Test_MIT, vmin = imgp.lbT, vmax = imgp.ubT)
    fig_MIT.colorbar(p2, ax=ax_MIT[2,2])
    ax_MIT[2,2].set_title(L"T_{est}(x,y) \  \mathrm{optimized}")
    
    p3 = ax_MIT[2,3].imshow( (Test_MIT .- Tmap_MIT)./Tmap_MIT .* 100)
    fig_MIT.colorbar(p3, ax=ax_MIT[2,3])
    MSE = sum((Tmap_MIT .- Test_MIT).^2) / sum(Tmap_MIT.^2)
    ax_MIT[2,3].set_title("% difference optimized.\n MSE = $MSE")

    ssim_map = ImageQualityIndexes._ssim_map(iqi, Tmap_MIT, Test_MIT )
    p4 = ax_MIT[2,4].imshow(ssim_map)
    fig_MIT.colorbar(p4, ax=ax_MIT[2,4])
    ax_MIT[2,4].set_title("SSIM optimized.\n mean = $(mean(ssim_map))")

    p5 = ax_MIT[2,5].imshow(image_Tmap_grid_MIT)
    fig_MIT.colorbar(p5, ax=ax_MIT[2,5])
    ax_MIT[2,5].set_title("image optimized")

    fig_MIT.tight_layout()
    fig_MIT.savefig("$directory/MIT_reconstruction_$opt_date.png")

    #save alpha vals
    if optp.optimize_alpha
        figure(figsize=(20,6))
        suptitle(L"\alpha \mathrm{values }")
        subplot(1,2,1)
        plot(alpha_vals,".-")

        subplot(1,2,2)
        semilogy(alpha_vals,".-")
        tight_layout()
        savefig("$directory/alpha_vals_$opt_date.png")
    end

end