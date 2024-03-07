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

function print_params(pp, imgp, optp, recp, print_pp::Bool=true, print_imgp::Bool=true, print_optp::Bool=true, print_recp::Bool=true)
    if print_pp
        println("######################### printing physics params #########################")
        lblambda = pp.wavcen / pp.ubfreq 
        ublambda = pp.wavcen / pp.lbfreq 
        println("wavelengths: $(round(lblambda,digits=4)) to $(round(ublambda,digits=4)) μm")
        println("chebyshev order in wavelength: $(pp.orderfreq) [$(pp.orderfreq+1) points]")
        unit_cell_length = pp.wavcen * pp.cellL
        println("unit cell length: $(round(unit_cell_length, digits=4)) μm")
        println("unit cells: $(pp.gridL) x $(pp.gridL)")
        println("chebyshev order in width: $(pp.orderwidth)")
        println()
    end
    
    if print_imgp
        println("######################### printing image params #########################")
        println("Tmap pixels: $(imgp.objL) x $(imgp.objL)")
        println("image pixels: $(imgp.imgL) x $(imgp.imgL)")
        println("binning: $(imgp.binL)")
        println("Tmaps to train on: $(imgp.objN)")
        println("Tmap lower bound: $(imgp.lbT) Kelvin [$(imgp.lbT - 273.15) C, $( (imgp.lbT - 273.15)*(9/5) + 32 ) F]")
        println("Tmap upper bound: $(imgp.ubT) Kelvin [$(imgp.ubT- 273.15) C, $( (imgp.ubT - 273.15)*(9/5) + 32 ) F]")
        println("differentiate noise?: $(imgp.differentiate_noise)")
        println("noise level: $(imgp.noise_level)")
        println()
    end
        
    if print_optp
        println("######################### printing optimization params #########################")
        println("initializing metasurface as: $(optp.geoms_init_type)")
        println("metasurface load type: $(optp.geoms_init_loadsavename)")
        println("initializing α as: $(optp.αinit)")
        println("maximum evaluations: $(optp.maxeval)")
        println("save every: $(optp.saveeval)")
        println("learning rate η: $(optp.η)")
        println()
    end
    
    if print_recp
        println("######################### printing reconstruction params #########################")
        println("initializing Tmap as: $(recp.T_init_type)")
        println("reconstruction objective tolerance: $(recp.ftol_rel)")
        println("mean Tmap value to subtract from regularization term: $(recp.subtract_reg)")
        println()
    end
end

function compute_system_params(pp, imgp)
    println("######################### printing more system params #########################")
    println()
    
    image_pixel_size = imgp.binL * pp.wavcen * pp.cellL
    println("image pixel size: $(round(image_pixel_size,digits=4)) μm")
    
    image_size = image_pixel_size * imgp.imgL
    println("image size: $( round(image_size,digits=4)) μm [$( round(image_size / 1e4,digits=4)) cm]")
    
    object_pixel_size = image_pixel_size * pp.depth / pp.F
    println("object pixel size: $( round(object_pixel_size,digits=4)) μm")
    
    object_size = object_pixel_size * imgp.objL
    println("object size: $( round(object_size,digits=4)) μm [$( round(object_size / 1e4,digits=4)) cm]")
    
    object_angle = 2 * atan(object_size / (2 * (pp.depth * pp.wavcen) )) * 180 / pi
    println("object angle: $(round(object_angle,digits=4)) degrees")
    
    diameter = pp.gridL * pp.cellL * pp.wavcen
    println("diameter: $(round(diameter,digits=4)) μm [$( round(diameter / 1e4,digits=4)) cm]")
    
    NA = sin(atan( pp.gridL * pp.cellL / (2 * pp.F) ))
    println("NA: $( round(NA,digits=4 ))")
    
    f_number = (pp.F)/(pp.gridL * pp.cellL)
    println("f number: $(round(f_number,digits=4))")
    
    FOV = 2 * atan(image_size / (2 * (pp.wavcen * pp.F) )) * 180 / pi
    println("FOV: $(round(FOV,digits=4)) degrees")
    
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
    
    Dict("object_pixel_size_μm" => object_pixel_size, "object_size" => object_size, "image_pixel_size_μm" => image_pixel_size, "image_size" => image_size, "NA" => NA, "diameter" => diameter, "f_number" => f_number, "diff_lim_middle_μm" => difflimmiddle, "diff_lim_lower_μm" => difflimlower, "diff_lim_upper_μm" => difflimupper, "FOV" => FOV, "object_angle" => object_angle)
end


function design_polychromatic_lens(pname, presicion, parallel, opt_date, maxeval = 1000, xtol_rel = 1e-8, ineq_tol = 1e-8)
    #assumes there are 21 PSFs!
    
    params = get_params(pname, presicion)
    pp = params.pp
    imgp = params.imgp
    
    lblambda = pp.wavcen / pp.ubfreq 
    ublambda = pp.wavcen / pp.lbfreq 
    unit_cell_length = pp.wavcen * pp.cellL
    opt_id = "$(opt_date)_polychromatic_$(round(lblambda,digits=4))_$(round(ublambda,digits=4))_$(pp.orderfreq)_$(round(unit_cell_length,digits=4))_$(pp.gridL)_$(pp.orderwidth)_$(imgp.objL)_$(imgp.imgL)_$(imgp.binL)_$(maxeval)_$(xtol_rel)_$(ineq_tol)"
    directory = @sprintf("ImagingOpt.jl/geomsoptdata/%s", opt_id)
    Base.Filesystem.mkdir( directory )
    
    #save input parameters in json file
    jsonread = JSON3.read(read("$PARAMS_DIR/$pname.json", String))
    open("$directory/$(pname)_$(opt_date).json", "w") do io
        JSON3.pretty(io, jsonread)
    end

    println("######################### params loaded #########################")
    println()
    flush(stdout)
    print_params(pp, imgp, params.optp, params.recp, true, true, false, false)
    flush(stdout)

    #file for saving objective vals
    file_save_objective_vals = "$directory/objective_vals_$opt_date.csv"  

    surrogates, freqs = prepare_surrogate(pp)
    plan_nearfar, _ = prepare_fft_plans(pp, imgp)
    geoms_init = fill((pp.lbwidth + pp.ubwidth)/2, pp.gridL^2)

    psfL = imgp.objL + imgp.imgL
    middle = div(psfL,2)
    nF = pp.orderfreq + 1
    
    middle_freq_idx = 11
    freqs_idx_1 = 1
    freq_idx_2 = 8
    freq_idx_3 = 14
    freq_idx_4 = 21
    freq_idx_list = [freqs_idx_1, freq_idx_2, freq_idx_3, freq_idx_4, middle_freq_idx]
    
    offset = Int(floor(3*div(imgp.imgL,8)))
    xoffsets = [-offset, 0, offset, 0, 0]
    yoffsets = [0, -offset, 0, offset, 0]
    
    function get_PSF_offset_center(freq, surrogate, geoms, xoffset, yoffset)
        PSF = get_PSF(freq, surrogate, pp, imgp, geoms, plan_nearfar, parallel)
        PSF[middle + yoffset, middle + xoffset] + PSF[middle+1+ yoffset,middle+ xoffset] + PSF[middle+ yoffset,middle+1+ xoffset] + PSF[middle+1+ yoffset,middle+1+ xoffset]
    end
    
    t_init = minimum(1:length(freq_idx_list)) do i
        idx = freq_idx_list[i]
        offset_center = get_PSF_offset_center(freqs[idx], surrogates[idx], reshape(geoms_init,pp.gridL,pp.gridL), xoffsets[i], yoffsets[i])
    end
    x_init = [geoms_init[:]; t_init]

    function myfunc(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1:end-1] = zeros( pp.gridL^2)
            grad[end] = 1
        end

        open(file_save_objective_vals, "a") do io
            writedlm(io, x[end], ',')
        end

        x[end]
    end

    function myconstraint(x::Vector, grad::Vector, i)
        idx = freq_idx_list[i]
        geoms_grid = reshape(x[1:end-1], pp.gridL, pp.gridL)
        constraint = g -> get_PSF_offset_center(freqs[idx], surrogates[idx], g, xoffsets[i], yoffsets[i])
        if length(grad) > 0
            grad[end] = 1
            grad_grid = -1 * Zygote.gradient( constraint, geoms_grid )[1]
            grad[1:end-1] = grad_grid[:]
        end
        x[end] - constraint(geoms_grid)
    end

    opt = Opt(:LD_MMA, pp.gridL^2 + 1)
    opt.lower_bounds = [fill(pp.lbwidth,pp.gridL^2); -Inf]
    opt.upper_bounds = [fill(pp.ubwidth,pp.gridL^2); Inf]
    opt.max_objective = myfunc
    for i = 1:length(freq_idx_list)
        inequality_constraint!(opt, (x,grad) -> myconstraint(x, grad, i), ineq_tol)
    end
    opt.xtol_rel = xtol_rel
    opt.maxeval = maxeval

    (maxf,maxx,ret) = NLopt.optimize(opt, x_init)
    mingeoms = maxx[1:end-1]

    println(ret)
    flush(stdout)

    #save output data in json file
    dict_output = Dict("return_value?" => ret, "maxeval" => maxeval, "xtol_rel" => xtol_rel, "ineq_tol" => ineq_tol)
    output_data_filename = "$directory/output_data_$opt_date.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output)
    end

    #save optimized metasurface parameters (geoms)
    geoms_filename = "$directory/geoms_$(opt_id).csv"
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
    savefig("$directory/geoms_$(opt_id).png")

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


function design_multifocal_lens(pname, presicion, parallel, opt_date, maxeval = 1000, xtol_rel = 1e-8, ineq_tol = 1e-8)
    params = get_params(pname, presicion)
    pp = params.pp
    imgp = params.imgp
    
    #prepare opt files
    lblambda = pp.wavcen / pp.ubfreq 
    ublambda = pp.wavcen / pp.lbfreq 
    unit_cell_length = pp.wavcen * pp.cellL
    opt_id = "$(opt_date)_multifocal_lens_$(round(lblambda,digits=4))_$(round(ublambda,digits=4))_$(pp.orderfreq)_$(round(unit_cell_length,digits=4))_$(pp.gridL)_$(pp.orderwidth)_$(imgp.objL)_$(imgp.imgL)_$(imgp.binL)_$(maxeval)_$(xtol_rel)_$(ineq_tol)"
    directory = @sprintf("ImagingOpt.jl/geomsoptdata/%s", opt_id)
    Base.Filesystem.mkdir( directory )
    
    #save input parameters in json file
    jsonread = JSON3.read(read("$PARAMS_DIR/$pname.json", String))
    open("$directory/$(pname)_$(opt_date).json", "w") do io
        JSON3.pretty(io, jsonread)
    end

    println("######################### params loaded #########################")
    println()
    flush(stdout)
    print_params(pp, imgp, params.optp, params.recp, true, true, false, false)
    flush(stdout)

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
    nF = pp.orderfreq + 1

    function get_PSF_offset_center(freq, surrogate, geoms, xoffset, yoffset)
        PSF = get_PSF(freq, surrogate, pp, imgp, geoms, plan_nearfar, parallel)
        PSF[middle + yoffset, middle + xoffset] + PSF[middle+1+ yoffset,middle+ xoffset] + PSF[middle+ yoffset,middle+1+ xoffset] + PSF[middle+1+ yoffset,middle+1+ xoffset]
    end

    offset = Int(floor(3*div(imgp.imgL,8)))
    xoffsets = [-offset, 0, offset, 0]
    yoffsets = [0, -offset, 0, offset]

    t_init = minimum(1:length(xoffsets)) do i
        offset_center = get_PSF_offset_center(freq, surrogate, reshape(geoms_init,pp.gridL,pp.gridL), xoffsets[i], yoffsets[i])
    end
    x_init = [geoms_init[:]; t_init]

    #=
    #if using optim
    function objective(x::Vector)
        open(file_save_objective_vals, "a") do io
            writedlm(io, x[end], ',')
        end

        x[end]
    end

    function grad!(grad::Vector, x::Vector)
        grad[1:end-1] = zeros(pp.gridL^2)
        grad[end] = 1
    end=#

    function myfunc(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1:end-1] = zeros( pp.gridL^2)
            grad[end] = 1
        end

        open(file_save_objective_vals, "a") do io
            writedlm(io, x[end], ',')
        end

        x[end]
    end

    function myconstraint(x::Vector, grad::Vector, i)
        geoms_grid = reshape(x[1:end-1], pp.gridL, pp.gridL)
        constraint = g -> get_PSF_offset_center(freq, surrogate, g, xoffsets[i], yoffsets[i])
        if length(grad) > 0
            grad[end] = 1
            grad_grid = -1 * Zygote.gradient( constraint, geoms_grid )[1]
            grad[1:end-1] = grad_grid[:]
        end
        x[end] - constraint(geoms_grid)
    end

    opt = Opt(:LD_MMA, pp.gridL^2 + 1)
    opt.lower_bounds = [fill(pp.lbwidth,pp.gridL^2); -Inf]
    opt.upper_bounds = [fill(pp.ubwidth,pp.gridL^2); Inf]
    opt.max_objective = myfunc
    for i = 1:length(xoffsets)
        inequality_constraint!(opt, (x,grad) -> myconstraint(x, grad, i), ineq_tol)
    end
    opt.xtol_rel = xtol_rel
    opt.maxeval = maxeval

    (maxf,maxx,ret) = NLopt.optimize(opt, x_init)
    mingeoms = maxx[1:end-1]

    println(ret)
    flush(stdout)

    #save output data in json file
    dict_output = Dict("return_value?" => ret, "maxeval" => maxeval, "xtol_rel" => xtol_rel, "ineq_tol" => ineq_tol)
    output_data_filename = "$directory/output_data_$opt_date.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output)
    end

    #save optimized metasurface parameters (geoms)
    geoms_filename = "$directory/geoms_$(opt_id).csv"
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
    savefig("$directory/geoms_$(opt_id).png")

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


function design_achromatic_lens(pname, presicion, parallel, opt_date, maxeval = 1000, xtol_rel = 1e-8, ineq_tol = 1e-8)
    params = get_params(pname, presicion)
    pp = params.pp
    imgp = params.imgp
    
    #prepare opt files
    lblambda = pp.wavcen / pp.ubfreq 
    ublambda = pp.wavcen / pp.lbfreq 
    unit_cell_length = pp.wavcen * pp.cellL
    opt_id = "$(opt_date)_achromatic_lens_$(round(lblambda,digits=4))_$(round(ublambda,digits=4))_$(pp.orderfreq)_$(round(unit_cell_length,digits=4))_$(pp.gridL)_$(pp.orderwidth)_$(imgp.objL)_$(imgp.imgL)_$(imgp.binL)_$(maxeval)_$(xtol_rel)_$(ineq_tol)"
    directory = @sprintf("ImagingOpt.jl/geomsoptdata/%s", opt_id)
    Base.Filesystem.mkdir( directory )
    
    #save input paoptimizerameters in json file
    jsonread = JSON3.read(read("$PARAMS_DIR/$pname.json", String))
    open("$directory/$(pname)_$(opt_date).json", "w") do io
        JSON3.pretty(io, jsonread)
    end

    println("######################### params loaded #########################")
    println()
    flush(stdout)
    print_params(pp, imgp, params.optp, params.recp, true, true, false, false)
    flush(stdout)

    #file for saving objective vals
    file_save_objective_vals = "$directory/objective_vals_$opt_date.csv"  

    surrogates, freqs = prepare_surrogate(pp)
    plan_nearfar, _ = prepare_fft_plans(pp, imgp)

    geoms_init = fill((pp.lbwidth + pp.ubwidth)/2, pp.gridL^2)

    psfL = imgp.objL + imgp.imgL
    middle = div(psfL,2)
    nF = pp.orderfreq + 1

    t_init = minimum(1:nF) do iF
        PSF = get_PSF(freqs[iF], surrogates[iF], pp, imgp, reshape(geoms_init,pp.gridL,pp.gridL), plan_nearfar, parallel)
        PSF[middle,middle] + PSF[middle+1,middle] + PSF[middle,middle+1] + PSF[middle+1,middle+1]
    end
    x_init = [geoms_init[:]; t_init]

    #=
    #if using optim
    function objective(x::Vector)
        open(file_save_objective_vals, "a") do io
            writedlm(io, x[end], ',')
        end

        x[end]
    end

    function grad!(grad::Vector, x::Vector)
        grad[1:end-1] = zeros(pp.gridL^2)
        grad[end] = 1
    end=#

    function myfunc(x::Vector, grad::Vector)
        if length(grad) > 0
            grad[1:end-1] = zeros( pp.gridL^2)
            grad[end] = 1
        end

        open(file_save_objective_vals, "a") do io
            writedlm(io, x[end], ',')
        end

        x[end]
    end

    function get_PSF_center(freq, surrogate, geoms)
        PSF = get_PSF(freq, surrogate, pp, imgp, geoms, plan_nearfar, parallel)
        PSF[middle,middle] + PSF[middle+1,middle] + PSF[middle,middle+1] + PSF[middle+1,middle+1]
    end

    function myconstraint(x::Vector, grad::Vector, iF)
        geoms_grid = reshape(x[1:end-1], pp.gridL, pp.gridL)
        freq = freqs[iF]
        surrogate = surrogates[iF]
        constraint = g -> get_PSF_center(freq, surrogate, g)
        if length(grad) > 0
            grad[end] = 1
            grad_grid = -1 * Zygote.gradient( constraint, geoms_grid )[1]
            grad[1:end-1] = grad_grid[:]
        end
        x[end] - constraint(geoms_grid)
    end

    opt = Opt(:LD_MMA, pp.gridL^2 + 1)
    opt.lower_bounds = [fill(pp.lbwidth,pp.gridL^2); -Inf]
    opt.upper_bounds = [fill(pp.ubwidth,pp.gridL^2); Inf]
    opt.max_objective = myfunc
    for iF = 1:nF
        inequality_constraint!(opt, (x,grad) -> myconstraint(x, grad, iF), ineq_tol)
    end
    opt.xtol_rel = xtol_rel
    opt.maxeval = maxeval

    (maxf,maxx,ret) = NLopt.optimize(opt, x_init)
    mingeoms = maxx[1:end-1]

    println(ret)
    flush(stdout)

    #save output data in json file
    dict_output = Dict("return_value?" => ret, "maxeval" => maxeval, "xtol_rel" => xtol_rel, "ineq_tol" => ineq_tol )
    output_data_filename = "$directory/output_data_$opt_date.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output)
    end

    #save optimized metasurface parameters (geoms)
    geoms_filename = "$directory/geoms_$(opt_id).csv"
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
    savefig("$directory/geoms_$(opt_id).png")

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



function design_singlefreq_lens_NLOPT(pname, presicion, parallel, opt_date, xtol_rel = 1e-8, maxeval = 100)
    params = get_params(pname, presicion)
    pp = params.pp
    imgp = params.imgp
    
    #prepare opt files
    lblambda = pp.wavcen / pp.ubfreq 
    ublambda = pp.wavcen / pp.lbfreq 
    unit_cell_length = pp.wavcen * pp.cellL
    opt_id = "$(opt_date)_NLOPT_singlefreq_lens_$(round(lblambda,digits=4))_$(round(ublambda,digits=4))_$(pp.orderfreq)_$(round(unit_cell_length,digits=4))_$(pp.gridL)_$(pp.orderwidth)_$(imgp.objL)_$(imgp.imgL)_$(imgp.binL)_$(xtol_rel)_$(maxeval)"
    directory = @sprintf("ImagingOpt.jl/geomsoptdata/%s", opt_id)
    Base.Filesystem.mkdir( directory )
    
    #save input parameters in json file
    jsonread = JSON3.read(read("$PARAMS_DIR/$pname.json", String))
    open("$directory/$(pname)_$(opt_date).json", "w") do io
        JSON3.pretty(io, jsonread)
    end

    println("######################### params loaded #########################")
    println()
    flush(stdout)
    print_params(pp, imgp, params.optp, params.recp, true, true, false, false)
    flush(stdout)

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


    function objective(parameters)
        geoms_grid = reshape(parameters, pp.gridL, pp.gridL)
        PSF = get_PSF(freq, surrogate, pp, imgp, geoms_grid, plan_nearfar, parallel)
        obj = (PSF[middle,middle] + PSF[middle+1,middle] + PSF[middle,middle+1] + PSF[middle+1,middle+1])
    end


    function myfunc(parameters, grad, save=true)
        obj = objective(parameters)
    
        if length(grad) > 0
            grad[1:end] = Zygote.gradient(geoms_flat -> objective(geoms_flat), parameters)[1]
        end
    
        if save
            @ignore_derivatives open(file_save_objective_vals, "a") do io
                writedlm(io, obj, ',')
            end
        end
    
        obj
    end

    #=
    #options = Optim.Options(time_limit = time_limit)
    options = Optim.Options(outer_iterations=outer_iterations, iterations=inner_iterations)
    method = Fminbox(Optim.LBFGS(m=10, linesearch=LineSearches.BackTracking(iterations=line_search_iterations) ))
    ret_optim = Optim.optimize(geoms_flat -> objective(geoms_flat, true), grad!, [pp.lbwidth for _ in 1:pp.gridL^2], [pp.ubwidth for _ in 1:pp.gridL^2], geoms_init, method, options)
    =#

    opt = Opt(:LD_MMA, pp.gridL^2)
    opt.lower_bounds = fill(pp.lbwidth,pp.gridL^2)
    opt.upper_bounds = fill(pp.ubwidth,pp.gridL^2)
    opt.max_objective = myfunc
    opt.xtol_rel = xtol_rel
    opt.maxeval = maxeval

    (maxf,mingeoms,ret) = NLopt.optimize(opt, geoms_init)

    println(ret)
    flush(stdout)

    #save output data in json file
    dict_output = Dict("return_val" => ret)
    output_data_filename = "$directory/output_data_$opt_date.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output)
    end

    #save optimized metasurface parameters (geoms)
    geoms_filename = "$directory/geoms_$(opt_id).csv"
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
    savefig("$directory/geoms_singlefreq_lens_$(opt_id).png")

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



function design_singlefreq_lens_OPTIM(pname, presicion, parallel, opt_date, outer_iterations=1000, inner_iterations=1, line_search_iterations=1)
    params = get_params(pname, presicion)
    pp = params.pp
    imgp = params.imgp
    
    #prepare opt files
    lblambda = pp.wavcen / pp.ubfreq 
    ublambda = pp.wavcen / pp.lbfreq 
    unit_cell_length = pp.wavcen * pp.cellL
    opt_id = "$(opt_date)_OPTIM_singlefreq_lens_$(round(lblambda,digits=4))_$(round(ublambda,digits=4))_$(pp.orderfreq)_$(round(unit_cell_length,digits=4))_$(pp.gridL)_$(pp.orderwidth)_$(imgp.objL)_$(imgp.imgL)_$(imgp.binL)_$(outer_iterations)_$(inner_iterations)_$(line_search_iterations)"
    directory = @sprintf("ImagingOpt.jl/geomsoptdata/%s", opt_id)
    Base.Filesystem.mkdir( directory )
    
    #save input parameters in json file
    jsonread = JSON3.read(read("$PARAMS_DIR/$pname.json", String))
    open("$directory/$(pname)_$(opt_date).json", "w") do io
        JSON3.pretty(io, jsonread)
    end

    println("######################### params loaded #########################")
    println()
    flush(stdout)
    print_params(pp, imgp, params.optp, params.recp, true, true, false, false)
    flush(stdout)

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

    function objective(geoms_flat, save=true)
        geoms_grid = reshape(geoms_flat, pp.gridL, pp.gridL)
        PSF = get_PSF(freq, surrogate, pp, imgp, geoms_grid, plan_nearfar, parallel)
        obj = -1*(PSF[middle,middle] + PSF[middle+1,middle] + PSF[middle,middle+1] + PSF[middle+1,middle+1])
    
        if save
            @ignore_derivatives open(file_save_objective_vals, "a") do io
                writedlm(io, obj, ',')
            end
        end
    
        obj
    end

    function grad!(grad::Vector, parameters::Vector)
    grad[1:end] = Zygote.gradient(geoms_flat -> objective(geoms_flat, false), parameters)[1]
    end

    #options = Optim.Options(time_limit = time_limit)
    options = Optim.Options(outer_iterations=outer_iterations, iterations=inner_iterations)
    method = Fminbox(Optim.LBFGS(m=10, linesearch=LineSearches.BackTracking(iterations=line_search_iterations) ))
    ret_optim = Optim.optimize(geoms_flat -> objective(geoms_flat, true), grad!, [pp.lbwidth for _ in 1:pp.gridL^2], [pp.ubwidth for _ in 1:pp.gridL^2], geoms_init, method, options)

    println(ret_optim)
    flush(stdout)

    mingeoms = Optim.minimizer(ret_optim)
    f_converged = Optim.f_converged(ret_optim)
    iteration_limit_reached = Optim.iteration_limit_reached(ret_optim)
    iterations = Optim.iterations(ret_optim)

    #save output data in json file
    dict_output = Dict("f_converged?" => f_converged, "iteration_limit_reached?" => iteration_limit_reached, "num_iterations_completed" => iterations, "num_iterations_input" => outer_iterations, "num_inner_iterations_input" => inner_iterations, "num_line_search_iterations_input" => line_search_iterations)
    output_data_filename = "$directory/output_data_$opt_date.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output)
    end

    #save optimized metasurface parameters (geoms)
    geoms_filename = "$directory/geoms_$(opt_id).csv"
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
    savefig("$directory/geoms_singlefreq_lens_$(opt_id).png")

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
        @time far_fields = ThreadsX.map(iF->get_far(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
        @time fftPSFs = ThreadsX.map(iF->get_fftPSF_from_far(far_fields[iF], freqs[iF], pp, imgp, plan_nearfar, plan_PSF),1:pp.orderfreq+1)
        dsur_dg_times_incidents = ThreadsX.map(iF->get_dsur_dg_times_incident(pp, freqs[iF], surrogates[iF], geoms, parallel),1:pp.orderfreq+1)

    else
        @time far_fields = map(iF->get_far(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, parallel),1:pp.orderfreq+1)
        @time fftPSFs = map(iF->get_fftPSF_from_far(far_fields[iF], freqs[iF], pp, imgp, plan_nearfar, plan_PSF),1:pp.orderfreq+1)
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
    relative_noise_levels = Matrix{Float64}(undef, 1, imgp.objN)
    for obji = 1:imgp.objN
        Tmap = Tmaps[obji]
        noise = noises[obji]
        B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
        
        @time image_Tmap_grid_noiseless = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel)
        relative_noise_level = imgp.noise_level * noise_multiplier / mean(image_Tmap_grid_noiseless) * 100
        relative_noise_levels[obji] = relative_noise_level
        image_Tmap_grid = add_noise_to_image(image_Tmap_grid_noiseless, imgp.differentiate_noise, noise, noise_multiplier)

        @time Test_flat, ret_val, num_evals = reconstruct_object(image_Tmap_grid, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, α, false, false, parallel, false, false)
        ret_vals[obji] = ret_val
        num_evals_list[obji] = num_evals

        MSE = sum((Tmap[:] .- Test_flat).^2) / sum(Tmap.^2)
        objective = objective +  (1/imgp.objN) * MSE

        grad_obji, num_cg_iters = dloss_dparams(pp, imgp, optp, recp, geoms, α, Tmap, B_Tmap_grid, Test_flat, image_Tmap_grid, noise, fftPSFs, dsur_dg_times_incidents, far_fields, freqs, plan_nearfar, plan_PSF, weights, parallel)
        num_cg_iters_list[obji] = num_cg_iters
        
        grad = grad + (1/imgp.objN) * grad_obji
    end
    objective, grad, num_cg_iters_list, ret_vals, num_evals_list, relative_noise_levels
end


function run_opt(pname, presicion, parallel, opt_date)
    params_init = get_params(pname, presicion)
    jsonread = JSON3.read(read("$PARAMS_DIR/$pname.json", String))
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
    flush(stdout)
    
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
    opt_id = "$(opt_date)_$(round(lblambda,digits=4))_$(round(ublambda,digits=4))_$(pp.orderfreq)_$(round(unit_cell_length,digits=4))_$(pp.gridL)_$(pp.orderwidth)_$(imgp.objL)_$(imgp.imgL)_$(imgp.binL)_$(imgp.objN)_$(imgp.lbT)_$(imgp.ubT)_$(imgp.differentiate_noise)_$(imgp.noise_level)_$(optp.geoms_init_loadsavename)_$(optp.αinit)_$(optp.maxeval)_$(optp.η)_$(recp.subtract_reg)"
    #opt_id = @sprintf("%s_geoms_%s_%d_%d_%d_batchsize_%d_alphainit_%.1e_maxeval_%d_diffnoise_%s", opt_date, optp.geoms_init_type, imgp.objL, imgp.imgL, pp.gridL, imgp.objN, optp.αinit, optp.maxeval, imgp.differentiate_noise)
    directory = "ImagingOpt.jl/optdata/$(opt_id)"
    Base.Filesystem.mkdir( directory )
    
    #save input parameters in json file
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
        file_save_best_alpha_vals = "$directory/best_alpha_vals_$opt_date.csv"  
    end

    #file for saving return values of reconstructions
    file_reconstruction_ret_vals = "$directory/reconstruction_return_vals_$opt_date.csv"

    #file for saving number of function evals for reconstructions
    file_reconstruction_num_evals = "$directory/reconstruction_num_function_evals_$opt_date.csv"
    
    #file for saving relative noise_levels for each image
    file_relative_noise_levels = "$directory/relative_noise_levels_$opt_date.csv"

    #file for saving conjugate gradient solve iterations
    file_save_cg_iters = "$directory/cg_iters_$opt_date.csv"
    open(file_save_cg_iters, "a") do io
        write(io, "Default maxiter = $(imgp.objL^2); set to maxiter = $(optp.cg_maxiter_factor * imgp.objL^2) \n")
    end
    
    #file for saving best geoms
    geoms_filename = "$directory/geoms_optimized_$opt_date.csv"
    
    #folder for saving geoms at intermediate iterations
    geoms_directory = "$directory/more_geoms"
    if ! isdir(geoms_directory)
        Base.Filesystem.mkdir(geoms_directory)
    end
    
    opt = Optimisers.ADAM(optp.η)
    setup = Optimisers.setup(opt, params_opt)

    params_opt_best = params_opt
    obj_best = Inf

    println("######################### beginning optimization #########################")
    println()
    for iter in 1:optp.maxeval
        @time objective, grad, num_cg_iters_list, ret_vals, num_evals_list, relative_noise_levels = compute_obj_and_grad(params_opt, params_init, freqs, surrogates, Tinit_flat, weights, noise_multiplier, plan_nearfar, plan_PSF, parallel)
    
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

        #save alpha val and best alpha val
        if optp.optimize_alpha
            α = params_opt[end] / optp.α_scaling
            open(file_save_alpha_vals, "a") do io
                writedlm(io, α, ',')
            end
            
            α_best = params_opt_best[end] / optp.α_scaling
            open(file_save_best_alpha_vals, "a") do io
                writedlm(io, α_best, ',')
            end
        end
        
        #save number of conjugate gradient solve iterations 
        open(file_save_cg_iters, "a") do io
            writedlm(io, num_cg_iters_list,',')
        end
    
        #save return value of reconstruction
        open(file_reconstruction_ret_vals, "a") do io
            writedlm(io, ret_vals,',')
        end
    
        #save number of function evals of reconstruction
        open(file_reconstruction_num_evals, "a") do io
            writedlm(io, num_evals_list,',')
        end
        
        #save relative noise values for all images
        open(file_relative_noise_levels, "a") do io
            writedlm(io, relative_noise_levels,',')
        end
        
        #save geoms every optp.saveeval iterations
        if mod(iter, optp.saveeval) == 0
            writedlm( geoms_filename,  params_opt_best[1:end-1] ,',')
            writedlm( "$(geoms_directory)/geoms_iter_$(iter)_$opt_date.csv",  params_opt_best[1:end-1],',')
        end
    
        setup, params_opt = Optimisers.update(setup, params_opt, grad)
        params_opt[1:end-1] = (x -> clamp(x, pp.lbwidth, pp.ubwidth)).(params_opt[1:end-1])
        params_opt[end] = clamp(params_opt[end], optp.α_lb * optp.α_scaling, Inf)
    end
    println()

    mingeoms = params_opt_best[1:end-1]
    println("######################### MAX EVAL REACHED #########################")
    println()
    dict_output = Dict("return_val" => "MAXEVAL_REACHED")
    #save output data in json file
    output_data_filename = "$directory/output_data_$opt_date.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output)
    end
    
    #save optimized metasurface parameters (geoms)
    writedlm( geoms_filename,  mingeoms,',')

    opt_id
end

