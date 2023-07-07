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

#function design_singlefreq_lens()
#end

#optimize metasurface parameters for fixed alpha; no noise
function run_opt(params, presicion, parallel, opt_date)
    #params = get_params(pname, presicion)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    
    opt_id = @sprintf("%s_geoms_%s_alphainit_%.1e_maxeval_%d_xtolrel_%.1e", opt_date, optp.geoms_init_type, optp.αinit, optp.maxeval, optp.xtol_rel)
    directory = @sprintf("ImagingOpt.jl/optdata/%s", opt_id)
    Base.Filesystem.mkdir( directory )
    
    #save input parameters in json file
    jsonread = JSON3.read(read("$PARAMS_DIR/$pname.json", String))
    open("$directory/$opt_id.json", "w") do io
        JSON3.pretty(io, jsonread)
    end

    #save additional system parameters in json file
    extra_params = compute_system_params(pp, imgp)
    extra_params_filename = "$directory/extra_params_$opt_date.json"
    open(extra_params_filename,"w") do io
        JSON3.pretty(io, extra_params)
    end
    
    #file for saving objective vals
    if optp.save_objective_vals == true
        file_save_objective_vals = "$directory/objective_vals_$opt_date.csv"
    end    

    #file for saving conjugate gradient solve iterations
    file_save_cg_iters = "$directory/cg_iters_$opt_date.txt"
    open(file_save_cg_iters, "a") do io
        write(io, "Default maxiter = $(imgp.objL^2); set to maxiter = $(optp.cg_maxiter_factor * imgp.objL^2) \n")
    end
    
    surrogates, freqs = prepare_surrogate(pp)
    Tinit_flat = prepare_reconstruction(recp, imgp)
    Tmaps = prepare_objects(imgp, pp)

    #save Tmaps
    Tmaps_flat = reduce(hcat, [Tmaps[i][:] for i in 1:length(Tmaps)])
    file_save_Tmaps_flat = "$directory/Tmaps_$opt_date.csv"
    writedlm( file_save_Tmaps_flat,  Tmaps_flat,',')
    
    plan_nearfar = plan_fft!(zeros(Complex{typeof(freqs[1])}, (2*pp.gridL, 2*pp.gridL)), flags=FFTW.MEASURE)
    plan_PSF = plan_fft!(zeros(Complex{typeof(freqs[1])}, (imgp.objL + imgp.imgL, imgp.objL + imgp.imgL)), flags=FFTW.MEASURE)
    weights = convert.( typeof(freqs[1]), ClenshawCurtisQuadrature(pp.orderfreq + 1).weights)
    
    geoms_init = prepare_geoms(params)
    parameters_init = geoms_init[:]


    function myfunc(parameters::Vector, grad::Vector)
        start = time()
        #parameters = geoms
        geoms = reshape(parameters, pp.gridL, pp.gridL)
        
        #=
        f = figure(figsize=(4,4))
        imshow(geoms, vmin=pp.lbwidth, vmax = pp.ubwidth)
        colorbar()
        display(f)
        =#
        
        if parallel == true
            fftPSFs = ThreadsX.map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
        else
            fftPSFs = map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
        end

        
        objective = convert(typeof(freqs[1]), 0)
        grad[:] = zeros(typeof(freqs[1]), pp.gridL^2)
        
        for obji = 1:imgp.objN
            Tmap = Tmaps[obji]
            B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
            
            image_Tmap_grid = make_images(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel);
            Test_flat = reconstruct_object(image_Tmap_grid, Tmap, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, optp.αinit, false, parallel)
            
            #=
            f = figure(figsize=(6,3))
            subplot(1,2,1)
            imshow(Tmap)
            colorbar()
            subplot(1,2,2)
            imshow(reshape(Test_flat,imgp.objL, imgp.objL))
            colorbar()
            display(f)
            =#
            
            objective = objective +  (1/imgp.objN) * ( (Tmap[:] - Test_flat)'*(Tmap[:] - Test_flat) ) / (Tmap[:]' * Tmap[:])

            term1plusterm2_hessian = term1plusterm2_hes(optp.αinit, pp, imgp, fftPSFs, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, parallel);
            H = Hes(pp.orderfreq + 1, pp.wavcen, imgp.objL, imgp.imgL, term1plusterm2_hessian, fftPSFs, freqs, Test_flat, plan_nearfar, plan_PSF, weights, pp.blackbody_scaling, parallel)
            b = 2 * (Tmap[:] - Test_flat) / (Tmap[:]' * Tmap[:])

            lambda = zeros(typeof(freqs[1]), imgp.objL^2)
            lambda, ch = cg!(lambda, H, b, log=true, maxiter = optp.cg_maxiter_factor * imgp.objL^2);
            println(ch)
            flush(stdout)
            open(file_save_cg_iters, "a") do io
                writedlm(io, length(ch[:resnorm]), ',')
            end
        
            if length(grad) > 0
                grad[1:end] = grad[1:end] + ( (1/imgp.objN) * jacobian_vp_manual(lambda, pp, imgp,  geoms, surrogates, freqs, Test_flat, plan_nearfar, plan_PSF, weights, image_Tmap_grid, B_Tmap_grid, fftPSFs, parallel)[:] )
                
            end
        end
        elapsed = time() - start
    
        if optp.save_objective_vals == true
            open(file_save_objective_vals, "a") do io
                writedlm(io, objective, ',')
            end
        end
    
        println()
        println(@sprintf("time elapsed = %f",elapsed))
        println(@sprintf("OBJECTIVE VAL IS %.12f",objective) )
        println()
        flush(stdout)
        objective
    end
    
    
    opt = Opt(:LD_MMA, pp.gridL^2)
    opt.min_objective = myfunc
    
    opt.lower_bounds = pp.lbwidth
    opt.upper_bounds = pp.ubwidth
    
    opt.xtol_rel = optp.xtol_rel
    opt.maxeval = optp.maxeval
    

    
    #myfunc(parameters_init, similar(parameters_init))
    (minobj,minparams,ret) = optimize(opt, parameters_init)
    println("RETURN VALUE IS $ret")
    
    #save output data in json file
    dict_output = Dict("return_val" => ret)
    output_data_filename = "$directory/output_data_$opt_date.json"
    open(output_data_filename,"w") do io
        JSON3.pretty(io, dict_output)
    end
    
    #save optimized metasurface parameters (geoms)
    geoms_filename = "$directory/geoms_$opt_date.csv"
    writedlm( geoms_filename,  minparams,',')

    
end


function process_opt(params, presicion, parallel, opt_date)
    #params = get_params(pname, presicion)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    recp = params.recp
    
    opt_id = @sprintf("%s_geoms_%s_alphainit_%.1e_maxeval_%d_xtolrel_%.1e", opt_date, optp.geoms_init_type, optp.αinit, optp.maxeval, optp.xtol_rel)
    directory = @sprintf("ImagingOpt.jl/optdata/%s", opt_id)
    
    surrogates, freqs = prepare_surrogate(pp)
    Tinit_flat = prepare_reconstruction(recp, imgp)
    
    #load Tmaps
    file_save_Tmaps_flat = "$directory/Tmaps_$opt_date.csv"
    Tmaps = readdlm(file_save_Tmaps_flat,',')
    Tmaps = [reshape(Tmaps[:,i], imgp.objL,imgp.objL) for i in 1:size(Tmaps)[2]]
    
    plan_nearfar = plan_fft!(zeros(Complex{typeof(freqs[1])}, (2*pp.gridL, 2*pp.gridL)), flags=FFTW.MEASURE)
    plan_PSF = plan_fft!(zeros(Complex{typeof(freqs[1])}, (imgp.objL + imgp.imgL, imgp.objL + imgp.imgL)), flags=FFTW.MEASURE)
    weights = convert.( typeof(freqs[1]), ClenshawCurtisQuadrature(pp.orderfreq + 1).weights)
    
    geoms_init = prepare_geoms(params)
    
    #save initial reconstruction
    if parallel == true
        fftPSFs = ThreadsX.map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    else
        fftPSFs = map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms_init, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    end
    
    figure(figsize=(12,10))
    suptitle("initial reconstruction")
    for obji = 1:imgp.objN
        Tmap = Tmaps[obji]
        B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)

        image_Tmap_grid = make_images(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel);
        Test = reshape(reconstruct_object(image_Tmap_grid, Tmap, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, optp.αinit, false, parallel), imgp.objL, imgp.objL)
        subplot(imgp.objN, 3, obji*3 - 2)
        imshow(Tmap, vmin = imgp.lbT, vmax = imgp.ubT)
        colorbar()
        title(L"T(x,y) \ %$obji")
    
        subplot(imgp.objN, 3, obji*3 - 1)
        imshow(Test, vmin = imgp.lbT, vmax = imgp.ubT)
        colorbar()
        title(L"T_{est}(x,y) \ %$obji")
    
        subplot(imgp.objN, 3, obji*3 )
        imshow( (Test .- Tmap)./Tmap .* 100)
        colorbar()
        title("% difference $obji")
    end
    tight_layout()
    savefig("$directory/reconstruction_initial_$opt_date.png")
    
    #MIT initial reconstruction
    fig_MIT, ax_MIT = subplots(2,4,figsize=(14,8))
    object_loadfilename_MIT = "MIT$(imgp.objL).csv"
    filename_MIT = @sprintf("ImagingOpt.jl/objdata/%s",object_loadfilename_MIT)
    lbT = imgp.lbT
    ubT = imgp.ubT
    diff = ubT - lbT
    Tmap_MIT = readdlm(filename_MIT,',',typeof(freqs[1])).* diff .+ lbT

    B_Tmap_grid_MIT = prepare_blackbody(Tmap_MIT, freqs, imgp, pp)
    image_Tmap_grid_MIT = make_images(pp, imgp, B_Tmap_grid_MIT, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel);
    Test_MIT = reshape(reconstruct_object(image_Tmap_grid_MIT, Tmap_MIT, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, optp.αinit, false, parallel), imgp.objL, imgp.objL)

    p1 = ax_MIT[1,1].imshow(Tmap_MIT, vmin = imgp.lbT, vmax = imgp.ubT)
    fig_MIT.colorbar(p1, ax=ax_MIT[1,1])
    ax_MIT[1,1].set_title(L"T(x,y) \  \mathrm{initial}")

    p2 = ax_MIT[1,2].imshow(Test_MIT, vmin = imgp.lbT, vmax = imgp.ubT)
    fig_MIT.colorbar(p2, ax=ax_MIT[1,2])
    ax_MIT[1,2].set_title(L"T_{est}(x,y) \  \mathrm{initial}")
    
    p3 = ax_MIT[1,3].imshow( (Test_MIT .- Tmap_MIT)./Tmap_MIT .* 100)
    fig_MIT.colorbar(p3, ax=ax_MIT[1,3])
    ax_MIT[1,3].set_title("% difference initial")

    p4 = ax_MIT[1,4].imshow(image_Tmap_grid_MIT)
    fig_MIT.colorbar(p4, ax=ax_MIT[1,4])
    ax_MIT[1,4].set_title("image initial")
    
    #now use optimized geoms
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
    savefig("$directory/geoms_$opt_date.png")
    subplot(1,3,3)
    imshow(geoms)
    colorbar()
    title("optimized metasurface \n parameters")
    savefig("$directory/geoms_$opt_date.png")


    #save optimized reconstructions and images
    if parallel == true
        fftPSFs = ThreadsX.map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    else
        fftPSFs = map(iF->get_fftPSF(freqs[iF], surrogates[iF], pp, imgp, geoms, plan_nearfar, plan_PSF, parallel),1:pp.orderfreq+1)
    end
    
    fig1, ax1 = subplots(imgp.objN,3,figsize=(12,10))
    fig1.suptitle("optimized reconstruction")

    fig2, ax2 = subplots(imgp.objN,2,figsize=(8,10))
    fig2.suptitle("optimized images")
    for obji = 1:imgp.objN
        Tmap = Tmaps[obji]
        B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)

        image_Tmap_grid = make_images(pp, imgp, B_Tmap_grid, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel);
        Test = reshape(reconstruct_object(image_Tmap_grid, Tmap, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, optp.αinit, false, parallel), imgp.objL, imgp.objL)
        
        p1 = ax1[obji,1].imshow(Tmap, vmin = imgp.lbT, vmax = imgp.ubT)
        fig1.colorbar(p1, ax=ax1[obji,1])
        ax1[obji,1].set_title(L"T(x,y) \ %$obji")
    
        p2 = ax1[obji,2].imshow(Test, vmin = imgp.lbT, vmax = imgp.ubT)
        fig1.colorbar(p2, ax=ax1[obji,2])
        ax1[obji,2].set_title(L"T_{est}(x,y) \ %$obji")
    
        p3 = ax1[ obji,3 ].imshow( (Test .- Tmap)./Tmap .* 100 )
        fig1.colorbar(p3, ax = ax1[ obji,3 ])
        ax1[obji,3].set_title("% difference $obji")
    
        p4 = ax2[obji,1].imshow(Tmap, vmin = imgp.lbT, vmax = imgp.ubT)
        fig2.colorbar(p4, ax = ax2[obji,1])
        ax2[obji,1].set_title(L"T(x,y) \ %$obji")
    
        p5 = ax2[obji,2].imshow(image_Tmap_grid)
        fig2.colorbar(p5, ax = ax2[obji,2])
        ax2[obji,2].set_title("image $obji" )
    end
    fig1.tight_layout()
    fig1.savefig("$directory/reconstruction_optimized_$opt_date.png")

    fig2.tight_layout()
    fig2.savefig("$directory/images_optimized_$opt_date.png")


    #plot objective values
    if optp.save_objective_vals == true
        file_save_objective_vals = "$directory/objective_vals_$opt_date.csv"
        objdata = readdlm(file_save_objective_vals,',')
        figure(figsize=(12,5))
        suptitle(L"\mathrm{objective \  data } ,  \langle \frac{|| T - T_{est} ||^2}{  || T ||^2} \rangle_{T}")
        subplot(1,2,1)
        plot(objdata,".-")
    
        subplot(1,2,2)
        semilogy(objdata,".-")
        tight_layout()
        savefig("$directory/objective_vals_$opt_date.png")
    end

    #plot PSFs (make sure there are only 21 of them)
    if parallel == true
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

    #now try reconstruction on more readable image    
    image_Tmap_grid_MIT = make_images(pp, imgp, B_Tmap_grid_MIT, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, parallel);
    Test_MIT = reshape(reconstruct_object(image_Tmap_grid_MIT, Tmap_MIT, Tinit_flat, pp, imgp, optp, recp, fftPSFs, freqs, weights, plan_nearfar, plan_PSF, optp.αinit, false, parallel), imgp.objL, imgp.objL)

    p1 = ax_MIT[2,1].imshow(Tmap_MIT, vmin = imgp.lbT, vmax = imgp.ubT)
    fig_MIT.colorbar(p1, ax=ax_MIT[2,1])
    ax_MIT[2,1].set_title(L"T(x,y) \  \mathrm{optimized}")

    p2 = ax_MIT[2,2].imshow(Test_MIT, vmin = imgp.lbT, vmax = imgp.ubT)
    fig_MIT.colorbar(p2, ax=ax_MIT[2,2])
    ax_MIT[2,2].set_title(L"T_{est}(x,y) \  \mathrm{optimized}")
    
    p3 = ax_MIT[2,3].imshow( (Test_MIT .- Tmap_MIT)./Tmap_MIT .* 100)
    fig_MIT.colorbar(p3, ax=ax_MIT[2,3])
    ax_MIT[2,3].set_title("% difference optimized")

    p4 = ax_MIT[2,4].imshow(image_Tmap_grid_MIT)
    fig_MIT.colorbar(p4, ax=ax_MIT[2,4])
    ax_MIT[2,4].set_title("image optimized")


    fig_MIT.tight_layout()
    fig_MIT.savefig("$directory/MIT_reconstruction_$opt_date.png")

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
