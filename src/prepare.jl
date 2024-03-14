struct PhysicsParams{FloatType <: AbstractFloat, IntType <: Signed}
    lbfreq::FloatType #lower bound of wavelength 
    ubfreq::FloatType #upper bound of wavelength 
    orderfreq::IntType #order of Chebyshev polynomial in frequency; number of points is orderfreq + 1 
    wavcen::FloatType
    
    F::FloatType #focal length 
    depth::FloatType #depth of depth plane 
    gridL::IntType #number of cells on each side of metasurface
    
    cellL::FloatType #unit cell size 
    lbwidth::FloatType #lower bound of pillar width 
    ubwidth::FloatType #upper bound of pillar width 
    
    lbwidth_load::FloatType #lower bound of pillar width (for chebyshev interpolation)
    ubwidth_load::FloatType #upper bound of pillar width (for chebyshev interpolation)
    orderwidth::IntType #order of Chebyshev polynomial in width; number of points is orderwidth + 1
    
    thicknessg::FloatType #pillar thickness 
    materialg::String #material of the pillars 
    
    thickness_sub::FloatType #substrate thickness
    materialsub::String #matieral of the substrate
    
    in_air::String #is the incident medium air or the substrate (True or False)
    models_dir::String #folder pointing to surrogate data
    
    blackbody_scaling::FloatType #scaling of the black body spectrum
    PSF_scaling::FloatType #scaling of the PSFs
    
    nG::IntType
    
    
    function PhysicsParams{FloatType, IntType}(lbλ_μm, ubλ_μm, orderλ, F_μm, depth_μm, gridL, cellL_μm, lbwidth_μm, ubwidth_μm, lbwidth_μm_load, ubwidth_μm_load, orderwidth, thicknessg_μm, materialg, thickness_sub_μm, materialsub, in_air, models_dir, blackbody_scaling, PSF_scaling, nG) where {FloatType <: AbstractFloat} where {IntType <: Signed} 
        wavcen = round(1/mean([1/lbλ_μm, 1/ubλ_μm]),digits=2)

        lbfreq = wavcen/ubλ_μm
        ubfreq = wavcen/lbλ_μm

        lbwidth=lbwidth_μm/wavcen 
        ubwidth=ubwidth_μm/wavcen
        
        lbwidth_load = lbwidth_μm_load/wavcen
        ubwidth_load = ubwidth_μm_load/wavcen

        F = F_μm/wavcen
        depth = depth_μm/wavcen
        cellL = cellL_μm/wavcen
        thicknessg = thicknessg_μm/wavcen
        thickness_sub = thickness_sub_μm/wavcen
        
        new(lbfreq, ubfreq, orderλ, wavcen, F, depth, gridL, cellL, lbwidth, ubwidth, lbwidth_load, ubwidth_load,  orderwidth, thicknessg, materialg, thickness_sub, materialsub, in_air, models_dir, blackbody_scaling, PSF_scaling, nG)
    end
    
end


struct ImagingParams{FloatType <: AbstractFloat, IntType <: Signed}
    objL::IntType
    imgL::IntType
    binL::IntType # binning of both sensor and object
    objN::IntType
    object_type::String
    lbT::FloatType
    ubT::FloatType
    object_loadfilename::String
    object_savefilestring::String
    noise_level::FloatType
    emiss_noise_level::FloatType
    differentiate_noise::Bool
end

struct OptimizeParams{FloatType <: AbstractFloat, IntType <: Signed}
    geoms_init_type::String
    geoms_init_loadfilename::String
    geoms_init_loadsavename::String
    αinit::FloatType
    α_scaling::FloatType
    α_lb::FloatType
    maxeval::IntType
    saveeval::IntType
    xtol_rel::FloatType
    cg_maxiter_factor::IntType
    pre_optimize_alpha::Bool
    optimize_alpha::Bool
    η::FloatType
    eval_test_data::Bool
end

struct ReconstructionParams{FloatType <: AbstractFloat}
    T_init_type::String
    T_init_uniform_val::FloatType
    ftol_rel::FloatType
    geoms_init_savefilestring::String
    subtract_reg::FloatType
end

struct JobParams
    pp::PhysicsParams
    imgp::ImagingParams
    optp::OptimizeParams
    recp::ReconstructionParams
end

function permittivity(mat::String, pp::PhysicsParams)
    wavcen = pp.wavcen
    if mat == "Si_no_absorption"
        file = "ImagingOpt.jl/materialdata/Si_Shkondin_n.csv"
        data = readdlm(file,',')
        wavelengths_n = data[:,1]
        nvals = data[:,2]
        idxs = (wavelengths_n .< pp.wavcen/pp.lbfreq + 0.1) .&& (wavelengths_n .> pp.wavcen/pp.ubfreq - 0.1)
        wavelengths_n = wavelengths_n[idxs]
        nvals = nvals[idxs]
        Si_n = Interpolations.LinearInterpolation(wavelengths_n, nvals)
        
        function Si_no_absorption(freq)
            λ_μm = pp.wavcen / freq
            Si_n(λ_μm)^2 
        end
        return Si_no_absorption
        
    elseif mat == "Si"
        file = "ImagingOpt.jl/materialdata/Si_Shkondin_n.csv"
        data = readdlm(file,',')
        wavelengths_n = data[:,1]
        nvals = data[:,2]
        idxs = (wavelengths_n .< pp.wavcen/pp.lbfreq + 0.1) .&& (wavelengths_n .> pp.wavcen/pp.ubfreq - 0.1)
        wavelengths_n = wavelengths_n[idxs]
        nvals = nvals[idxs]
        Si_n = Interpolations.LinearInterpolation(wavelengths_n, nvals)
        
        file = "ImagingOpt.jl/materialdata/Si_Shkondin_k.csv"
        data = readdlm(file,',')
        wavelengths_k = data[:,1]
        kvals = data[:,2]
        idxs = (wavelengths_k .< pp.wavcen/pp.lbfreq + 0.1) .&& (wavelengths_k .> pp.wavcen/pp.ubfreq - 0.1)
        wavelengths_k = wavelengths_k[idxs]
        kvals = kvals[idxs]
        Si_k = Interpolations.LinearInterpolation(wavelengths_k, kvals)

        function Si(freq)
            λ_μm = pp.wavcen / freq
            Si_n(λ_μm)^2 - Si_k(λ_μm)^2 + (2im * Si_n(λ_μm) * Si_k(λ_μm) )
        end
    end
        
    #=if mat == "Si"
        function Si(freq)
            λ_μm = wavcen / freq
            convert(typeof(freq),11.67316) + (1/λ_μm^2) + (convert(typeof(freq),0.004482633) / (λ_μm^2 - convert(typeof(freq),1.108205)^2) )
        end 
        return Si
    end=#
    
end
    
#TODO: have function (prepare_physics?) to process PhysicsParams into unit-less parameters, and provide center wavelength
function prepare_incident(pp::PhysicsParams,freq::AbstractFloat)
    if pp.in_air == "True"
        incident = incident_field(pp.depth, freq, 1, pp.gridL, pp.cellL)
    else
        ϵsubfunc = permittivity(pp.materialsub,pp)
        ϵsub = ϵsubfunc(freq) 
        incident = incident_field(pp.depth, freq, √(ϵsub), pp.gridL, pp.cellL)
    end
    
    #incident = incident_field(pp.depth, freq, convert(typeof(freq),1), pp.gridL, pp.cellL)
    incident
end

function prepare_n2f_kernel(pp::PhysicsParams,imgp::ImagingParams,freq::AbstractFloat, plan_nearfar::FFTW.cFFTWPlan)
    n2f_size = pp.gridL + imgp.binL*(imgp.objL + imgp.imgL)
    n2f_kernel = plan_nearfar * greens(pp.F, freq, 1,1, n2f_size, pp.cellL)
    
end

function prepare_surrogate(pp::PhysicsParams)
    surrogates, freqs = get_models1D(pp.materialsub, pp.materialg, pp.in_air, pp.lbfreq, pp.ubfreq, pp.orderfreq, pp.lbwidth_load, pp.ubwidth_load, pp.orderwidth, pp.cellL, pp.thicknessg, pp.thickness_sub, pp.nG, pp.models_dir)
end

function prepare_geoms(params::JobParams)
    init = params.optp.geoms_init_type
    pp = params.pp
    if init == "uniform" #uniform width equal to half the lower bound and upper bound
        return fill((pp.lbwidth + pp.ubwidth)/2, pp.gridL, pp.gridL)
    elseif init == "load"
        filename = @sprintf("ImagingOpt.jl/geomsdata/%s",params.optp.geoms_init_loadfilename )
        reshape(readdlm(filename,',',typeof(pp.lbfreq)),pp.gridL,pp.gridL)
        
    elseif init == "random"
        return rand(pp.lbwidth:eps( typeof(pp.lbfreq) ):pp.ubwidth, pp.gridL, pp.gridL)
    end
end


function prepare_objects(imgp::ImagingParams, pp::PhysicsParams)
    floattype = typeof(pp.lbfreq)
    
    if imgp.object_type == "uniform"
        lbT = imgp.lbT
        ubT = imgp.ubT
        random_object = function(seed = nothing)
            if ! isnothing(seed)
                Random.seed!(seed)
            end
            T = rand(lbT:eps(floattype):ubT)
            Tmap = fill(T, imgp.objL, imgp.objL);
        end
        Tmaps = [random_object() for i in 1:imgp.objN]
        return Tmaps
    elseif imgp.object_type == "random"
        lbT = imgp.lbT
        ubT = imgp.ubT
        random_object = function(seed = nothing)
            if ! isnothing(seed)
                Random.seed!(seed)
            end
            Tmap = rand(lbT:eps(floattype):ubT,imgp.objL, imgp.objL)
        end
        Tmaps = [random_object() for i in 1:imgp.objN]
        return Tmaps
    elseif imgp.object_type == "load_scale_single"
        #expects loaded array to be values from 0 to 1. scales values from lbT to ubT
        filename = @sprintf("ImagingOpt.jl/objdata/%s",imgp.object_loadfilename)
        lbT = imgp.lbT
        ubT = imgp.ubT
        diff = ubT - lbT
        Tmap = readdlm(filename,',',floattype).* diff .+ lbT
        if size(Tmap)[1] != imgp.objL
            error("imgp.objL does not match size of file")
        elseif imgp.objN != 1
            error("imgp.objN not equal to one")
        else
            return [Tmap,]
        end
    end
end

#TODO - doesn't need imgp as parameter
function prepare_blackbody(Tmap::Matrix,  freqs::Vector, imgp::ImagingParams, pp::PhysicsParams)
    c = convert(typeof(freqs[1]),299792458)
    h = convert(typeof(freqs[1]),6.62607015e-34)
    kb = convert(typeof(freqs[1]),1.380649e-23)
    
    Binit = [(2 * freq ^3 ) ./ (exp.(h * (freq * c * 10^6 / pp.wavcen) ./ (kb * Tmap) ) .- 1) for freq in freqs]
    B = pp.blackbody_scaling .* arrarr_to_multi(Binit)
end

function prepare_blackbody(Tmaps::Vector, freqs::Vector, imgp::ImagingParams, pp::PhysicsParams)
    Bs =  Vector(undef, imgp.objN)
    for i in 1:imgp.objN
        Tmap = Tmaps[i]
        Bs[i] = prepare_blackbody(Tmap, freqs, imgp, pp)
    end
    Bs
end

function prepare_reconstruction(recp::ReconstructionParams, imgp::ImagingParams)
    floattype = typeof(imgp.lbT)
    if recp.T_init_type == "uniform"
        Tinit = repeat( [recp.T_init_uniform_val,], imgp.objL^2)
    elseif recp.T_init_type == "random"
        lbT = imgp.lbT
        ubT = imgp.ubT
        Tinit = rand(lbT:eps(floattype):ubT,imgp.objL^2)
    end
    Tinit
end

function prepare_noise(imgp::ImagingParams)
    noise = imgp.noise_level .* randn(imgp.imgL, imgp.imgL) 
end

function prepare_noises(imgp::ImagingParams)
    #Random.seed!(0)
    noises = [prepare_noise(imgp) for _ in 1:imgp.objN]
end

function prepare_fft_plans(pp::PhysicsParams, imgp::ImagingParams)
    n2f_size = pp.gridL + imgp.binL*(imgp.objL + imgp.imgL)
    plan_nearfar = plan_fft!(zeros(Complex{typeof(pp.lbfreq)}, (n2f_size, n2f_size)), flags=FFTW.MEASURE)
    plan_PSF = plan_fft!(zeros(Complex{typeof(pp.lbfreq)}, (imgp.objL + imgp.imgL, imgp.objL + imgp.imgL)), flags=FFTW.MEASURE)

    plan_nearfar, plan_PSF
end

#only need if differentiate noise is false
function prepare_noise_multiplier(pp::PhysicsParams, imgp::ImagingParams, surrogates, freqs, weights, plan_nearfar, plan_PSF, parallel::Bool=true)
    Tmap = prepare_objects(imgp, pp)[1]
    B_Tmap_grid = prepare_blackbody(Tmap, freqs, imgp, pp)
    get_fftPSF_freespace_iF = iF->get_fftPSF_freespace(freqs[iF], surrogates[iF], pp, imgp, plan_nearfar, plan_PSF)
    if parallel
        fftPSFs_freespace = ThreadsX.map(get_fftPSF_freespace_iF,1:pp.orderfreq+1)
    else
        fftPSFs_freespace = map(get_fftPSF_freespace_iF,1:pp.orderfreq+1)
    end
    image_Tmap_grid_freespace = make_image_noiseless(pp, imgp, B_Tmap_grid, fftPSFs_freespace, freqs, weights, plan_nearfar, plan_PSF, parallel)
    
    mean(image_Tmap_grid_freespace)
end

function prepare_weights(pp::PhysicsParams)
    ClenshawCurtisQuadrature(pp.orderfreq + 1).weights
end
    