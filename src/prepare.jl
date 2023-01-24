struct PhysicsParams
    lbfreq::Float64 #lower bound of wavelength 
    ubfreq::Float64 #upper bound of wavelength 
    orderfreq::Int #order of Chebyshev polynomial in frequency; number of points is orderfreq + 1 
    wavcen::Float64
    
    F::Float64 #focal length 
    depth::Float64 #depth of depth plane 
    gridL::Int #number of cells on each side of metasurface
    
    cellL::Float64 #unit cell size 
    lbwidth::Float64 #lower bound of pillar width 
    ubwidth::Float64 #upper bound of pillar width 
    orderwidth::Int #order of Chebyshev polynomial in width; number of points is orderwidth + 1 
    thicknessg::Float64 #pillar thickness 
    materialg::String #material of the pillars 
    
    thickness_sub::Float64 #substrate thickness
    materialsub::String #matieral of the substrate
    
    in_air::String #is the incident medium air or the substrate (True or False)
    models_dir::String #folder pointing to surrogate data
end

struct ImagingParams
    objL::Int
    imgL::Int
    binL::Int # binning of both sensor and object
    objN::Int
    object_type::String
    object_data::AbstractArray
    noise_level::Float64
    noise_abs::Bool
    emiss_noise_level::Float64
end

struct OptimizeParams
    geoms_init::String
    geoms_init_data::AbstractArray
end

struct ReconstructionParams
    Tinit::String
    Tinit_data::AbstractArray
    tol::Float64
end

struct JobParams
    pp::PhysicsParams
    imgp::ImagingParams
    optp::OptimizeParams
    recp::ReconstructionParams
end

function permittivity(mat::String, pp::PhysicsParams)
    wavcen = pp.wavcen
    if mat == "Si"
        function Si(freq)
            λ_μm = wavcen / freq
            11.67316 + (1/λ_μm^2) + (0.004482633 / (λ_μm^2 - 1.108205^2) )
        end 
        return Si
    end
end
    
#TODO: have function (prepare_physics?) to process PhysicsParams into unit-less parameters, and provide center wavelength
function prepare_physics(pp::PhysicsParams,freq::Float64, plan_nearfar::FFTW.cFFTWPlan)
    
    ϵsubfunc = permittivity(pp.materialsub,pp)
    ϵsub = ϵsubfunc(freq) 
    
    if pp.in_air == "True"
        incident = incident_field(pp.depth, freq, 1, pp.gridL, pp.cellL)
    else
        incident = incident_field(pp.depth, freq, √(ϵsub), pp.gridL, pp.cellL)
    end

    n2f_kernel = plan_nearfar * greens(pp.F, freq, 1., 1., pp.gridL, pp.cellL)
    incident, n2f_kernel
end

function prepare_surrogate(pp::PhysicsParams)
    #model2D = get_model2D(pp.materialsub, pp.materialg, pp.lbwidth, pp.lbfreq, pp.ubwidth, pp.ubfreq, pp.orderwidth, pp.orderfreq, pp.models_dir)
    #(;models1D, freqs) = get_models1D(model2D, pp.orderfreqPSF)
    models1D = get_models1D(pp.materialsub, pp.materialg, pp.in_air, pp.lbfreq, pp.ubfreq, pp.orderfreq, pp.lbwidth, pp.ubwidth, pp.orderwidth, pp.models_dir)
    
end

function prepare_geoms(params::JobParams)
    init = params.optp.geoms_init
    pp = params.pp
    if init == "uniform" #uniform width equal to half the lower bound and upper bound
        return fill((pp.lbwidth + pp.ubwidth)/2, pp.gridL, pp.gridL)
    elseif init == "load"
        filename = @sprintf("ImagingOpt.jl/geomsdata/%s",params.optp.geoms_init_data[1])
        reshape(readdlm(filename,',',Float64),pp.gridL,pp.gridL)
    end
end


function prepare_objects(imgp::ImagingParams, pp::PhysicsParams)
    if imgp.object_type == "uniform"
        lbT = imgp.object_data[1]
        ubT = imgp.object_data[2]
        random_object = function(seed = nothing)
            if ! isnothing(seed)
                Random.seed!(seed)
            end
            T = rand(lbT:eps():ubT)
            Tmap = fill(T, imgp.objL, imgp.objL);
        end
        Tmaps = [random_object(i) for i in 1:imgp.objN]
        return Tmaps
    elseif imgp.object_type == "random"
        lbT = imgp.object_data[1]
        ubT = imgp.object_data[2]
        random_object = function(seed = nothing)
            if ! isnothing(seed)
                Random.seed!(seed)
            end
            Tmap = rand(lbT:eps():ubT,imgp.objL, imgp.objL)
        end
        Tmaps = [random_object(i) for i in 1:imgp.objN]
        return Tmaps
    elseif imgp.object_type == "load_scale_single"
        #expects loaded array to be values from 0 to 1. scales values from lbT to ubT
        filename = @sprintf("ImagingOpt.jl/objdata/%s",imgp.object_data[1])
        lbT = imgp.object_data[2]
        ubT = imgp.object_data[3]
        diff = ubT - lbT
        Tmap = readdlm(filename,',',Float64).* diff .+ lbT
        if size(Tmap)[1] != imgp.objL
            error("imgp.objL does not match size of file")
        elseif imgp.objN != 1
            error("imgp.objN not equal to one")
        else
            return Tmap
        end
    end
end

function prepare_blackbody(Tmap::Matrix,  freqs::Vector, imgp::ImagingParams, pp::PhysicsParams)
    c = 299792458
    h = 6.62607015 * 10^-34
    kb = 1.380649 * 10^-23
    
    freqs_tmp = [repeat([freq,], imgp.objL, imgp.objL) for freq in freqs]
    freqs_tmp = arrarr_to_multi(freqs_tmp)
    
    B = (2 .* freqs_tmp.^3 ) ./ (exp.(h .* (freqs_tmp .* c .* 10^6 ./ pp.wavcen) ./ (kb .* Tmap) ) .- 1)  ;
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
    if recp.Tinit == "uniform"
        Tinit = repeat(recp.Tinit_data, imgp.objL^2)
    end
    Tinit
end
