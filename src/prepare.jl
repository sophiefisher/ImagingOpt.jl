struct PhysicsParams
    lbfreq::Float64 #lower bound of wavelength 
    ubfreq::Float64 #upper bound of wavelength 
    orderfreq::Int #order of Chebyshev polynomial in frequency; number of points is orderfreq + 1
    orderfreqPSF::Int 
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

    models_dir::String #folder pointing to surrogate data
end

struct ImagingParams
    objL::Int
    imgL::Int
    binL::Int # binning of both sensor and object
    objN::Int
    object_type::String
    object_data::AbstractArray
end

struct OptimizeParams
    init::String
end

struct JobParams
    pp::PhysicsParams
    imgp::ImagingParams
    optp::OptimizeParams
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
function prepare_physics(pp::PhysicsParams,freq::Float64)
    
    ϵsubfunc = permittivity(pp.materialsub,pp)
    ϵsub = ϵsubfunc(freq) 
    
    incident = incident_field(pp.depth, freq, √(ϵsub), pp.gridL, pp.cellL)

    n2f_kernel = fft(greens(pp.F, freq, 1., 1., pp.gridL, pp.cellL)) 
    incident, n2f_kernel
end

function prepare_surrogate(pp::PhysicsParams)
    model2D = get_model2D(pp.materialsub, pp.materialg, pp.lbwidth, pp.lbfreq, pp.ubwidth, pp.ubfreq, pp.orderwidth, pp.orderfreq, pp.models_dir)
    (;models1D, freqs) = get_models1D(model2D, pp.orderfreqPSF)
    
end

function prepare_geoms(params::JobParams)
    init = params.optp.init
    pp = params.pp
    if init == "uniform" #uniform width equal to half the lower bound and upper bound
        return fill((pp.lbwidth + pp.ubwidth)/2, 1, pp.gridL, pp.gridL)
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
    elseif imgp.object_type == "random"
        lbT = imgp.object_data[1]
        ubT = imgp.object_data[2]
        random_object = function(seed = nothing)
            if ! isnothing(seed)
                Random.seed!(seed)
            end
            Tmap = rand(lbT:eps():ubT,imgp.objL, imgp.objL)
        end
    end
    
    Tmaps = [random_object(i) for i in 1:imgp.objN]
    Tmaps
    #still need to add noise
end

function prepare_blackbody(Tmaps, imgp::ImagingParams, pp::PhysicsParams)
    c = 299792458
    h = 6.62607015 * 10^-34
    kb = 1.380649 * 10^-23
    
    freqs = reverse(chebpoints(pp.orderfreqPSF, pp.lbfreq, pp.ubfreq))
    freqs_tmp = [repeat([freq,], imgp.objL, imgp.objL) for freq in freqs]
    freqs_tmp = arrarr_to_multi(freqs_tmp)
    
    Bs =  Vector(undef, imgp.objN)
    for i in 1:imgp.objN
        Tmap = Tmaps[i]
        Tmap_tmp = repeat(Tmap,1,1,pp.orderfreqPSF+1)
        Bs[i] = (2 .* freqs_tmp.^3 ) ./ (exp.(h .* (freqs_tmp .* c .* 10^6 ./ pp.wavcen) ./ (kb .* Tmap) ) .- 1)  ;
    end
    Bs
end
