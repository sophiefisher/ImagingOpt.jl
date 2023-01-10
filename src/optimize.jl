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

function get_params(pname)
    paramstmp = copy(JSON3.read(read("$PARAMS_DIR/$pname.json", String)) )
    pptmp = paramstmp[:pp]
    wavcen = round(1/mean([1/pptmp[:lbλ_μm], 1/pptmp[:ubλ_μm]]),digits=2)
    
    lbfreq = wavcen/pptmp[:ubλ_μm]
    ubfreq = wavcen/pptmp[:lbλ_μm]
    
    lbwidth=pptmp[:lbwidth_μm]/wavcen 
    ubwidth=pptmp[:ubwidth_μm]/wavcen 
    
    F = pptmp[:F_μm]/wavcen
    depth = pptmp[:depth_μm]/wavcen
    cellL = pptmp[:cellL_μm]/wavcen
    thicknessg = pptmp[:thicknessg_μm]/wavcen
    thickness_sub = pptmp[:thickness_sub_μm]/wavcen
    
    pp = PhysicsParams(lbfreq, ubfreq, pptmp[:orderλ], wavcen, F, depth, pptmp[:gridL], cellL, lbwidth,ubwidth, pptmp[:orderwidth], thicknessg, pptmp[:materialg],  thickness_sub, pptmp[:materialsub], pptmp[:in_air],pptmp[:models_dir] )
    
    imgptmp = paramstmp[:imgp]
    imgp = ImagingParams(imgptmp[:objL], imgptmp[:imgL], imgptmp[:binL], imgptmp[:objN], imgptmp[:object_type], imgptmp[:object_data], imgptmp[:noise_level], imgptmp[:noise_abs], imgptmp[:emiss_noise_level])
    
    optptmp = paramstmp[:optp]
    optp = OptimizeParams(optptmp[:geoms_init], optptmp[:geoms_init_data])
    
    recptmp = paramstmp[:recp]
    recp = ReconstructionParams(recptmp[:Tinit],recptmp[:Tinit_data],recptmp[:tol])
    
    
    params = JobParams(pp, imgp, optp, recp)
end

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

#=
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
