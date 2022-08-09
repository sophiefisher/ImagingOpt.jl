# collection of useful jobs to run on a params file
struct JobParams
    pp::PhysicsParams
    imgp::ImagingParams
    optp::OptimizeParams
end

StructTypes.StructType(::Type{PhysicsParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{ImagingParams}) = StructTypes.Struct()
StructTypes.StructType(::Type{OptimizeParams}) = StructTypes.Struct()
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
    
    pp = PhysicsParams(lbfreq, ubfreq, pptmp[:orderλ], pptmp[:orderλPSF], wavcen, F, depth, pptmp[:gridL], pptmp[:cellL_μm],lbwidth,ubwidth, pptmp[:orderwidth], thicknessg, pptmp[:materialg],  thickness_sub, pptmp[:materialsub], pptmp[:models_dir] )
    
    imgptmp = paramstmp[:imgp]
    imgp = ImagingParams(imgptmp[:objL], imgptmp[:imgL], imgptmp[:binL], imgptmp[:objN], imgptmp[:object_type], imgptmp[:object_data])
    
    optptmp = paramstmp[:optp]
    optp = OptimizeParams(optptmp[:init])
    
    params = JobParams(pp, imgp, optp)
end

function test_forwardmodel(pname)
    params = get_params(pname)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    psfL = imgp.objL + imgp.imgL
    nF = pp.orderfreqPSF + 1
    
    Tmaps = prepare_objects(imgp::ImagingParams, pp::PhysicsParams)
    Bs = prepare_blackbody(Tmaps, imgp::ImagingParams, pp::PhysicsParams)
    
    surrogate = prepare_surrogate(pp)
    geoms = prepare_geoms(pp, optp.init)
    freqs = reverse(chebpoints(pp.orderfreqPSF, pp.lbfreq, pp.ubfreq))
    
    ys = [zeros(imgp.imgL, imgp.imgL) for _ in 1:imgp.objN]
    
    for iF in 1:nF
        freq = freqs[iF]
        incident, n2f_kernel = prepare_physics(pp, freq)
        far, _, _ = geoms_to_far(geoms, surrogate, incident, n2f_kernel, freq)
        PSFs = far_to_PSFs(far, psfL, imgp.binL)
        G, _ = PSFs_to_G(PSFs, imgp.objL, imgp.imgL, nF, iF, pp.lbfreq, pp.ubfreq)
        
        for iO in 1:imgp.objN
            y_temp = G * Bs[iO][:,:,iF][:]
            y_temp = reshape(y_temp, (imgp.imgL,imgp.imgL))
            ys[iO] = ys[iO] + y_temp
        end
    end
    
    ys
    
end

function test_init(pname)
    params = get_params(pname)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    psfL = imgp.objL + imgp.imgL
    nF = pp.orderfreqPSF + 1
    
    Tmaps = prepare_objects(imgp::ImagingParams, pp::PhysicsParams)
    Bs = prepare_blackbody(Tmaps, imgp::ImagingParams, pp::PhysicsParams)
    
    surrogate = prepare_surrogate(pp)
    geoms = prepare_geoms(pp, optp.init)
end


function test_forwardmodel_noinit(params, Bs, surrogate, geoms, freqs)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    psfL = imgp.objL + imgp.imgL
    nF = pp.orderfreqPSF + 1
    
    ys = [zeros(imgp.imgL, imgp.imgL) for _ in 1:imgp.objN]
    
    for iF in 1:nF
        freq = freqs[iF]
        incident, n2f_kernel = prepare_physics(pp, freq)
        far, _, _ = geoms_to_far(geoms, surrogate, incident, n2f_kernel, freq)
        PSFs = far_to_PSFs(far, psfL, imgp.binL)
        G, _ = PSFs_to_G(PSFs, imgp.objL, imgp.imgL, nF, iF, pp.lbfreq, pp.ubfreq)
        
        for iO in 1:imgp.objN
            y_temp = G * Bs[iO][:,:,iF][:]
            y_temp = reshape(y_temp, (imgp.imgL,imgp.imgL))
            ys[iO] = ys[iO] + y_temp
        end
    end
    
    ys
    
end

function test_forwardmodel_perfreq_noinit(params, Bs, surrogate, geoms, ys, freq, iF)
    pp = params.pp
    imgp = params.imgp
    optp = params.optp
    psfL = imgp.objL + imgp.imgL
    nF = pp.orderfreqPSF + 1
    
    incident, n2f_kernel = prepare_physics(pp, freq)
    far, _, _ = geoms_to_far(geoms, surrogate, incident, n2f_kernel, freq)
    PSFs = far_to_PSFs(far, psfL, imgp.binL)
    G, _ = PSFs_to_G(PSFs, imgp.objL, imgp.imgL, nF, iF, pp.lbfreq, pp.ubfreq)

    for iO in 1:imgp.objN
        y_temp = G * Bs[iO][:,:,iF][:]
        y_temp = reshape(y_temp, (imgp.imgL,imgp.imgL))
        ys[iO] = ys[iO] + y_temp
    end
    
    ys
    
end

