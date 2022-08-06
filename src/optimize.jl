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

